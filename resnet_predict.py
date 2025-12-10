import os
import csv
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.ResNetUNet import ResNetUNet  # 确保与训练时相同的模型结构

# 配置参数
class Config:
    def __init__(self):
        # 模型路径 - 使用整个模型的最佳权重
        self.model_path = '/root/vitunet/result1/best_model.pth'
        
        # 测试集路径
        self.test_img_dir = '/root/SAM2-UNet/data_001/val/images'
        self.test_mask_dir = '/root/SAM2-UNet/data_001/val/masks'
        
        # 输出文件路径
        self.dice_csv_path = '/root/vitunet/predict_alldice/resnet_tune_dice.csv'
        self.metrics_csv_path = '/root/vitunet/result6_ResnetUNet/resnet_tune_metrics.csv'
        
        # 图像处理参数（需要与训练时一致）
        self.input_img_size = 224  # 根据训练配置修改
        self.crop_size = 224       # 根据训练配置修改
        
        # 设备设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 类别映射规则
        self.class_mapping = {
            'Breast': ['Breast_benign', 'Breast_luminal', 'Breast_malignant'],
            'BUSI': ['BUS-BRA', 'BUSI', 'BUSIS'],
            'CAMUS': ['CAMUS'],
            'Cardiac': ['Cardiac'],
            'DDTI': ['DDTI'],
            'Fetal_HC': ['Fetal_HC'],
            'KidneyUS': ['KidneyUS'],
            'Thyroid': ['Thyroid']
        }

# 使用概率值计算 Dice 系数（与训练损失一致）
def calculate_dice(pred, target, smooth=1e-5):
    """
    使用概率值计算 Dice 系数（与训练损失计算方式一致）
    """
    pred_prob = torch.sigmoid(pred)  # 应用 sigmoid 得到概率值
    intersection = (pred_prob * target).sum()
    union = pred_prob.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

# 预测函数
def predict(model, img_tensor):
    with torch.no_grad():
        seg_pred, _ = model(img_tensor.unsqueeze(0))
        return seg_pred.squeeze(0)  # 返回原始输出（未应用 sigmoid）

# 获取图像类别
def get_image_class(img_path, class_mapping):
    filename = os.path.basename(img_path)
    for class_name, prefixes in class_mapping.items():
        for prefix in prefixes:
            if filename.startswith(prefix):
                return class_name
    return 'Other'

# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 计算FLOPS
def calculate_flops(model, input_size=(1, 3, 224, 224)):
    from thop import profile
    input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return flops

# 计算FPS - 修复版本
def calculate_fps(model, input_size=(1, 3, 224, 224), num_runs=100, warmup=10):
    model.eval()
    input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
    
    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Timing - 使用torch.cuda.Event进行更精确的计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    end_event.record()
    
    # 等待所有CUDA操作完成
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    return num_runs / elapsed_time

def main():
    # 加载配置
    config = Config()
    
    # 创建模型
    model = ResNetUNet(
        num_classes=1,  # 二值分割
        decoder_channels=[256, 128, 64, 32, 16],  # 根据训练配置修改
        pretrained_encoder=False,  # 这里设置为False
        freeze_encoder=False
    ).to(config.device)
    
    # 加载训练好的权重
    print(f"加载模型权重: {config.model_path}")
    state_dict = torch.load(config.model_path, map_location=config.device)
    
    # 检查键匹配情况
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    model.eval()
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize(config.input_img_size),
        transforms.CenterCrop(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize(config.input_img_size),
        transforms.CenterCrop(config.crop_size),
        transforms.ToTensor()
    ])
    
    # 创建测试数据集
    class TestDataset(Dataset):
        def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            self.transform = transform
            self.mask_transform = mask_transform
            self.img_paths = []
            
            for file in os.listdir(img_dir):
                if file.endswith('.png'):
                    img_path = os.path.join(img_dir, file)
                    mask_path = os.path.join(mask_dir, file)
                    if os.path.exists(mask_path):
                        self.img_paths.append(img_path)
            
            print(f"找到 {len(self.img_paths)} 个测试图像")
        
        def __len__(self):
            return len(self.img_paths)
        
        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
            
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            if self.transform:
                img = self.transform(img)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            
            return img, mask, img_path
    
    # 创建测试数据集和加载器
    test_dataset = TestDataset(
        config.test_img_dir,
        config.test_mask_dir,
        transform=transform,
        mask_transform=mask_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化类别统计
    class_stats = {class_name: {'dice_sum': 0.0, 'count': 0} 
                  for class_name in config.class_mapping.keys()}
    class_stats['Other'] = {'dice_sum': 0.0, 'count': 0}
    total_dice_sum = 0.0
    
    # 进行预测并计算Dice系数
    print("开始预测并计算Dice系数...")
    for images, masks, img_paths in tqdm(test_loader, desc="预测"):
        images = images.to(config.device)
        masks = masks.to(config.device)
        
        # 预测（返回原始输出）
        pred_masks = predict(model, images[0])
        
        # 计算Dice系数（使用概率值）
        dice_score = calculate_dice(pred_masks, masks[0])
        
        # 获取图像类别
        img_class = get_image_class(img_paths[0], config.class_mapping)
        
        # 更新统计
        class_stats[img_class]['dice_sum'] += dice_score
        class_stats[img_class]['count'] += 1
        total_dice_sum += dice_score
    
    # 计算平均Dice系数
    total_count = len(test_loader.dataset)
    overall_dice = total_dice_sum / total_count if total_count > 0 else 0.0
    
    # 保存类别Dice系数到CSV
    with open(config.dice_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['类别', '图像数量', '平均Dice系数'])
        
        for class_name, stats in class_stats.items():
            if stats['count'] > 0:
                avg_dice = stats['dice_sum'] / stats['count']
                writer.writerow([class_name, stats['count'], f"{avg_dice:.4f}"])
        
        writer.writerow(['总体', total_count, f"{overall_dice:.4f}"])
    
    print(f"类别Dice系数已保存到: {config.dice_csv_path}")
    
    # 计算模型指标
    print("\n计算模型指标...")
    
    # 参数量
    params = count_parameters(model)
    print(f"参数量: {params}")
    
    # FLOPS
    try:
        # 使用临时模型副本计算FLOPS
        from copy import deepcopy
        model_copy = deepcopy(model)
        flops = calculate_flops(model_copy, input_size=(1, 3, config.crop_size, config.crop_size))
        print(f"FLOPS: {flops}")
        del model_copy
    except ImportError:
        print("thop库未安装，无法计算FLOPS")
        flops = 0
    except Exception as e:
        print(f"计算FLOPS时出错: {str(e)}")
        flops = 0
    
    # FPS
    try:
        fps = calculate_fps(model, input_size=(1, 3, config.crop_size, config.crop_size))
        print(f"FPS: {fps:.2f}")
    except Exception as e:
        print(f"计算FPS时出错: {str(e)}")
        fps = 0.0
    
    # 保存模型指标到CSV
    with open(config.metrics_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['指标', '值'])
        writer.writerow(['参数量', params])
        writer.writerow(['FLOPS', flops])
        writer.writerow(['FPS', f"{fps:.2f}"])
    
    print(f"模型指标已保存到: {config.metrics_csv_path}")

if __name__ == '__main__':
    main()