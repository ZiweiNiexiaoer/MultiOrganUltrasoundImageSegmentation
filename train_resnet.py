import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from utils.ResNetUNet import ResNetUNet  # 使用修改后的模型
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm
import time
import csv
import torch.multiprocessing
import seaborn as sns
from thop import profile, clever_format  # 用于计算FLOPS和Parameters
import torch.backends.cudnn as cudnn
torch.multiprocessing.set_sharing_strategy('file_system')


def safe_float_convert(value):
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(value.replace('e-', 'e-').replace('e+', 'e+'))
    return float(value)

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        self.img_paths = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    rel_path = os.path.relpath(img_path, img_dir)
                    mask_path = os.path.join(mask_dir, rel_path)
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.img_paths.append(img_path)
                    else:
                        print(f"Warning: Missing pair - Image: {img_path} or Mask: {mask_path}")

        if len(self.img_paths) == 0:
            raise RuntimeError("No valid image-mask pairs found in the dataset")
        
        print(f"\n✅ 数据集加载完成:")
        print(f"  图像目录: {img_dir}")
        print(f"  掩码目录: {mask_dir}")
        print(f"  有效图像-掩码对数: {len(self.img_paths)}")


    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            rel_path = os.path.relpath(img_path, self.img_dir)
            mask_path = os.path.join(self.mask_dir, rel_path)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            seed = np.random.randint(2147483647)
            if self.transform:
                torch.manual_seed(seed)
                img = self.transform(img)
            
            if self.mask_transform:
                torch.manual_seed(seed)
                mask = self.mask_transform(mask)
            
            return img, mask, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            raise

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

class Metrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_pixels = 0
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
        self.fpr = None
        self.tpr = None
        self.thresholds = None
    
    def update(self, pred, target):
        pred_prob = torch.sigmoid(pred).float()
        pred_bin = (pred_prob > 0.5).float()
        target = (target > 0.5).float()
        
        tn, fp, fn, tp = confusion_matrix(
            target.view(-1).cpu().numpy(), 
            pred_bin.view(-1).cpu().numpy(),
            labels=[0, 1]
        ).ravel()
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.total_pixels += (tp + fp + fn + tn)
        
        self.all_preds.extend(pred_bin.view(-1).cpu().numpy())
        self.all_targets.extend(target.view(-1).cpu().numpy())
        self.all_probs.extend(pred_prob.view(-1).cpu().numpy())
    
    def compute_roc_curve(self):
        """计算ROC曲线数据"""
        if len(self.all_probs) == 0 or len(self.all_targets) == 0:
            return None, None, None
        self.fpr, self.tpr, self.thresholds = roc_curve(self.all_targets, self.all_probs)
        return self.fpr, self.tpr, self.thresholds
    
    def get_confusion_matrix(self, normalized=True):
        cm = np.array([[self.tn, self.fp], 
                       [self.fn, self.tp]])
        if normalized and self.total_pixels > 0:
            return cm / self.total_pixels * 100
        return cm
    
    def iou(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-10)
    
    def dice(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-10)
    
    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-10)
    
    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-10)
    
    def map(self):
        if len(self.all_preds) == 0 or len(self.all_targets) == 0:
            return 0.0
        return average_precision_score(self.all_targets, self.all_probs)
    
    def auc_roc(self):
        if len(self.all_probs) == 0 or len(self.all_targets) == 0:
            return 0.0
        return roc_auc_score(self.all_targets, self.all_probs)

def save_confusion_matrix(cm, epoch, save_dir, normalized=True):
    plt.figure(figsize=(10, 8))
    
    fmt = '.2f' if normalized else 'd'
    suffix = '%' if normalized else ''
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['Background', 'Foreground'],
                yticklabels=['Background', 'Foreground'])
    
    plt.xlabel(f'Predicted ({suffix})')
    plt.ylabel(f'Actual ({suffix})')
    title = f'Confusion Matrix (Epoch {epoch})'
    if normalized:
        title += ' (Normalized)'
    plt.title(title)
    
    os.makedirs(f'{save_dir}/confusion_matrix', exist_ok=True)
    filename = f'latest_cm{"_norm" if normalized else ""}.png'
    plt.savefig(f'{save_dir}/confusion_matrix/{filename}')
    plt.close()

def save_roc_curve(fpr, tpr, auc, thresholds, epoch, save_dir):
    """保存ROC曲线图，包含AUC值和最佳阈值点"""
    plt.figure(figsize=(10, 8))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 标记最佳阈值点
    if len(fpr) > 0 and len(tpr) > 0:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                   marker='o', color='red', 
                   label=f'Optimal threshold = {optimal_threshold:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (Epoch {epoch})')
    plt.legend(loc="lower right")
    
    os.makedirs(f'{save_dir}/roc_curves', exist_ok=True)
    plt.savefig(f'{save_dir}/roc_curves/latest_roc.png', 
               bbox_inches='tight', dpi=300)
    plt.close()
    
    # 同时保存ROC数据为CSV文件
    roc_data = np.column_stack((fpr, tpr, thresholds))
    np.savetxt(f'{save_dir}/roc_curves/latest_roc_data.csv', 
              roc_data, delimiter=',',
              header='fpr,tpr,threshold', comments='')

def save_visualization(img, true_mask, pred_mask, save_path):
    """保存输入图像、真实掩码和预测结果的叠加可视化"""
    try:
        # 处理输入图像（反归一化）
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        
        # 处理掩码
        true_mask = true_mask.cpu().squeeze().numpy()
        pred_mask = pred_mask.cpu().squeeze().numpy() > 0.5
        
        # 创建叠加效果
        overlay = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        overlay[pred_mask & (true_mask > 0.5)] = [0, 255, 0]    # 真阳性-绿色
        overlay[pred_mask & (true_mask <= 0.5)] = [255, 0, 0]  # 假阳性-红色
        overlay[~pred_mask & (true_mask > 0.5)] = [0, 0, 255]  # 假阴性-蓝色
        
        # 创建图像
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].set_title('Input')
        ax[0].axis('off')
        
        ax[1].imshow(true_mask, cmap='gray')
        ax[1].set_title('True Mask')
        ax[1].axis('off')
        
        ax[2].imshow(overlay)
        ax[2].set_title('Prediction Overlay')
        ax[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 验证保存结果
        if not os.path.exists(save_path):
            raise Exception(f"文件未被创建: {save_path}")
        if os.path.getsize(save_path) < 1024:  # 小于1KB可能是空白文件
            raise Exception(f"文件过小，可能保存失败: {os.path.getsize(save_path)} bytes")
            
        return True
        
    except Exception as e:
        print(f"save_visualization函数内部错误: {str(e)}")
        return False

def get_dataset_category(img_path):
    """根据图像路径确定数据集类别"""
    filename = os.path.basename(img_path)
    
    # 定义类别映射规则
    category_map = {
        "Breast": ["Breast_benign", "Breast_luminal", "Breast_malignant"],
        "BUSI": ["BUS-BRA", "BUSI", "BUSIS"],
        "CAMUS": ["CAMUS"],
        "Cardiac": ["Cardiac"],
        "DDTI": ["DDTI"],
        "Fetal_HC": ["Fetal_HC"],
        "KidneyUS": ["KidneyUS"],
        "Thyroid": ["Thyroid"]
    }
    
    # 检查文件名是否包含任何类别前缀
    for category, prefixes in category_map.items():
        for prefix in prefixes:
            if filename.startswith(prefix):
                return category
    
    # 如果未匹配到任何类别，返回"Unknown"
    return "Unknown"

def compute_model_stats(model, input_size=(1, 3, 256, 256), device='cuda'):
    """计算模型的FLOPS、Parameters和FPS"""
    model.eval()
    model.to(device)
    
    # 创建临时模型包装器，只返回分割输出
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            seg_pred, _ = self.model(x)  # 只返回分割输出
            return seg_pred
    
    wrapped_model = ModelWrapper(model).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算FLOPS
    input_tensor = torch.randn(input_size).to(device)
    
    # 使用thop的profile计算MACs和参数量
    macs, params = profile(wrapped_model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2  # FLOPS = MACs * 2
    
    # 计算FPS
    warmup = 10
    repeats = 50
    total_time = 0
    
    # 预热
    for _ in range(warmup):
        _ = wrapped_model(input_tensor)
    
    # 实际计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(repeats):
            start_event.record()
            _ = wrapped_model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)
    
    avg_time = total_time / repeats
    fps = 1000 / avg_time  # 每秒处理的图像数
    
    return {
        'params': total_params,
        'flops': flops,
        'fps': fps
    }
    
def validate_full_dataset(model, val_loader, device, criterion_seg, criterion_recon, epoch, save_dir, save_images=False):
    model.eval()
    val_loss = 0.0
    metrics = Metrics()
    
    # 为每个数据集类别创建单独的Metrics对象
    category_metrics = {}
    
    val_visuals_dir = os.path.join(save_dir, 'val_visuals')
    os.makedirs(val_visuals_dir, exist_ok=True)
    print(f"validate_full_dataset: 可视化保存目录: {val_visuals_dir}")
    
    # 计算模型统计信息（只计算一次）
    if not hasattr(validate_full_dataset, 'model_stats_computed'):
        print("\n===== 计算模型统计信息 =====")
        model_stats = compute_model_stats(model, input_size=(1, 3, 256, 256), device=device)
        print(f"参数量: {model_stats['params'] / 1e6:.2f} M")
        print(f"FLOPS: {model_stats['flops'] / 1e9:.2f} G")
        print(f"FPS: {model_stats['fps']:.2f}")
        validate_full_dataset.model_stats = model_stats
        validate_full_dataset.model_stats_computed = True
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Full Validation', leave=False)
        for batch_idx, (images, masks, img_paths) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            seg_pred, recon_pred = model(images)
            seg_loss = criterion_seg(seg_pred, masks)
            rec_loss = 0.1 * criterion_recon(recon_pred, images)
            loss = seg_loss + rec_loss
            
            val_loss += loss.item() * images.size(0)
            metrics.update(seg_pred, masks)
            
            # 更新每个类别的Metrics
            for i in range(images.size(0)):
                img_path = img_paths[i]
                category = get_dataset_category(img_path)
                
                if category not in category_metrics:
                    category_metrics[category] = Metrics()
                
                # 提取单个样本的预测和掩码
                single_pred = seg_pred[i].unsqueeze(0)
                single_mask = masks[i].unsqueeze(0)
                category_metrics[category].update(single_pred, single_mask)
            
            # 保存验证图像
            if save_images and (epoch % 50 == 0):
                print(f"validate_full_dataset: 满足保存条件，epoch={epoch}, batch={batch_idx}")
                for i in range(min(2, images.size(0))):  # 限制保存数量
                    img_name = os.path.basename(img_paths[i])
                    save_path = os.path.join(val_visuals_dir, f'fullval_epoch_{epoch}_batch_{batch_idx}_{i}_{img_name}')
                    print(f"validate_full_dataset: 尝试保存到 {save_path}")
                    
                    if save_visualization(images[i], masks[i], seg_pred[i], save_path):
                        print(f"validate_full_dataset: 保存成功 {save_path}")
                    else:
                        print(f"validate_full_dataset: 保存失败 {save_path}")
            
            pbar.set_postfix({'dice': f'{metrics.dice():.4f}'})
    
    val_loss /= len(val_loader.dataset)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = metrics.compute_roc_curve()
    auc = metrics.auc_roc()
    
    # 保存每个类别的Dice系数到CSV
    category_dice_dir = os.path.join(save_dir, 'category_dice')
    os.makedirs(category_dice_dir, exist_ok=True)
    category_csv_path = os.path.join(category_dice_dir, f'category_dice_epoch_{epoch}.csv')
    
    with open(category_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Category', 'Dice', 'Num_Samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for category, cat_metrics in category_metrics.items():
            dice_score = cat_metrics.dice()
            # 估算样本数：总像素数除以图像大小（假设图像大小为256x256）
            num_samples = cat_metrics.total_pixels // (256 * 256)
            writer.writerow({
                'Category': category,
                'Dice': f'{dice_score:.4f}',
                'Num_Samples': num_samples
            })
    
    print(f"✅ 已保存类别Dice系数到: {category_csv_path}")
    
    return val_loss, metrics, (fpr, tpr, thresholds, auc), category_metrics

def plot_training_curves(logs, save_dir):
    """绘制并保存训练曲线，包括损失和所有指标"""
    plt.figure(figsize=(15, 15))
    
    # 损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(logs['epoch'], logs['train_loss'], label='Train Loss')
    plt.plot(logs['epoch'], logs['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Dice和IoU曲线
    plt.subplot(3, 2, 2)
    plt.plot(logs['epoch'], logs['dice'], label='Dice Score')
    plt.plot(logs['epoch'], logs['iou'], label='IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Dice and IoU Scores')
    plt.legend()
    plt.grid(True)
    
    # 精确率和召回率曲线
    plt.subplot(3, 2, 3)
    plt.plot(logs['epoch'], logs['precision'], label='Precision')
    plt.plot(logs['epoch'], logs['recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision and Recall')
    plt.legend()
    plt.grid(True)
    
    # mAP曲线
    plt.subplot(3, 2, 4)
    plt.plot(logs['epoch'], logs['map'], label='mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.legend()
    plt.grid(True)
    
    # AUC-ROC曲线
    plt.subplot(3, 2, 5)
    plt.plot(logs['epoch'], logs['auc_roc'], label='AUC-ROC', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.title('AUC-ROC Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(logs['epoch'], logs['train_loss'], label='Train Loss')
    plt.plot(logs['epoch'], logs['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_best_model_metrics(best_metrics, save_dir):
    """保存最佳模型的指标"""
    with open(os.path.join(save_dir, 'best_model_metrics.txt'), 'w') as f:
        f.write(f"Best Model Metrics:\n")
        f.write(f"Dice Score: {best_metrics['dice']:.4f}\n")
        f.write(f"IoU: {best_metrics['iou']:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"mAP: {best_metrics['map']:.4f}\n")
        f.write(f"AUC-ROC: {best_metrics['auc_roc']:.4f}\n")
        f.write(f"Confusion Matrix (Counts):\n")
        f.write(f"TN: {best_metrics['tn']}, FP: {best_metrics['fp']}\n")
        f.write(f"FN: {best_metrics['fn']}, TP: {best_metrics['tp']}\n")
        f.write(f"\nModel Statistics:\n")
        f.write(f"Parameters: {best_metrics['params'] / 1e6:.2f} M\n")
        f.write(f"FLOPS: {best_metrics['flops'] / 1e9:.2f} G\n")
        f.write(f"FPS: {best_metrics['fps']:.2f}\n")
        
        # 添加类别Dice系数
        if 'category_metrics' in best_metrics:
            f.write(f"\nCategory Dice Scores:\n")
            for category, metrics in best_metrics['category_metrics'].items():
                dice_score = metrics.dice()
                f.write(f"{category}: {dice_score:.4f}\n")

def train_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True  # 启用cudnn基准测试以优化性能
    
    # 加载配置
    config_path = '/root/vitunet/config_resnet.yaml'
    print(f"加载配置文件: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 解析配置
    model_config = {
        'vit_type': str(config['model']['vit_type']),
        'decoder_channels': [int(x) for x in config['model']['decoder_channels']],
        'num_classes': int(config['model']['num_classes']),
        'vit_img_size': int(config['model']['vit_img_size']),
        'pretrained_encoder': str(config['model']['pretrained_encoder']),
        'freeze_encoder': bool(config['model']['freeze_encoder'])
    }
    
    data_config = {
        'train_img_dir': str(config['data']['train_img_dir']),
        'train_mask_dir': str(config['data']['train_mask_dir']),
        'val_img_dir': str(config['data']['val_img_dir']),
        'val_mask_dir': str(config['data']['val_mask_dir']),
        'input_img_size': int(config['data']['input_img_size']),
        'crop_size': int(config['data']['crop_size']),
        'batch_size': int(config['data']['batch_size']),
        'num_workers': int(config['data']['num_workers'])
    }
    
    train_config = {
        'device': str(config['training']['device']),
        'learning_rate': safe_float_convert(config['training']['learning_rate']),
        'scheduler': str(config['training']['scheduler']),
        'T_max': int(config['training']['T_max']),
        'eta_min': safe_float_convert(config['training']['eta_min']),
        'num_epochs': int(config['training']['num_epochs']),
        'save_dir': str(config['training']['save_dir']),
        'save_period': int(config['training']['save_period']),
        'full_val_interval': int(config['training']['full_val_interval']),
        'amp': bool(config['training']['amp']),
        'save_val_images': bool(config['training']['save_val_images'])
    }
    
    # 关键配置检查
    print("\n===== 配置验证 =====")
    print(f"save_val_images: {train_config['save_val_images']} (类型: {type(train_config['save_val_images'])})")
    print(f"full_val_interval: {train_config['full_val_interval']}")
    print(f"save_dir: {train_config['save_dir']}")
    print(f"save_period (检查点保存间隔): {train_config['save_period']}")
    
    # 初始化保存目录
    save_dir = train_config['save_dir']
    val_visuals_dir = os.path.join(save_dir, 'val_visuals')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'encoder_weights'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'roc_curves'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'category_dice'), exist_ok=True)  # 新增类别Dice保存目录
    os.makedirs(val_visuals_dir, exist_ok=True)
    
    # 目录写入测试
    print("\n===== 目录权限测试 =====")
    print(f"验证可视化目录: {val_visuals_dir}")
    print(f"目录是否存在: {os.path.exists(val_visuals_dir)}")
    
    try:
        test_file = os.path.join(val_visuals_dir, 'test_write_permission.txt')
        with open(test_file, 'w') as f:
            f.write("测试写入权限成功")
        os.remove(test_file)
        print(f"✅ 目录写入权限测试通过")
    except Exception as e:
        print(f"❌ 目录写入权限测试失败: {str(e)}")
        print("请检查目录权限或路径是否正确")
    
    # 设备检查
    print("\n===== 设备信息 =====")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    device = torch.device(train_config['device'])
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ResNetUNet(
        num_classes=model_config['num_classes'],
        decoder_channels=model_config['decoder_channels'],
        pretrained_encoder=model_config['pretrained_encoder'],
        freeze_encoder=model_config['freeze_encoder']
    ).to(device)
    
    # 损失函数和优化器
    criterion_seg = CombinedLoss(alpha=0.5)
    criterion_recon = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=train_config['learning_rate'], 
                           weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=train_config['T_max'], 
                                 eta_min=train_config['eta_min'])
    scaler = GradScaler(init_scale=8192.0, enabled=train_config['amp'])
    best_dice = 0.0
    best_metrics = None
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize(data_config['input_img_size']),
        transforms.RandomCrop(data_config['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(data_config['input_img_size']),
        transforms.RandomCrop(data_config['crop_size']),
        transforms.ToTensor()
    ])
    
    # 数据集和加载器
    print("\n===== 数据集加载 =====")
    train_dataset = SegmentationDataset(
        img_dir=data_config['train_img_dir'],
        mask_dir=data_config['train_mask_dir'],
        transform=transform,
        mask_transform=mask_transform
    )
    
    val_dataset = SegmentationDataset(
        img_dir=data_config['val_img_dir'],
        mask_dir=data_config['val_mask_dir'],
        transform=transform,
        mask_transform=mask_transform
    )
    
    loader_config = {
        'batch_size': data_config['batch_size'],
        'num_workers': data_config['num_workers'],
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'timeout': 60
    }
    
    train_loader = DataLoader(train_dataset, **loader_config, shuffle=True)
    val_loader = DataLoader(val_dataset, **loader_config, shuffle=False)
    
    # 日志初始化
    logs = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'dice': [], 'iou': [], 'precision': [], 'recall': [], 'map': [], 'auc_roc': [],
        'lr': [], 'tn': [], 'fp': [], 'fn': [], 'tp': []
    }
    
    # 存储所有ROC数据用于最终保存
    all_roc_data = {
        'fpr': [], 'tpr': [], 'thresholds': [], 'auc': []
    }
    
    # 训练循环
    print("\n===== 开始训练 =====")
    epochs_pbar = tqdm(
        range(1, train_config['num_epochs'] + 1),
        desc="Total Training",
        position=0,
        mininterval=1.0,
        dynamic_ncols=True
    )

    for epoch in epochs_pbar:
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        processed_samples = 0

        batch_pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}',
            position=1,
            leave=False,
            mininterval=0.5
        )

        for images, masks, _ in batch_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=train_config['amp']):
                seg_pred, recon_pred = model(images)
                seg_loss = criterion_seg(seg_pred, masks)
                rec_loss = 0.1 * criterion_recon(recon_pred, images)
                loss = seg_loss + rec_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            processed_samples += images.size(0)
            
            batch_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if processed_samples % 100 == 0:
                torch.cuda.empty_cache()

        batch_pbar.close()
        train_loss /= processed_samples
        
        epochs_pbar.set_postfix({
            'epoch_loss': f'{train_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # 每个epoch都执行验证
        print(f"\n===== Epoch {epoch} 验证阶段 =====")
        
        # 再次确认保存目录
        os.makedirs(val_visuals_dir, exist_ok=True)
        
        # 调用验证函数
        val_loss, val_metrics, roc_data, category_metrics = validate_full_dataset(
            model, val_loader, device, 
            criterion_seg, criterion_recon, 
            epoch, save_dir, train_config['save_val_images']
        )
        
        fpr, tpr, thresholds, auc = roc_data
        
        # 记录ROC数据用于最终保存
        if fpr is not None and tpr is not None:
            all_roc_data['fpr'].append(fpr)
            all_roc_data['tpr'].append(tpr)
            all_roc_data['thresholds'].append(thresholds)
            all_roc_data['auc'].append(auc)
        
        # 保存指标（每个epoch都会记录）
        logs['epoch'].append(epoch)
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['dice'].append(val_metrics.dice())
        logs['iou'].append(val_metrics.iou())
        logs['precision'].append(val_metrics.precision())
        logs['recall'].append(val_metrics.recall())
        logs['map'].append(val_metrics.map())
        logs['auc_roc'].append(auc)
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        logs['tn'].append(val_metrics.tn)
        logs['fp'].append(val_metrics.fp)
        logs['fn'].append(val_metrics.fn)
        logs['tp'].append(val_metrics.tp)
        
        # 保存混淆矩阵（每个epoch）
        save_confusion_matrix(val_metrics.get_confusion_matrix(normalized=False), 
                             epoch, save_dir, normalized=False)
        save_confusion_matrix(val_metrics.get_confusion_matrix(normalized=True), 
                             epoch, save_dir, normalized=True)
        
        # 保存ROC曲线（每个epoch）
        if fpr is not None and tpr is not None:
            save_roc_curve(fpr, tpr, auc, thresholds, epoch, save_dir)
        
        # 保存最佳模型（当当前Dice分数超过历史最佳时）
        if val_metrics.dice() > best_dice:
            best_dice = val_metrics.dice()
            best_metrics = {
                'dice': val_metrics.dice(),
                'iou': val_metrics.iou(),
                'precision': val_metrics.precision(),
                'recall': val_metrics.recall(),
                'map': val_metrics.map(),
                'auc_roc': auc,
                'tn': val_metrics.tn,
                'fp': val_metrics.fp,
                'fn': val_metrics.fn,
                'tp': val_metrics.tp,
                'params': validate_full_dataset.model_stats['params'],
                'flops': validate_full_dataset.model_stats['flops'],
                'fps': validate_full_dataset.model_stats['fps'],
                'category_metrics': category_metrics  # 保存类别指标
            }
            
           # 确保目录存在
            os.makedirs(f'{save_dir}/encoder_weights', exist_ok=True)

            # 保存完整模型
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')

            # 保存编码器部分
            torch.save(model.resnet.state_dict(), f'{save_dir}/encoder_weights/best_resnet_full.pth')
            torch.save(model.encoder.state_dict(), f'{save_dir}/encoder_weights/best_encoder.pth')
                        
            # 保存最佳模型指标
            save_best_model_metrics(best_metrics, save_dir)
            print(f"✅ 保存新的最佳模型，Dice分数: {best_dice:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存模型检查点（按照save_period配置的间隔）
        if epoch % train_config['save_period'] == 0:
            # 确保checkpoints目录存在
            checkpoint_dir = os.path.join(save_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)  # 关键修改：自动创建目录
            
            # 保存模型
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"📌 检查点已保存到: {checkpoint_path}")
            
        # 记录日志
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} 完成，耗时 {epoch_time:.1f}s")
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f} | Dice: {logs['dice'][-1]:.4f}")
        print(f"AUC-ROC: {logs['auc_roc'][-1]:.4f}")
        
        # 打印类别Dice系数
        print("\n数据集类别 Dice 系数:")
        for category, metrics in category_metrics.items():
            print(f"  {category}: {metrics.dice():.4f}")
        
        # 保存训练日志（每个epoch）
        with open(f'{save_dir}/training_logs.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'dice', 'iou', 
                            'precision', 'recall', 'map', 'auc_roc', 'lr', 'tn', 'fp', 'fn', 'tp'])
            for i in range(len(logs['epoch'])):
                writer.writerow([
                    logs['epoch'][i],
                    f"{logs['train_loss'][i]:.4f}",
                    f"{logs['val_loss'][i]:.4f}",
                    f"{logs['dice'][i]:.4f}",
                    f"{logs['iou'][i]:.4f}",
                    f"{logs['precision'][i]:.4f}",
                    f"{logs['recall'][i]:.4f}",
                    f"{logs['map'][i]:.4f}",
                    f"{logs['auc_roc'][i]:.4f}",
                    f"{logs['lr'][i]:.2e}",
                    int(logs['tn'][i]),
                    int(logs['fp'][i]),
                    int(logs['fn'][i]),
                    int(logs['tp'][i])
                ])
        
        # 每个epoch都绘制训练曲线
        plot_training_curves(logs, save_dir)
    
    # 训练结束后保存ROC曲线和数据
    print("\n===== 训练完成，保存ROC曲线和数据 =====")
    os.makedirs(f'{save_dir}/roc_curves', exist_ok=True)
    
    # 保存最佳epoch的ROC曲线
    if len(all_roc_data['auc']) > 0:
        best_epoch = np.argmax(all_roc_data['auc'])
        best_fpr = all_roc_data['fpr'][best_epoch]
        best_tpr = all_roc_data['tpr'][best_epoch]
        best_thresholds = all_roc_data['thresholds'][best_epoch]
        best_auc = all_roc_data['auc'][best_epoch]
        
        plt.figure(figsize=(10, 8))
        plt.plot(best_fpr, best_tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {best_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        if len(best_fpr) > 0 and len(best_tpr) > 0:
            optimal_idx = np.argmax(best_tpr - best_fpr)
            optimal_threshold = best_thresholds[optimal_idx]
            plt.scatter(best_fpr[optimal_idx], best_tpr[optimal_idx], 
                       marker='o', color='red', 
                       label=f'Optimal threshold = {optimal_threshold:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Best ROC Curve (Epoch {best_epoch+1})')
        plt.legend(loc="lower right")
        plt.savefig(f'{save_dir}/roc_curves/best_roc_curve.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存最佳ROC数据
        roc_data = np.column_stack((best_fpr, best_tpr, best_thresholds))
        np.savetxt(f'{save_dir}/roc_curves/best_roc_data.csv', 
                  roc_data, delimiter=',',
                  header='fpr,tpr,threshold', comments='')
        
        print(f"✅ 保存最佳ROC曲线 (Epoch {best_epoch+1}, AUC={best_auc:.4f})")
    
    # 保存最终模型和曲线
    torch.save(model.state_dict(), f'{save_dir}/final_model.pth')
    plot_training_curves(logs, save_dir)
    
    print(f"\n训练完成! 结果保存到 {save_dir}")
    print(f"最佳模型Dice分数: {best_dice:.4f}")
    print(f"最佳模型AUC-ROC分数: {best_metrics['auc_roc']:.4f}")
    print(f"训练曲线和指标已保存到: {save_dir}/training_curves.png")
    print(f"损失曲线已保存到: {save_dir}/loss_curves.png")
    print(f"最佳模型指标已保存到: {save_dir}/best_model_metrics.txt")
    print(f"模型复杂度信息已保存到: {save_dir}/best_model_metrics.txt")
    print(f"各类别Dice系数已保存到: {save_dir}/category_dice/ 目录")


if __name__ == '__main__':
    train_model()