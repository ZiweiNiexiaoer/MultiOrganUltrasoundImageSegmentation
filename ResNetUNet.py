import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1, decoder_channels=[256, 128, 64, 32, 16], 
                 pretrained_encoder="/root/autodl-tmp/resnet_weights/resnet18.pth", 
                 freeze_encoder=False):
        super().__init__()
        
        # ResNet18 encoder
        self.resnet = resnet18(weights=None)
        
        # Load pretrained encoder weights
        if pretrained_encoder:
            print(f"Loading pretrained encoder weights from {pretrained_encoder}")
            state_dict = torch.load(pretrained_encoder, map_location='cpu')
            
            # Handle possible key mismatches (e.g. from different PyTorch versions)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            
            self.resnet.load_state_dict(new_state_dict, strict=False)
            
            if freeze_encoder:
                for param in self.resnet.parameters():
                    param.requires_grad = False
                print("Encoder weights frozen")
        
        # Remove the original classification head
        self.encoder = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(512, decoder_channels[0], 1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # UNet decoder
        decoder_layers = []
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            decoder_layers.append(UNetDecoderBlock(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, num_classes, 1)
        )
        
        # Reconstruction decoder
        self.recon_decoder = nn.Sequential(
            UNetDecoderBlock(decoder_channels[-1], 16),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Original input size
        orig_size = x.shape[-2:]
        
        # Encoder forward pass
        x = self.encoder(x)
        
        # Project to decoder dimensions
        x = self.projection(x)
        
        # UNet decoder
        for decode_block in self.decoder:
            x = decode_block(x)
        
        # Segmentation output
        seg_out = self.seg_head(x)
        seg_out = F.interpolate(seg_out, size=orig_size, mode='bilinear', align_corners=False)
        
        # Reconstruction output
        recon_out = self.recon_decoder(x)
        recon_out = F.interpolate(recon_out, size=orig_size, mode='bilinear', align_corners=False)
        
        return seg_out, recon_out