"""
Local Ball Classifier Model
16x16ピクセルパッチでボールの有無を判定する軽量CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LocalBallClassifier(nn.Module):
    """
    軽量CNN-based ローカルボール分類器
    
    入力: 16x16x3 RGB パッチ
    出力: ボール存在確率 (0-1)
    """
    
    def __init__(self, 
                 input_size: int = 16,
                 dropout_rate: float = 0.2,
                 use_attention: bool = True):
        """
        Args:
            input_size (int): 入力パッチサイズ
            dropout_rate (float): ドロップアウト率
            use_attention (bool): 空間アテンション使用フラグ
        """
        super().__init__()
        
        self.input_size = input_size
        self.use_attention = use_attention
        
        # Feature Extractor - 軽量設計
        self.features = nn.Sequential(
            # Block 1: 16x16 -> 8x8
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 8x8 -> 4x4  
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 4x4 -> 2x2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Spatial Attention (オプション)
        if self.use_attention:
            self.attention = SpatialAttention(128)
            
        # Classifier Head
        feature_dim = 128 * 2 * 2  # 2x2 feature maps
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),  # Binary classification
            nn.Sigmoid()
        )
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input patches [B, 3, H, W]
            
        Returns:
            torch.Tensor: Ball presence probability [B, 1]
        """
        # Feature extraction
        features = self.features(x)  # [B, 128, 2, 2]
        
        # Spatial attention
        if self.use_attention:
            features = self.attention(features)
            
        # Global pooling & classification
        features = features.view(features.size(0), -1)  # [B, 128*2*2]
        output = self.classifier(features)  # [B, 1]
        
        return output


class SpatialAttention(nn.Module):
    """空間アテンションモジュール"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention map
        attention = torch.sigmoid(self.conv(x))  # [B, 1, H, W]
        
        # Apply attention
        return x * attention


class EfficientLocalClassifier(nn.Module):
    """
    より効率的な軽量分類器（MobileNet-inspired）
    推論速度重視版
    """
    
    def __init__(self, input_size: int = 16):
        super().__init__()
        
        # Depthwise Separable Convolutions
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            self._make_dw_block(16, 32, 2),  # 16x16 -> 8x8
            self._make_dw_block(32, 64, 2),  # 8x8 -> 4x4
            self._make_dw_block(64, 128, 2), # 4x4 -> 2x2
        )
        
        # Global pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def _make_dw_block(self, in_ch: int, out_ch: int, stride: int):
        """Depthwise Separable Convolution Block"""
        return nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            
            # Pointwise conv
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.classifier(features)
        return output


def create_local_classifier(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    ローカル分類器のファクトリ関数
    
    Args:
        model_type (str): "standard" or "efficient"
        **kwargs: モデル固有のパラメータ
        
    Returns:
        nn.Module: 分類器インスタンス
    """
    if model_type == "standard":
        return LocalBallClassifier(**kwargs)
    elif model_type == "efficient":
        return EfficientLocalClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # モデル検証
    print("🔍 Local Ball Classifier Test")
    print("=" * 40)
    
    # Standard model
    model_std = create_local_classifier("standard")
    params_std = count_parameters(model_std)
    
    # Efficient model  
    model_eff = create_local_classifier("efficient")
    params_eff = count_parameters(model_eff)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 16, 16)
    
    print(f"Standard Model: {params_std:,} parameters")
    with torch.no_grad():
        output_std = model_std(test_input)
        print(f"  Output shape: {output_std.shape}")
        print(f"  Sample outputs: {output_std.squeeze()[:3].numpy()}")
        
    print(f"\nEfficient Model: {params_eff:,} parameters")
    with torch.no_grad():
        output_eff = model_eff(test_input)
        print(f"  Output shape: {output_eff.shape}")
        print(f"  Sample outputs: {output_eff.squeeze()[:3].numpy()}")
        
    print(f"\nParameter reduction: {(1 - params_eff/params_std)*100:.1f}%") 