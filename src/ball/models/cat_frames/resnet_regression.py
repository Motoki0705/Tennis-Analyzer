import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50CoordRegression(nn.Module):
    """
    Pretrained ResNet-50 backbone + regression head for normalized 2D coordinate output.

    Args:
        in_channels (int): Number of input channels (e.g., 9 for 3 RGB frames concatenated).
        pretrained (bool): Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, in_channels: int = 9, pretrained: bool = True):
        super().__init__()
        # 1) Load pretrained ResNet-50
        backbone = models.resnet50(pretrained=pretrained)

        # 2) Adapt first conv to accept `input_channels` if different from 3
        if in_channels != 3:
            orig_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=(orig_conv.bias is not None),
            )
            # Initialize by averaging original weights and repeating
            with torch.no_grad():
                # orig_conv.weight.shape = [64, 3, 7, 7]
                avg_weight = orig_conv.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
                if orig_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
            backbone.conv1 = new_conv

        # 3) Remove classification head
        in_features = backbone.fc.in_features  # typically 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # 4) Regression head for (x, y) in [0,1]
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Sigmoid(),  # normalize to [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        returns: [B, 2] with x, y in [0,1]
        """
        features = self.backbone(x)  # [B, 2048, H', W']
        coords = self.reg_head(features)  # [B, 2]
        return coords


# Usage example
if __name__ == "__main__":
    model = ResNet50CoordRegression(in_channels=9, pretrained=True)
    dummy = torch.randn(4, 9, 360, 640)
    out = model(dummy)
    print(out.shape)  # -> torch.Size([4, 2])
    print(model)
