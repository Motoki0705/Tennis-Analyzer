import torch
import torch.nn as nn

from src.models.conv_modules import ConvKernel_3_Stride_2_, InceptionModule


class ParentModelSimpleConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.convblock1 = ConvKernel_3_Stride_2_(in_channels, 32)
        self.convblock2 = ConvKernel_3_Stride_2_(32, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcblock = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fcblock(x)
        return x


class ParentModelInception(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.convblock1 = ConvKernel_3_Stride_2_(in_channels, 32)
        self.inception1 = InceptionModule(32, 64)
        self.convblock2 = ConvKernel_3_Stride_2_(64, 128)
        self.inception2 = InceptionModule(128, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.inception1(x)
        x = self.convblock2(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ParentModelMaxpool(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
