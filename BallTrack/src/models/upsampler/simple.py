import torch
import torch.nn as nn

from BallTrack.src.models.base import UpsamplerBase
from BallTrack.src.models.feature_extractors.segformer import SegFormerFeatureExtractor

class SimpleUpsampler(UpsamplerBase):
    def __init__(self, orig_channels, mask_channels, num_keypoints):
        super().__init__()

        self.upsmapler = nn.ModuleDict({
            'convtranspose1': nn.ConvTranspose2d(
                mask_channels,
                mask_channels // 2,
                kernel_size=(2, 2),
                stride=(2, 2)
                ),
            'batch_norm1': nn.BatchNorm2d(mask_channels // 2),
            'convtranspose2': nn.ConvTranspose2d(
                mask_channels // 2,
                num_keypoints,
                kernel_size=(2, 2),
                stride=(2, 2)
            ),
            'batch_norm2': nn.BatchNorm2d(num_keypoints) if num_keypoints > 1 else nn.Identity(),
            'channel_conform': nn.Conv2d(orig_channels, num_keypoints, kernel_size=(1, 1), stride=(1, 1)),
            'final_conv': nn.Conv2d(num_keypoints, num_keypoints, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        })

        self.res_param = nn.Parameter(torch.tensor([0.1],dtype=torch.float32))

    def forward(self, feature, orig_inputs):
        feature = self.upsmapler['convtranspose1'](feature)
        feature = self.upsmapler['batch_norm1'](feature)
        feature = self.upsmapler['convtranspose2'](feature)
        feature = self.upsmapler['batch_norm2'](feature)
        orig_inputs = self.upsmapler['channel_conform'](orig_inputs)
        output = self.upsmapler['final_conv'](feature + self.res_param * orig_inputs)
        return output
    
if __name__ == '__main__':
    segformer = SegFormerFeatureExtractor()
    upsampler = SimpleUpsampler(orig_channels=9, mask_channels=150, num_keypoints=32)
    inputs = torch.rand(2, 9, 512, 512)

    with torch.no_grad():
        feature = segformer(inputs)
        output = upsampler(feature, inputs)

    print(output.shape)


