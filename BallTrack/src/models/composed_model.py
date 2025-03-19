import torch
import torch.nn as nn

from BallTrack.src.models.feature_extractors.segformer import SegFormerFeatureExtractor
from BallTrack.src.models.feature_extractors.mask2former import Mask2FormerFeatureExtractor
from BallTrack.src.models.upsampler.simple import SimpleUpsampler

class ComposedModel(nn.Module):
    def __init__(self, orig_channels, num_keypoints, extractor=None, upsampler=None):
        super().__init__()

        # mask_channelsを初期化
        mask_channels = None

        # feature_extractorを設定
        if extractor == 'segformer':
            self.feature_extractor = SegFormerFeatureExtractor()
            mask_channels = 150
        
        elif extractor == 'mask2former':
            self.feature_extractor = Mask2FormerFeatureExtractor()
            mask_channels = 100
        
        # upsamplerを設定
        if mask_channels is None:
            print('mask_channels is not defined')
        
        if upsampler == 'simple':
            self.upsampler = SimpleUpsampler(
                orig_channels=orig_channels,
                mask_channels=mask_channels,
                num_keypoints=num_keypoints
                )

    def forward(self, orig_inputs):
        feature = self.feature_extractor(orig_inputs)
        logits = self.upsampler(feature, orig_inputs)

        return logits

if __name__ == '__main__':
    model = ComposedModel(orig_channels=3, num_keypoints=30, extractor='segformer', upsampler='simple')
    inputs = torch.rand(2, 3, 512, 512)

    with torch.no_grad():
        output = model(inputs)

    print(output.shape)