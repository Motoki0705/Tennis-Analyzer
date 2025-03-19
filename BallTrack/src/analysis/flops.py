import torch
from fvcore.nn import FlopCountAnalysis

from BallTrack.src.models.feature_extractors.segformer import SegFormerFeatureExtractor
from BallTrack.src.models.feature_extractors.mask2former import Mask2FormerFeatureExtractor

segformer = SegFormerFeatureExtractor()
mask2former = Mask2FormerFeatureExtractor()

inputs = torch.randn(1, 3, 512, 512)
flops = FlopCountAnalysis(mask2former, inputs)
print(f"FLOPs: {flops.total()/1e9:.2f} GFLOPs")
