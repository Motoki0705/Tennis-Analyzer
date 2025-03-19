import torch.nn as nn
import abc

class FeatureExtractorBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        """
        画像テンソル x を受け取り、特徴量（[B, C_feat, H_feat, W_feat]）を返す
        """
        pass

class UpsamplerBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, features, orig_inputs):
        """
        特徴量 (例：[B, C_feat, H_feat, W_feat]), オリジナルインプット (例: [B, C, H, W])
        アップサンプルしたヒートマップ ([B, num_keypoints, H_img, W_img]) を返す
        """
        pass