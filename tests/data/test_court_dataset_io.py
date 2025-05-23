import pytest
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import sys
import os
from unittest.mock import patch, MagicMock

# プロジェクトルートをimportパスに追加（srcなどをimport可能にする）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# テストパラメータの候補
INPUT_SIZES = [(224, 224), (256, 256)]  # 入力サイズ
HEATMAP_SIZES = [(224, 224), (64, 64)]  # ヒートマップサイズ
NUM_KEYPOINTS = [15]                    # キーポイント数
IS_EACH_KEYPOINT = [True, False]        # 各キーポイント用のヒートマップを生成するか

class DummyCourtDataset(torch.utils.data.Dataset):
    """テスト用のダミーCourtDataset"""
    def __init__(self, input_size, heatmap_size, num_keypoints, is_each_keypoint, size=10):
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.is_each_keypoint = is_each_keypoint
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 入力画像: [C, H, W]
        image = torch.rand(3, self.input_size[0], self.input_size[1])
        
        # ヒートマップ: [num_keypoints, H, W] または [1, H, W]
        channels = self.num_keypoints if self.is_each_keypoint else 1
        heatmap = torch.rand(channels, self.heatmap_size[0], self.heatmap_size[1])
        
        return image, heatmap

# ダミーの変換関数
def dummy_transform(img):
    return img

@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("heatmap_size", HEATMAP_SIZES)
@pytest.mark.parametrize("num_keypoints", NUM_KEYPOINTS)
@pytest.mark.parametrize("is_each_keypoint", IS_EACH_KEYPOINT)
def test_dataset_io(input_size, heatmap_size, num_keypoints, is_each_keypoint):
    """
    CourtDataModuleのI/O仕様（画像、ヒートマップ）を検証する。
    - frames: 入力画像テンソル
    - heatmaps: 出力ヒートマップ
    """

    # hydra設定の上書き（YAMLを直接書き換えずに動的に調整）
    overrides = [
        f"litdatamodule.input_size={list(input_size)}",
        f"litdatamodule.heatmap_size={list(heatmap_size)}",
        f"litdatamodule.default_keypoints={num_keypoints}",
        f"litdatamodule.is_each_keypoint={str(is_each_keypoint).lower()}",
    ]

    # prepare_transform関数をモックする
    with patch('src.court.arguments.prepare_transform.prepare_transform', return_value=dummy_transform):
        with initialize(version_base="1.3", config_path="../../configs/test/court", job_name="test_dataset_io_instance"):
            cfg = compose(config_name="config_dataset_test.yaml", overrides=overrides)

            # DataModuleをパッチしてダミーデータセットを返すようにする
            with patch('src.court.lit_datamodule.court_datamodule.CourtDataModule._prepare_dataset') as mock_prepare:
                # ダミーデータセットを作成
                dummy_dataset = DummyCourtDataset(
                    input_size=input_size,
                    heatmap_size=heatmap_size,
                    num_keypoints=num_keypoints,
                    is_each_keypoint=is_each_keypoint
                )
                mock_prepare.return_value = dummy_dataset
                
                # DataModuleをインスタンス化
                datamodule = instantiate(cfg.litdatamodule)
                datamodule.prepare_data()
                datamodule.setup(stage="fit")
                
                # DataLoaderからバッチを取得
                train_loader = datamodule.train_dataloader()
                batch = next(iter(train_loader))

            # frames, heatmaps は DataModule の戻り値
            frames, heatmaps = batch

            # ==== framesの形状確認 ====
            assert frames.dim() == 4, "Frames should be 4D: [B, C, H, W]"
            assert frames.shape[1] == 3, f"Expected 3 channels, got {frames.shape[1]}"
            assert frames.shape[2:] == input_size, f"Expected frame size {input_size}, got {frames.shape[2:]}"

            # ==== heatmapsの形状確認 ====
            if is_each_keypoint:
                # 各キーポイント用のヒートマップ: [B, num_keypoints, H, W]
                assert heatmaps.dim() == 4, "Heatmaps should be 4D: [B, num_keypoints, H, W]"
                assert heatmaps.shape[1] == num_keypoints, f"Expected {num_keypoints} keypoints, got {heatmaps.shape[1]}"
            else:
                # 単一チャネルヒートマップ: [B, 1, H, W]
                assert heatmaps.dim() == 4, "Heatmaps should be 4D: [B, 1, H, W]"
                assert heatmaps.shape[1] == 1, f"Expected 1 channel, got {heatmaps.shape[1]}"

            assert heatmaps.shape[2:] == heatmap_size, f"Expected heatmap size {heatmap_size}, got {heatmaps.shape[2:]}"

            # ==== ヒートマップの値範囲確認 ====
            assert torch.all(heatmaps >= 0), "Heatmap values should be non-negative"
            assert torch.all(heatmaps <= 1), "Heatmap values should be <= 1"

            print(f"✅ Test passed: input_size={input_size}, heatmap_size={heatmap_size}, "
                f"num_keypoints={num_keypoints}, is_each_keypoint={is_each_keypoint}")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 