import pytest
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import sys
import os

# プロジェクトルートをimportパスに追加（srcなどをimport可能にする）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# テストパラメータの候補
INPUT_TYPES = ["cat", "stack"]         # 連結 or スタック
OUTPUT_TYPES = ["last", "all"]         # 最後のフレームのみ or 全フレーム
DATASET_TYPES = ["coord", "heatmap"]   # 座標回帰 or ヒートマップ回帰
T_VALUES = [1, 3, 5]                    # 入力フレーム数

@pytest.mark.parametrize("input_type", INPUT_TYPES)
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
@pytest.mark.parametrize("dataset_type", DATASET_TYPES)
@pytest.mark.parametrize("T", T_VALUES)
def test_dataset_io(input_type: str, output_type: str, dataset_type: str, T: int):
    """
    DataModuleのI/O仕様（フレーム、ターゲット、可視性）を検証する。
    - frames: 入力画像テンソル
    - targets: 出力（座標 or ヒートマップ）
    - visibility: 各時刻のボール可視性
    """

    # hydra設定の上書き（YAMLを直接書き換えずに動的に調整）
    overrides = [
        f"litdatamodule.input_type={input_type}",
        f"litdatamodule.output_type={output_type}",
        f"litdatamodule.dataset_type={dataset_type}",
        f"litdatamodule.T={T}",
    ]

    with initialize(version_base="1.3", config_path="../../configs/test/ball", job_name="test_dataset_io_instance"):
        cfg = compose(config_name="config_dataset_test.yaml", overrides=overrides)

        try:
            # DataModuleを設定ファイルからインスタンス化
            datamodule = instantiate(cfg.litdatamodule)
            datamodule.prepare_data()
            datamodule.setup(stage="fit")
        except Exception as e:
            pytest.fail(f"Failed to instantiate or setup datamodule: {e}")

        try:
            train_loader = datamodule.train_dataloader()
            batch = next(iter(train_loader))
        except Exception as e:
            pytest.fail(f"Failed to get batch from train_loader: {e}")

        # frames, targets, visibility は DataModule の戻り値
        frames, targets, visibility = batch

        # ==== framesの形状確認 ====
        if input_type == "cat":
            # cat: [B, C*T, H, W]
            assert frames.dim() == 4
            assert frames.shape[1] == 3 * T, f"Expected channels: {3*T}, got {frames.shape[1]}"
        elif input_type == "stack":
            # stack: [B, T, C, H, W]
            assert frames.dim() == 5
            assert frames.shape[1] == T and frames.shape[2] == 3

        # ==== framesのサイズ（H×W) ====
        input_h, input_w = datamodule.input_size
        assert frames.shape[-2:] == (input_h, input_w), f"Expected frame size {(input_h, input_w)}, got {frames.shape[-2:]}"

        # ==== targetsの形状と内容確認 ====
        if dataset_type == "coord":
            # 座標形式: [B, 2] (last) or [B, T, 2] (all)
            if output_type == "last":
                assert targets.dim() == 2 and targets.shape[1] == 2
            else:
                assert targets.dim() == 3 and targets.shape[1] == T and targets.shape[2] == 2
        elif dataset_type == "heatmap":
            # ヒートマップ形式: [B, H, W] (last) or [B, T, H, W] (all)
            if output_type == "last":
                assert targets.dim() == 3
            else:
                assert targets.dim() == 4 and targets.shape[1] == T

            # ==== ヒートマップの最大値確認 ====
            # 可視時には max≈1.0、不可視時には max=0.0 を許容
            max_vals = (
                targets.view(targets.shape[0], -1).max(dim=-1)[0]
                if output_type == "last"
                else targets.flatten(2).max(dim=-1)[0]  # [B, T]
            )

            for i in range(visibility.shape[0]):  # バッチごとに確認
                vis = visibility[i] if output_type == "all" else visibility[i, 0]
                max_val = max_vals[i] if output_type == "last" else max_vals[i].max()
                if torch.any(vis > 0):
                    assert torch.isclose(max_val, torch.tensor(1.0), atol=0.1), f"Expected heatmap max ≈ 1.0, got {max_val}"
                else:
                    assert torch.isclose(max_val, torch.tensor(0.0), atol=0.1), f"Expected heatmap max ≈ 0.0, got {max_val}"

        # ==== visibilityの形状確認 ====
        if output_type == "last":
            assert visibility.shape[1] == 1
        else:
            assert visibility.shape[1] == T

        print(f"✅ Test passed: input_type={input_type}, output_type={output_type}, dataset_type={dataset_type}, T={T}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
