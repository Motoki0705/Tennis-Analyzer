# __init__.py

"""
WASB-SBDT: Weakly-Supervised Ball Detection and Tracking

このパッケージは、テニスボールの検出と追跡のための包括的なツールキットです。
利用者のニーズに合わせて、3つのレベルのAPIを提供します。

========================================
🚀 高レベルAPI (for End-Users)
========================================
動画ファイルを入力し、トラッキング結果を動画とCSVに出力する、すぐに使えるパイプライン。
マルチスレッドで処理が高速化されています。

基本的な使用方法：
```python
from third_party.WASB_SBDT import MultithreadedTennisTracker
import argparse

# コマンドライン引数と同様のオブジェクトを作成
args = argparse.Namespace(
    video="path/to/your/video.mp4",
    output="output.mp4",
    results_csv="results.csv",
    model_path=None,  # デフォルトモデルを使用
    device="auto",
    batch_size=8
)

# パイプラインを実行
pipeline = MultithreadedTennisTracker(args)
pipeline.run()
```

========================================
🛠️ 中レベルAPI (for Custom Pipelines)
========================================
パイプラインの各ステージ（前処理、推論、後処理）を個別のコンポーネントとして提供。
これらを組み合わせることで、独自の処理フローを構築できます。
(例: リアルタイムカメラストリームへの適用など)

基本的な使用方法：
```python
from third_party.WASB_SBDT import (
    load_default_config,
    FramePreprocessor,
    BallDetector,
    DetectionPostprocessor,
    build_tracker,
)
import torch

# 1. 設定とコンポーネントの初期化
config = load_default_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor = FramePreprocessor(config)
detector = BallDetector(config, device)
postprocessor = DetectionPostprocessor(config)
tracker = build_tracker(config)

# 2. 独自のループで処理 (例)
# video_frames: フレームシーケンスのリスト [[f1, f2, f3], [f2, f3, f4], ...]
# for frame_sequence_batch in video_frames:
#     batch_tensor, batch_meta = preprocessor.process_batch(frame_sequence_batch)
#     batch_preds = detector.predict_batch(batch_tensor)
#     batch_detections = postprocessor.process_batch(batch_preds, batch_meta, device)
#
#     for detections in batch_detections:
#         tracking_result = tracker.update(detections)
#         print(tracking_result)
```

========================================
🔩 低レベルAPI (for Experts)
========================================
モデル、ポストプロセッサ、トラッカーなどのコアコンポーネントを直接生成するファクトリ関数。
最大限の自由度で、パッケージの部品を制御したい上級者向けです。

基本的な使用方法：
```python
from third_party.WASB_SBDT import create_model_package, load_default_config

# デフォルト設定でモデル・ポストプロセッサ・トラッカーを作成
config = load_default_config()
model, postprocessor, tracker = create_model_package(config)

# カスタム設定とモデル重みを使用
# model, postprocessor, tracker = create_model_package('path/to/config.yaml', 'path/to/model.pth')

# これで各コンポーネントを個別に利用可能
# e.g., model(input_tensor)
```
"""

import os
from typing import Optional, Union, Tuple
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# --- 低レベルAPI ---
# 内部モジュールのインポート
from .models import build_model
from .detectors.postprocessor import TracknetV2Postprocessor
from .trackers import build_tracker as build_tracker_internal

# --- 中レベルAPI ---
# パイプラインコンポーネントのインポート
from .pipeline_modules import (
    FramePreprocessor,
    BallDetector,
    DetectionPostprocessor,
)

# --- 高レベルAPI ---
# 完全なパイプラインのインポート
from .pipeline_demo import MultithreadedTennisTracker


def load_default_config() -> DictConfig:
    """
    テニスボール検出と追跡のためのデフォルト設定をロードします。

    Returns:
        DictConfig: デフォルト設定を含むOmegaConfオブジェクト。
    """
    cfg = {
        'model': {
            'name': 'hrnet',
            'frames_in': 3,
            'frames_out': 3,
            'inp_height': 288,
            'inp_width': 512,
            'out_height': 288,
            'out_width': 512,
            'rgb_diff': False,
            'out_scales': [0],
            'MODEL': {
                'EXTRA': {
                    'FINAL_CONV_KERNEL': 1,
                    'PRETRAINED_LAYERS': ['*'],
                    'STEM': {'INPLANES': 64, 'STRIDES': [1, 1]},
                    'STAGE1': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [1], 'NUM_CHANNELS': [32], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE2': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2], 'NUM_CHANNELS': [16, 32], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2], 'NUM_CHANNELS': [16, 32, 64], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2, 2], 'NUM_CHANNELS': [16, 32, 64, 128], 'FUSE_METHOD': 'SUM'
                    },
                    'DECONV': {'NUM_DECONVS': 0, 'KERNEL_SIZE': [], 'NUM_BASIC_BLOCKS': 2}
                },
                'INIT_WEIGHTS': True
            }
        },
        'detector': {
            'name': 'tracknetv2',
            'model_path': 'third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar',
            'postprocessor': {
                'name': 'tracknetv2',
                'score_threshold': 0.5,
                'scales': [0],
                'blob_det_method': 'concomp',
                'use_hm_weight': True
            }
        },
        'tracker': {
            'name': 'online',
            'max_disp': 100
        },
        'dataloader': {
            'heatmap': {
                'sigmas': {0: 2.0}
            }
        },
        'runner': {
            'device': 'cuda',
            'gpus': [0]
        }
    }
    return OmegaConf.create(cfg)


def create_model_package(
    config: Union[str, Path, DictConfig],
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.nn.Module, object, object]:
    """
    設定に基づき、モデル、ポストプロセッサ、トラッカーを生成して返します。

    Args:
        config (Union[str, Path, DictConfig]): 設定ファイルのパスまたはDictConfigオブジェクト。
        model_path (Optional[Union[str, Path]]): 学習済みモデルの重みファイルへのパス。
                                                  指定した場合、設定ファイル内のパスを上書きします。

    Returns:
        tuple: (model, postprocessor, tracker)
            - model (torch.nn.Module): 推論準備済みのニューラルネットワークモデル。
            - postprocessor (object): ヒートマップを座標に変換するポストプロセッサ。
            - tracker (object): フレーム間でボールを追跡するオンライントラッカー。
    """
    # 1. 設定のロード
    if isinstance(config, (str, Path)):
        cfg = OmegaConf.load(config)
    else:
        cfg = config

    # 2. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.runner.device = str(device)

    # 3. モデルパスの上書き
    if model_path is not None:
        cfg.detector.model_path = str(model_path)
    
    # 4. モデルの構築と重みのロード
    model = build_model(cfg).to(device)
    
    ckpt_path = cfg.detector.model_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 'module.'プレフィックスを削除
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 5. ポストプロセッサの構築
    postprocessor = TracknetV2Postprocessor(cfg)

    # 6. トラッカーの構築
    tracker = build_tracker_internal(cfg)
    
    return model, postprocessor, tracker


def build_tracker(config: DictConfig) -> object:
    """
    設定に基づき、オンライントラッカーを構築します。

    Args:
        config (DictConfig): トラッカー設定を含むOmegaConfオブジェクト。

    Returns:
        object: 初期化されたオンライントラッカー。
    """
    return build_tracker_internal(config)


# エクスポートするモジュールを__all__で明示的に指定
__all__ = [
    # --- 高レベルAPI ---
    "MultithreadedTennisTracker",
    # --- 中レベルAPI ---
    "FramePreprocessor",
    "BallDetector",
    "DetectionPostprocessor",
    # --- 低レベルAPI ---
    "create_model_package",
    "build_tracker",
    "load_default_config",
]
