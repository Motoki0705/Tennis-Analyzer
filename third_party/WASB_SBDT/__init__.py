"""
WASB-SBDT: Weakly-Supervised Ball Detection and Tracking
========================================================

🎾 テニスボール検出・トラッキング統合パッケージ

このパッケージは、最も優秀なHRNetベースのモデルと高精度トラッキングシステムを
シンプルなAPIで提供します。

主な特徴:
- 🏆 最高精度のテニスボール検出モデル
- 🎯 リアルタイム対応の高速トラッキング
- 🔧 設定ファイル/デフォルト設定両方に対応
- 📊 明確なデータ構造と型安全性

========================================
🚀 Quick Start
========================================

```python
from third_party.WASB_SBDT import create_model_package, load_default_config

# 1. デフォルト設定で簡単開始
config = load_default_config()
model, postprocessor, tracker = create_model_package(config)

# 2. カスタム設定で高度な制御
model, postprocessor, tracker = create_model_package('config.yaml', 'weights.pth')

# 3. 動画フレーム処理
for frame_detections in video_data:
    result = tracker.update(frame_detections)
    print(f"Ball: ({result['x']}, {result['y']}) visible={result['visi']}")
```

========================================
📋 API Overview
========================================

create_model_package(config_path, model_path=None, device="auto")
    -> (model, postprocessor, tracker)
    
    HRNetベースのボール検出システム全体を作成
    
    Parameters:
    -----------
    config_path : str | Path | DictConfig
        設定ファイルパスまたは設定オブジェクト
    model_path : str | Path, optional
        学習済みモデルの重みファイルパス
    device : str, default="auto"
        実行デバイス ("cuda", "cpu", "auto")
    
    Returns:
    --------
    tuple[model, postprocessor, tracker]
        model: ニューラルネットワークモデル [B,9,288,512] -> [B,3,288,512]
        postprocessor: ヒートマップから球体座標を抽出するポストプロセッサ
        tracker: フレーム間での球体追跡を行うトラッカー

load_default_config() -> DictConfig
    
    テニスボール検出用のデフォルト設定を読み込み
    
    Returns:
    --------
    DictConfig
        HRNet + TrackNetV2 + Online Tracker の組み合わせ設定

========================================
🎯 Data Format
========================================

Processing Pipeline (3-Stage):
    1. Model: Raw frames -> Heatmaps
       Input:  [B, 9, 288, 512] (3 consecutive frames)
       Output: [B, 3, 288, 512] (heatmaps for each frame)
    
    2. Postprocessor: Heatmaps -> Ball coordinates  
       Input:  Heatmaps + affine matrices
       Output: List of detected ball candidates with scores
    
    3. Tracker: Frame detections -> Consistent tracking
       Input:  frame_detections (from postprocessor)
       Output: Single tracked ball position per frame

Input (frame_detections):
    List[Dict[str, Any]] = [
        {'xy': np.array([x, y]), 'score': float},
        ...
    ]

Output (tracker result):
    Dict[str, Any] = {
        'x': float,      # 選択された球体のx座標
        'y': float,      # 選択された球体のy座標  
        'visi': bool,    # 球体検出成功フラグ
        'score': float   # 信頼度スコア
    }

========================================
ℹ️ Package Info
========================================

Version: 1.0.0
Purpose: 最高精度のテニスボール検出・トラッキング統合システム
Architecture: HRNet + TrackNetV2 + Online Tracker
Performance: リアルタイム処理対応、高精度検出保証
"""

# 外部に公開するAPIを定義
# 隠蔽性を保ちながら、必要な機能のみを外部に露出
from .src import create_model_package, load_default_config
from .src.trackers import build_tracker

__version__ = "1.0.0"
__author__ = "NTT Communications Corporation"
__all__ = [
    "create_model_package",
    "load_default_config",
    "build_tracker"
]