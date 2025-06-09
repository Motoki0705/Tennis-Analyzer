#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
イベント検知ストリーミング推論スクリプト

複数のタスク（ball, court, pose）の結果を統合してイベント検知を行い、
動画に信号波をオーバーレイして出力します。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multi.streaming_overlayer.multi_event_predictor import MultiEventPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """
    設定ファイルの妥当性をチェックします。
    
    Args:
        cfg: Hydra設定
        
    Raises:
        ValueError: 設定に問題がある場合
    """
    required_keys = [
        "input_video", "output_video", "event_predictor",
        "intervals", "batch_sizes", "event_sequence_length"
    ]
    
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"必須設定項目が不足しています: {key}")
    
    # 入力動画の存在確認
    input_path = Path(cfg.input_video)
    if not input_path.exists():
        raise ValueError(f"入力動画が見つかりません: {input_path}")
    
    # 出力ディレクトリの作成
    output_path = Path(cfg.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"設定ファイルの検証完了")
    logger.info(f"入力動画: {input_path}")
    logger.info(f"出力動画: {output_path}")


def create_predictors(cfg: DictConfig) -> Dict[str, Any]:
    """
    各種予測器を初期化します。
    
    Args:
        cfg: Hydra設定
        
    Returns:
        Dict[str, Any]: 初期化された予測器の辞書
    """
    predictors = {}
    
    try:
        # イベント予測器
        logger.info("イベント予測器を初期化中...")
        predictors["event"] = instantiate(cfg.event_predictor)
        
        # その他の予測器（設定が存在する場合のみ）
        for predictor_name in ["ball_predictor", "court_predictor", "pose_predictor"]:
            if predictor_name in cfg:
                logger.info(f"{predictor_name}を初期化中...")
                predictors[predictor_name.replace("_predictor", "")] = instantiate(cfg[predictor_name])
            else:
                logger.warning(f"{predictor_name}の設定が見つかりません。ダミー予測器を使用します。")
                # ダミー予測器を作成（テスト用）
                predictors[predictor_name.replace("_predictor", "")] = DummyPredictor(predictor_name)
        
        logger.info("全ての予測器の初期化が完了しました")
        return predictors
        
    except Exception as e:
        logger.error(f"予測器の初期化エラー: {e}")
        raise


class DummyPredictor:
    """
    テスト用のダミー予測器クラス。
    
    実際の予測器が利用できない場合のフォールバック。
    """
    
    def __init__(self, name: str):
        self.name = name
        logger.warning(f"ダミー予測器を作成: {name}")
    
    def preprocess(self, frames):
        """ダミーの前処理"""
        return frames, None
    
    def inference(self, data):
        """ダミーの推論"""
        if self.name == "ball_predictor":
            return [{"x": 320, "y": 240, "confidence": 0.5}]
        elif self.name == "court_predictor":
            return [[0.0] * 45]  # 15 keypoints * 3
        elif self.name == "pose_predictor":
            return [{"bbox": [100, 100, 200, 300], "confidence": 0.7, "keypoints": [0.0] * 51}]
        else:
            return None
    
    def postprocess(self, result, meta_data=None):
        """ダミーの後処理"""
        return result
    
    def overlay(self, frame, prediction):
        """ダミーのオーバーレイ"""
        return frame


@hydra.main(version_base="1.3", config_path="../../configs/infer/event", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    メイン実行関数
    
    Args:
        cfg: Hydra設定
    """
    try:
        logger.info("=== イベント検知ストリーミング推論を開始 ===")
        
        # 設定の検証
        validate_config(cfg)
        
        # 予測器の初期化
        predictors = create_predictors(cfg)
        
        # MultiEventPredictorの初期化
        logger.info("MultiEventPredictorを初期化中...")
        multi_predictor = MultiEventPredictor(
            ball_predictor=predictors["ball"],
            court_predictor=predictors["court"],
            pose_predictor=predictors["pose"],
            event_predictor=predictors["event"],
            intervals=cfg.intervals,
            batch_sizes=cfg.batch_sizes,
            event_sequence_length=cfg.event_sequence_length,
            debug=cfg.get("debug", False)
        )
        
        # ストリーミング推論の実行
        logger.info("ストリーミング推論を開始...")
        multi_predictor.run(
            input_path=cfg.input_video,
            output_path=cfg.output_video
        )
        
        logger.info("=== 推論完了 ===")
        logger.info(f"出力動画: {cfg.output_video}")
        
    except Exception as e:
        logger.error(f"推論実行エラー: {e}")
        if cfg.get("debug", False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 