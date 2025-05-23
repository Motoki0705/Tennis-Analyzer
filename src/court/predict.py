import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from src.court.lit_module.heatmap_regression_lit_module import HeatmapRegressionLitModule
from src.court.predictor import CourtPredictor


def predict_image(
    model: HeatmapRegressionLitModule,
    image_path: str,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    visualize_mode: str = "overlay",
) -> Dict[str, Any]:
    """
    単一画像に対してコートキーポイント検出を実行する
    
    Args:
        model: 学習済みのHeatmapRegressionLitModule
        image_path: 入力画像のパス
        output_dir: 出力ディレクトリ（Noneの場合は保存しない）
        threshold: ヒートマップの閾値
        visualize_mode: 視覚化モード（"overlay" または "heatmap"）
        
    Returns:
        検出結果の辞書
    """
    # 予測器を初期化
    predictor = CourtPredictor(
        model=model.model,
        device=model.device,
        threshold=threshold,
        visualize_mode=visualize_mode,
    )
    
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 推論
    heatmaps, keypoints = predictor.predict_single_image(image_rgb)
    
    # 結果を可視化
    vis_image = predictor.visualize_keypoints(image, keypoints)
    
    # 結果を保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, vis_image)
        
        # キーポイントをJSONで保存
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump({"keypoints": keypoints.tolist()}, f, indent=2)
    
    return {
        "image_path": image_path,
        "keypoints": keypoints.tolist(),
    }


def predict_batch(
    model: HeatmapRegressionLitModule,
    image_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    visualize_mode: str = "overlay",
    extensions: List[str] = [".jpg", ".jpeg", ".png"],
) -> List[Dict[str, Any]]:
    """
    ディレクトリ内の画像に対してバッチ予測を実行する
    
    Args:
        model: 学習済みのHeatmapRegressionLitModule
        image_dir: 入力画像のディレクトリ
        output_dir: 出力ディレクトリ
        threshold: ヒートマップの閾値
        visualize_mode: 視覚化モード（"overlay" または "heatmap"）
        extensions: 処理する画像の拡張子
        
    Returns:
        検出結果のリスト
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像ファイルを取得
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(Path(image_dir).glob(f"*{ext}")))
    
    results = []
    for image_path in image_paths:
        try:
            result = predict_image(
                model=model,
                image_path=str(image_path),
                output_dir=output_dir,
                threshold=threshold,
                visualize_mode=visualize_mode
            )
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
    
    # 全体の結果をJSONで保存
    summary_path = os.path.join(output_dir, "predictions.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


@hydra.main(config_path="../../configs/train/court", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    メイン関数
    
    Args:
        cfg: Hydra設定
    """
    # シードを設定
    pl.seed_everything(cfg.get("seed", 42))
    
    # モデルをロード
    checkpoint_path = cfg.get("checkpoint_path", None)
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for prediction")
    
    # モデルをインスタンス化
    model = instantiate(cfg.model)
    
    # LightningModuleをインスタンス化してチェックポイントをロード
    lit_module = HeatmapRegressionLitModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        strict=False
    )
    lit_module.eval()
    
    # GPUが利用可能なら使用
    if torch.cuda.is_available():
        lit_module = lit_module.cuda()
    
    # 予測を実行
    image_dir = cfg.get("predict", {}).get("image_dir", None)
    output_dir = cfg.get("predict", {}).get("output_dir", "outputs/court")
    threshold = cfg.get("predict", {}).get("threshold", 0.5)
    visualize_mode = cfg.get("predict", {}).get("visualize_mode", "overlay")
    
    if image_dir:
        # バッチ予測
        results = predict_batch(
            model=lit_module,
            image_dir=image_dir,
            output_dir=output_dir,
            threshold=threshold,
            visualize_mode=visualize_mode
        )
        logging.info(f"Processed {len(results)} images")
    else:
        # 単一画像の予測
        image_path = cfg.get("predict", {}).get("image_path", None)
        if not image_path:
            raise ValueError("Either image_dir or image_path must be specified")
        
        result = predict_image(
            model=lit_module,
            image_path=image_path,
            output_dir=output_dir,
            threshold=threshold,
            visualize_mode=visualize_mode
        )
        logging.info(f"Processed image: {image_path}")


if __name__ == "__main__":
    main() 