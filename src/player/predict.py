import os
import logging
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from transformers import RTDetrImageProcessor

from src.player.lit_module.detr_lit_module import DetrLitModule


def predict_image(
    model: DetrLitModule,
    processor: RTDetrImageProcessor,
    image_path: str,
    confidence_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    draw: bool = True,
) -> Dict[str, Any]:
    """
    単一画像に対してプレイヤー検出を実行する
    
    Args:
        model: 学習済みのDetrLitModule
        processor: 画像プロセッサ
        image_path: 入力画像のパス
        confidence_threshold: 検出信頼度の閾値
        output_dir: 出力ディレクトリ（Noneの場合は保存しない）
        draw: 検出結果を描画するかどうか
        
    Returns:
        検出結果の辞書
    """
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 前処理
    inputs = processor(images=image_rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device)
    pixel_mask = inputs.get("pixel_mask", None)
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(model.device)
    
    # 推論
    with torch.no_grad():
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    # 後処理
    target_sizes = torch.tensor([image.shape[:2]])
    results = processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes
    )[0]
    
    # 結果を整形
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    # 検出結果をリストに変換
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        detections.append({
            "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # [x, y, w, h]
            "score": float(score),
            "category_id": int(label)
        })
    
    # 結果を描画
    if draw:
        image_draw = image.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            score = det["score"]
            
            # バウンディングボックスを描画
            cv2.rectangle(image_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # スコアを描画
            text = f"{score:.2f}"
            cv2.putText(image_draw, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 結果を保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image_draw)
            
            # JSONも保存
            json_path = os.path.splitext(output_path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(detections, f, indent=2)
    
    return {
        "image_path": image_path,
        "detections": detections
    }


def predict_batch(
    model: DetrLitModule,
    processor: RTDetrImageProcessor,
    image_dir: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    extensions: List[str] = [".jpg", ".jpeg", ".png"],
) -> List[Dict[str, Any]]:
    """
    ディレクトリ内の画像に対してバッチ予測を実行する
    
    Args:
        model: 学習済みのDetrLitModule
        processor: 画像プロセッサ
        image_dir: 入力画像のディレクトリ
        output_dir: 出力ディレクトリ
        confidence_threshold: 検出信頼度の閾値
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
                processor=processor,
                image_path=str(image_path),
                confidence_threshold=confidence_threshold,
                output_dir=output_dir,
                draw=True
            )
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
    
    # 全体の結果をJSONで保存
    summary_path = os.path.join(output_dir, "predictions.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


@hydra.main(config_path="../../configs/train/player", config_name="config", version_base="1.3")
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
    lit_module = DetrLitModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        strict=False
    )
    lit_module.eval()
    
    # GPUが利用可能なら使用
    if torch.cuda.is_available():
        lit_module = lit_module.cuda()
    
    # プロセッサをインスタンス化
    processor = instantiate(cfg.litdatamodule.processor)
    
    # 予測を実行
    image_dir = cfg.get("predict", {}).get("image_dir", None)
    output_dir = cfg.get("predict", {}).get("output_dir", "outputs/player")
    confidence_threshold = cfg.get("predict", {}).get("confidence_threshold", 0.5)
    
    if image_dir:
        # バッチ予測
        results = predict_batch(
            model=lit_module,
            processor=processor,
            image_dir=image_dir,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold
        )
        logging.info(f"Processed {len(results)} images")
    else:
        # 単一画像の予測
        image_path = cfg.get("predict", {}).get("image_path", None)
        if not image_path:
            raise ValueError("Either image_dir or image_path must be specified")
        
        result = predict_image(
            model=lit_module,
            processor=processor,
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir
        )
        logging.info(f"Processed image: {image_path}")


if __name__ == "__main__":
    main() 