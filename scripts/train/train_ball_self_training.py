#!/usr/bin/env python
"""
ボール検出モデルの自己学習用スクリプト

使用例:
    python scripts/train/train_ball_self_training.py
"""

import json
import logging
import os
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from src.ball.self_training.trajectory_tracker import TrajectoryTracker
from src.ball.self_training.self_training_cycle import SelfTrainingCycle


def validate_model_litmodule_compatibility(model_config, litmodule_config):
    """
    モデルとLitModuleの出力タイプの互換性を検証する関数
    
    Args:
        model_config: モデルの設定
        litmodule_config: LitModuleの設定
        
    Raises:
        ValueError: 互換性がない場合
    """
    model_output_type = model_config.meta.output_type
    litmodule_output_type = litmodule_config.meta.output_type
    
    if model_output_type != litmodule_output_type:
        raise ValueError(
            f"モデルの出力タイプ '{model_output_type}' と "
            f"LitModuleの出力タイプ '{litmodule_output_type}' が一致しません。"
            f"モデル: {model_config.meta.name}, LitModule: {litmodule_config.meta.name}"
        )
    
    logging.info(f"モデルとLitModuleの出力タイプ '{model_output_type}' が一致しています")


def train_cycle(cfg: DictConfig, cycle_index: int) -> pl.Trainer:
    """
    1サイクルの自己学習を実行する関数
    
    Args:
        cfg: Hydra設定
        cycle_index: 現在のサイクルインデックス
    
    Returns:
        trainer: トレーニング済みのTrainer
    """
    # シードを設定
    pl.seed_everything(cfg.seed + cycle_index)
    
    # モデルとLitModuleの互換性をチェック
    validate_model_litmodule_compatibility(cfg.model, cfg.litmodule)
    
    # モデルをインスタンス化
    model = instantiate(cfg.model.net)
    logging.info(f"Model: {model.__class__.__name__}")
    
    # LightningModuleをインスタンス化
    lit_module = instantiate(cfg.litmodule.module, model=model)
    logging.info(f"LightningModule: {lit_module.__class__.__name__}")
    
    # DataModuleをインスタンス化
    datamodule = instantiate(cfg.litdatamodule)
    logging.info(f"DataModule: {datamodule.__class__.__name__}")
    
    # コールバックをインスタンス化
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate(cb_conf))
    
    # トレーナーをインスタンス化
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
    )
    
    # トレーニングを実行
    trainer.fit(lit_module, datamodule=datamodule)
    
    return trainer, lit_module, datamodule


def self_training(cfg: DictConfig) -> None:
    """
    自己学習の全サイクルを実行する関数
    
    Args:
        cfg: Hydra設定
    """
    # 設定を表示
    logging.info(f"Self-training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 自己学習サイクルのインスタンス化
    cycle = SelfTrainingCycle(
        max_cycles=cfg.self_training.max_cycles,
        confidence_threshold=cfg.self_training.confidence_threshold,
        trajectory_window=cfg.self_training.trajectory_window,
        max_trajectory_gap=cfg.self_training.max_trajectory_gap,
        min_trajectory_length=cfg.self_training.min_trajectory_length,
        output_dir=os.path.join("outputs", "ball", "self_training"),
    )
    
    # 各サイクルを実行
    for i in range(cfg.self_training.max_cycles):
        logging.info(f"Starting self-training cycle {i+1}/{cfg.self_training.max_cycles}")
        
        # サイクルを実行
        trainer, lit_module, datamodule = train_cycle(cfg, i)
        
        # 未ラベルデータに対する予測を生成
        unlabeled_dataloader = datamodule.unlabeled_dataloader()
        unlabeled_dataset = datamodule.get_unlabeled_dataset()
        
        # 予測を実行
        lit_module.eval()
        device = next(lit_module.parameters()).device
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for batch in unlabeled_dataloader:
                # 入力をデバイスに転送
                if isinstance(batch, dict):
                    batch_images = batch["image"].to(device)
                    batch_image_ids = batch["image_id"]
                else:
                    batch_images = batch[0].to(device)
                    batch_image_ids = batch[1]
                
                # 予測を実行
                outputs = lit_module(batch_images)
                
                # 予測結果を処理（モデルの出力タイプに応じて）
                if cfg.model.meta.output_type == "heatmap":
                    # ヒートマップからキーポイントを抽出
                    coords, scores = lit_module.extract_coordinates_from_heatmap(outputs)
                    coords = coords.cpu().numpy()
                    scores = scores.cpu().numpy()
                else:  # coord
                    coords = outputs.cpu().numpy()
                    # 座標ベースの場合、信頼度スコアは別途計算する必要がある
                    scores = np.ones(coords.shape[0])
                
                # 予測結果を保存
                for i in range(len(batch_image_ids)):
                    image_id = batch_image_ids[i].item() if isinstance(batch_image_ids[i], torch.Tensor) else batch_image_ids[i]
                    prediction = {
                        "image_id": image_id,
                        "file_name": unlabeled_dataset.get_image_info(i)["file_name"],
                        "coordinates": coords[i].tolist(),
                        "confidence": float(scores[i]),
                    }
                    predictions.append(prediction)
                    confidences.append(float(scores[i]))
        
        # 軌跡追跡で擬似ラベルを洗練
        tracker = TrajectoryTracker(
            window_size=cfg.self_training.trajectory_window,
            max_gap=cfg.self_training.max_trajectory_gap,
            min_length=cfg.self_training.min_trajectory_length,
        )
        
        refined_predictions = tracker.refine_predictions(predictions)
        
        # 次のサイクルのための擬似ラベルを生成
        pseudo_label_file = cycle.generate_pseudo_labels(refined_predictions, cycle_index=i)
        
        logging.info(f"Generated pseudo-labels for cycle {i+1}: {pseudo_label_file}")
        
        # テストを実行
        trainer.test(lit_module, datamodule=datamodule)
        
        # 最終サイクルならループを抜ける
        if i == cfg.self_training.max_cycles - 1:
            break
        
        # 次のサイクルのための擬似ラベルパスを設定
        cfg.litdatamodule.pseudo_label_path = pseudo_label_file


@hydra.main(config_path="../../configs/train/ball", config_name="self_training_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    メイン関数
    
    Args:
        cfg: Hydra設定
    """
    try:
        self_training(cfg)
    except Exception as e:
        logging.exception(f"Self-training failed: {e}")
        raise


if __name__ == "__main__":
    main() 