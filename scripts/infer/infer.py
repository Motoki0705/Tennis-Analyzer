#!/usr/bin/env python
"""
Hydra-driven Inference System for Tennis Analyzer
=================================================
モード
------
- ball   : ボール検出 & トラッキング
- court  : コート 15 キーポイント検出
- player : プレーヤー検出
- pose   : 検出器＋ViT-Pose による姿勢推定
- event  : ボール、プレイヤー、コートの特徴からイベント（ヒット、バウンド）検出
- streaming_overlayer : マルチタスク推論を並列実行しストリーミングにオーバレイ
- multi  : ball + court + pose を同時にオーバレイ
- frames : 各フレームを推論し JSONL へ書き出し
- image  : 画像ディレクトリ内の画像に対して推論し、JSONのみ出力
- multi_event : ball + court + pose + event を並列実行しオーバレイ
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

# カレントディレクトリをPythonパスに追加（モジュールインポート用）
sys.path.append('.')

# 設定ディレクトリのパス
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

# ──────────────────────  logger  ──────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ╭──────────────────────────────────────────────────╮
# │  1. モデルインスタンス化                        │
# ╰──────────────────────────────────────────────────╯
def instantiate_model(cfg: DictConfig, task: str) -> torch.nn.Module:
    """
    タスクに応じてモデルをインスタンス化します。
    
    - LightningModule: load_from_checkpoint を使用
    - transformers: from_pretrained を使用（poseタスクのみ）
    
    Args:
        cfg (DictConfig): 全体設定
        task (str): タスク名 ('ball', 'court', 'player', 'pose', 'event')
        
    Returns:
        torch.nn.Module: インスタンス化されたモデル
    """
    task_cfg = cfg[task]
    target_class = task_cfg.get("_target_", "")
    
    if not target_class:
        raise ValueError(f"_target_ is required for task '{task}'")
    
    logger.info(f"Loading {task} model: {target_class}")
    
    # transformersモデルの場合（poseタスクなど）
    if target_class.startswith("transformers."):
        logger.info(f"Using transformers.from_pretrained for {task}")
        model = instantiate(task_cfg)
        return model
    
    # LightningModuleの場合（ball, court, player, eventタスクなど）
    ckpt_path = task_cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(f"ckpt_path is required for LightningModule task '{task}'")
    
    # モジュールパスとクラス名を分離
    try:
        module_path, class_name = target_class.rsplit('.', 1)
    except ValueError:
        raise ValueError(f"Invalid _target_ format for task '{task}': {target_class}")
    
    try:
        # 動的インポート
        import importlib
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
        
        # チェックポイントからロード
        ckpt_abs = to_absolute_path(ckpt_path)
        logger.info(f"Loading {task} model from checkpoint: {ckpt_abs}")
        
        if not Path(ckpt_abs).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")
            
        model = model_cls.load_from_checkpoint(ckpt_abs)
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model for task '{task}': {e}")
        raise


# ╭──────────────────────────────────────────────────╮
# │  2. Predictor インスタンス化                     │
# ╰──────────────────────────────────────────────────╯
def get_predictor(
    cfg: DictConfig, mode: str, cache: Dict[str, torch.nn.Module] | None = None
):
    """
    単一モード用 Predictor を返します。
    既に生成済みモデルは `cache` から再利用します。
    
    Args:
        cfg (DictConfig): 全体設定
        mode (str): モード名
        cache (Dict[str, torch.nn.Module], optional): モデルキャッシュ
        
    Returns:
        Predictor: インスタンス化されたPredictor
    """
    if cache is None:
        cache = {}

    device = cfg.common.device
    use_half = cfg.common.use_half

    if mode == "ball":
        cache.setdefault("ball", instantiate_model(cfg, "ball").to(device))
        return instantiate(
            cfg.predictors.ball, 
            litmodule=cache["ball"], 
            device=device, 
            use_half=use_half
        )

    elif mode == "court":
        cache.setdefault("court", instantiate_model(cfg, "court").to(device))
        return instantiate(
            cfg.predictors.court, 
            litmodule=cache["court"], 
            device=device, 
            use_half=use_half
        )

    elif mode == "player":
        cache.setdefault("player", instantiate_model(cfg, "player").to(device))
        processor = instantiate(cfg.processors.player)
        return instantiate(
            cfg.predictors.player,
            litmodule=cache["player"],
            processor=processor,
            device=device,
            use_half=use_half,
        )

    elif mode == "pose":
        # poseモードは検出器（player）+ 姿勢推定（pose）の組み合わせ
        cache.setdefault("pose_det", instantiate_model(cfg, "player").to(device))
        cache.setdefault("pose_est", instantiate_model(cfg, "pose").to(device))
        det_processor = instantiate(cfg.processors.player)
        pose_processor = instantiate(cfg.processors.pose)
        return instantiate(
            cfg.predictors.pose,
            det_litmodule=cache["pose_det"],
            det_processor=det_processor,
            pose_litmodule=cache["pose_est"],
            pose_processor=pose_processor,
            device=device,
            use_half=use_half,
        )
        
    elif mode == "event":
        cache.setdefault("event", instantiate_model(cfg, "event").to(device))
        return instantiate(
            cfg.predictors.event,
            litmodule=cache["event"],
            device=device,
            use_half=use_half,
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")


# ╭──────────────────────────────────────────────────╮
# │  3. 複合モード用のモデルとPredictor初期化        │
# ╰──────────────────────────────────────────────────╯
def setup_composite_predictors(cfg: DictConfig) -> Tuple:
    """
    複合モード（multi/frames/image）用の初期化処理
    モデルとプロセッサをロードし、各種Predictorを返します。
    
    Args:
        cfg (DictConfig): 全体設定
        
    Returns:
        Tuple: モードに応じたPredictorのタプル
    """
    device = cfg.common.device
    use_half = cfg.common.use_half
    
    # モデルのロード
    models = {
        "ball": instantiate_model(cfg, "ball").to(device),
        "court": instantiate_model(cfg, "court").to(device),
        "player": instantiate_model(cfg, "player").to(device),
        "pose": instantiate_model(cfg, "pose").to(device),
    }
    
    # eventモデルが必要な場合のみロード
    if hasattr(cfg, 'multi') and cfg.multi.mode in ("image", "image_with_event_detector"):
        models["event"] = instantiate_model(cfg, "event").to(device)

    # プロセッサのインスタンス化
    player_processor = instantiate(cfg.processors.player)
    pose_processor = instantiate(cfg.processors.pose)

    # 各Predictorのインスタンス化
    ball_predictor = instantiate(
        cfg.predictors.ball,
        litmodule=models["ball"],
        device=device,
        use_half=use_half,
    )
    
    court_predictor = instantiate(
        cfg.predictors.court,
        litmodule=models["court"],
        device=device,
        use_half=use_half,
    )
    
    player_predictor = instantiate(
        cfg.predictors.player,
        litmodule=models["player"],
        processor=player_processor,
        device=device,
        use_half=use_half,
    )
    
    pose_predictor = instantiate(
        cfg.predictors.pose,
        det_litmodule=models["player"],
        det_processor=player_processor,
        pose_litmodule=models["pose"],
        pose_processor=pose_processor,
        device=device,
        use_half=use_half,
    )

    # モードに応じて返すPredictorを決定
    if hasattr(cfg, 'multi'):
        if cfg.multi.mode == "frames":
            return (ball_predictor, court_predictor, player_predictor, pose_predictor)
        elif cfg.multi.mode == "image":
            event_predictor = instantiate(
                cfg.predictors.event,
                litmodule=models["event"],
                device=device,
                use_half=use_half,
            )
            return (ball_predictor, court_predictor, player_predictor, pose_predictor, event_predictor)
        elif cfg.multi.mode == "image_with_event_detector":
            event_predictor = instantiate(
                cfg.predictors.event,
                litmodule=models["event"],
                device=device,
                use_half=use_half,
            )
            return (ball_predictor, event_predictor)
    
    # デフォルト（multi等）
    return (ball_predictor, court_predictor, pose_predictor)


# ╭──────────────────────────────────────────────────╮
# │  4. 出力パス関連                                  │
# ╰──────────────────────────────────────────────────╯
def get_output_path(cfg: DictConfig, input_path: Path, mode: str) -> Path:
    """
    出力パスを生成します。
    
    Args:
        cfg (DictConfig): 全体設定
        input_path (Path): 入力パス
        mode (str): モード名
        
    Returns:
        Path: 出力パス
    """
    # 設定から出力パスを取得
    out_cfg = cfg.get("output_path")
    if out_cfg:
        return Path(to_absolute_path(out_cfg))
    
    # デフォルトの出力パス生成
    if mode in cfg and "_target_" in cfg[mode]:
        model_name = Path(cfg[mode]._target_.split(".")[-1]).stem
        return input_path.with_name(f"{input_path.stem}_{mode}_{model_name}.mp4")
    else:
        return input_path.with_name(f"{input_path.stem}_{mode}.mp4")


# ╭──────────────────────────────────────────────────╮
# │  5. メイン処理                                    │
# ╰──────────────────────────────────────────────────╯
@hydra.main(config_path=str(CONFIG_DIR / "infer"), config_name="infer")
def infer(cfg: DictConfig):
    """
    Hydraによって設定が渡され、推論処理を実行します。
    
    Args:
        cfg (DictConfig): Hydra設定
    """
    print(f"Current working directory : {os.getcwd()}")

    logger.info("Starting inference...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    mode = cfg.mode
    inp = Path(to_absolute_path(cfg.input_path))
    batch_size = cfg.common.batch_size

    # 出力パスの設定
    out_cfg = cfg.get("output_path")
    if mode == "frames" and out_cfg:
        out_path = Path(to_absolute_path(out_cfg)).with_suffix("")
    else:
        out_path = Path(to_absolute_path(out_cfg)) if out_cfg else None

    # ─── 単一モード ───────────────────────────
    if mode in ("ball", "court", "player", "pose", "event"):
        predictor = get_predictor(cfg, mode)
        
        if not out_path:
            out_path = get_output_path(cfg, inp, mode)
            
        logger.info(f"[{mode}] Input: {inp} → Output: {out_path}")
        
        # eventモードの場合はjson出力パスも設定
        if mode == "event":
            json_cfg = cfg.get("output_json_path")
            json_path = Path(to_absolute_path(json_cfg)) if json_cfg else out_path.with_suffix(".json")
            logger.info(f"[{mode}] JSON → {json_path}")
            predictor.run(inp, out_path, json_output_path=json_path, batch_size=batch_size)
        else:
            predictor.run(inp, out_path, batch_size=batch_size)
            
        logger.info("Finished.")
        return

    # ─── ストリーミングオーバーレイ ───────────────────────
    elif mode == "streaming_overlayer":
        ball_pred, court_pred, pose_pred = setup_composite_predictors(cfg)

        streaming_pred = instantiate(
            cfg.predictors.streaming_overlayer,
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
        )

        if not out_path:
            out_path = inp.with_name(f"{inp.stem}_streaming.mp4")
        logger.info(f"[streaming_overlayer] Input: {inp} → Output: {out_path}")
        streaming_pred.run(inp, out_path)
        logger.info("Finished.")
        return

    # ─── マルチイベントオーバーレイ ───────────────────────
    elif mode == "multi_event":
        ball_pred, court_pred, pose_pred = setup_composite_predictors(cfg)
        event_pred = get_predictor(cfg, "event")

        multi_event_pred = instantiate(
            cfg.predictors.multi_event,
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            event_predictor=event_pred,
        )

        if not out_path:
            out_path = inp.with_name(f"{inp.stem}_multi_event.mp4")
        logger.info(f"[multi_event] Input: {inp} → Output: {out_path}")
        multi_event_pred.run(inp, out_path)
        logger.info("Finished.")
        return

    # ─── 複合モード (multi/frames/image) ───────────────────────
    elif mode in ("multi", "frames", "image"):
        predictors = setup_composite_predictors(cfg)

        if mode == "multi":
            ball_pred, court_pred, pose_pred = predictors
            multi_pred = instantiate(
                cfg.predictors.multi,
                ball_predictor=ball_pred,
                court_predictor=court_pred,
                pose_predictor=pose_pred,
            )
            if not out_path:
                out_path = inp.with_name(f"{inp.stem}_multi.mp4")
            logger.info(f"[multi] Input: {inp} → Output: {out_path}")
            multi_pred.run(inp, out_path)

        elif mode == "frames":
            ball_pred, court_pred, player_pred, pose_pred = predictors
            frames_ann = instantiate(
                cfg.predictors.frames_annotator,
                ball_predictor=ball_pred,
                court_predictor=court_pred,
                pose_predictor=pose_pred,
            )
            if not out_path:
                out_path = inp.with_name(f"{inp.stem}_frames")
            out_path.mkdir(parents=True, exist_ok=True)
            
            jsonl_cfg = cfg.get("output_json_path")
            jsonl_path = (
                Path(to_absolute_path(jsonl_cfg))
                if jsonl_cfg
                else out_path / "annotations.json"
            )
            logger.info(f"[frames] Input: {inp} → Dir: {out_path}, JSONL: {jsonl_path}")
            frames_ann.run(inp, out_path, jsonl_path)
            
        elif mode == "image":
            ball_pred, court_pred, player_pred, pose_pred, event_pred = predictors
            
            from src.multi.image_annotator import ImageAnnotator
            
            image_ann = ImageAnnotator(
                ball_predictor=ball_pred,
                court_predictor=court_pred,
                pose_predictor=pose_pred,
                batch_sizes=cfg.predictors.get("image_annotator", {}).get("batch_sizes", {"ball": 16, "court": 16, "pose": 16}),
                ball_vis_thresh=cfg.predictors.ball.threshold,
                court_vis_thresh=cfg.predictors.court.threshold,
                pose_vis_thresh=cfg.predictors.pose.pose_score_thresh,
            )
            
            json_cfg = cfg.get("output_json_path")
            json_path = Path(to_absolute_path(json_cfg)) if json_cfg else Path(f"outputs/image_annotations_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
            
            logger.info(f"[image] Input Dir: {inp} → Output JSON: {json_path}")
            image_ann.run(inp, json_path)

        logger.info("Finished.")
        return

    else:
        logger.error(f"Unknown mode: {mode}")
        raise ValueError(f"Unknown mode: {mode}")

 
if __name__ == "__main__":
    infer()
