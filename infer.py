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
- multi  : ball + court + pose を同時にオーバレイ
- frames : 各フレームを推論し JSONL へ書き出し
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

# ──────────────────────  logger  ──────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ╭──────────────────────────────────────────────────╮
# │  1. モデル設定取得ユーティリティ                │
# ╰──────────────────────────────────────────────────╯
def _get_model_cfg(cfg: DictConfig, task: str) -> DictConfig:
    """
    `cfg.<task>` があればそれを、なければ `cfg._group_[task]` を返す。
    ─ task: 'ball' | 'court' | 'player' | 'pose'
    """
    if task in cfg:
        return cfg[task]
    if "_group_" in cfg and task in cfg["_group_"]:
        return cfg["_group_"][task]
    raise KeyError(f"Model config for '{task}' not found.")


# ╭──────────────────────────────────────────────────╮
# │  2. モデル instantiate + ckpt 読み込み           │
# ╰──────────────────────────────────────────────────╯
def instantiate_model(cfg: DictConfig, task: str):
    """
    - DictConfig から _target_ を持つ設定を取り出し
    - ckpt_path は **コンストラクタに渡さず**、ロード処理にのみ使用
    """
    full_cfg = _get_model_cfg(cfg, task)

    # ckpt_path を退避してコンストラクタ用 DictConfig を作成
    ckpt_path = full_cfg.get("ckpt_path")
    ctor_cfg = OmegaConf.create({k: v for k, v in full_cfg.items() if k != "ckpt_path"})

    logger.info(f"Instantiating '{task}' model: {ctor_cfg._target_}")
    model = instantiate(ctor_cfg)

    # ─── checkpoint をロード ───
    if ckpt_path:
        ckpt_abs = to_absolute_path(ckpt_path)
        logger.info(f"Loading checkpoint for '{task}' from: {ckpt_abs}")
        try:
            from src.utils.load_model import load_model_weights

            model = load_model_weights(model, ckpt_abs)
        except Exception as e:
            logger.warning(f"load_model_weights failed ({e}); fallback to torch.load.")
            state = torch.load(ckpt_abs, map_location="cpu")
            # Lightning 形式と純 state_dict 両対応
            sd = state.get("state_dict", state)
            if isinstance(sd, dict):
                sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)

    return model


# ╭──────────────────────────────────────────────────╮
# │  3. Predictor インスタンス化                     │
# ╰──────────────────────────────────────────────────╯
def get_predictor(cfg: DictConfig, mode: str, cache: Dict[str, torch.nn.Module] | None = None):
    """
    単一モード(ball / court / player / pose)用 Predictor を返す。
    既に生成済みモデルは `cache` から再利用。
    """
    if cache is None:
        cache = {}

    device = cfg.common.device
    use_half = cfg.common.use_half

    if mode == "ball":
        cache.setdefault("ball", instantiate_model(cfg, "ball").to(device))
        return instantiate(cfg.predictors.ball, model=cache["ball"], device=device, use_half=use_half)

    if mode == "court":
        cache.setdefault("court", instantiate_model(cfg, "court").to(device))
        return instantiate(cfg.predictors.court, model=cache["court"], device=device, use_half=use_half)

    if mode == "player":
        cache.setdefault("player", instantiate_model(cfg, "player").to(device))
        proc = instantiate(cfg.processors.player)
        return instantiate(cfg.predictors.player, model=cache["player"], processor=proc, device=device, use_half=use_half)

    if mode == "pose":
        cache.setdefault("pose_det", instantiate_model(cfg, "player").to(device))
        cache.setdefault("pose_est", instantiate_model(cfg, "pose").to(device))
        det_proc = instantiate(cfg.processors.player)
        pose_proc = instantiate(cfg.processors.pose)
        return instantiate(
            cfg.predictors.pose,
            det_model=cache["pose_det"],
            det_processor=det_proc,
            pose_model=cache["pose_est"],
            pose_processor=pose_proc,
            device=device,
            use_half=use_half,
        )

    raise ValueError(f"Unsupported mode: {mode}")


# ╭──────────────────────────────────────────────────╮
# │  4. メイン                                      │
# ╰──────────────────────────────────────────────────╯
@hydra.main(config_path="configs/infer", config_name="infer", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("Starting inference…")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    mode = cfg.mode
    inp = Path(to_absolute_path(cfg.input_path))
    out_cfg = cfg.get("output_path")
    raw_out = cfg.get("output_path")
    if mode == "frames" and raw_out:
        out_path = Path(to_absolute_path(raw_out)).with_suffix("")
    else:
        out_path = Path(to_absolute_path(raw_out)) if raw_out else None
    batch = cfg.common.batch_size
    device = cfg.common.device

    # ─── 単一モード ───────────────────────────
    if mode in ("ball", "court", "player", "pose"):
        predictor = get_predictor(cfg, mode)
        if not out_path:
            sel = Path(_get_model_cfg(cfg, mode)._target_.split(".")[-1]).stem
            out_path = inp.with_name(f"{inp.stem}_{mode}_{sel}.mp4")
        logger.info(f"[{mode}] → {out_path}")
        predictor.run(inp, out_path, batch_size=batch)
        logger.info("Finished.")
        return

    # ─── multi / frames ───────────────────────
    if mode in ("multi", "frames"):
        models = {
            "ball": instantiate_model(cfg, "ball").to(device),
            "court": instantiate_model(cfg, "court").to(device),
            "pose_det": instantiate_model(cfg, "player").to(device),
            "pose_est": instantiate_model(cfg, "pose").to(device),
        }
        player_proc = instantiate(cfg.processors.player)
        pose_proc = instantiate(cfg.processors.pose)

        ball_pred = instantiate(cfg.predictors.ball, model=models["ball"], device=device, use_half=cfg.common.use_half)
        court_pred = instantiate(cfg.predictors.court, model=models["court"], device=device, use_half=cfg.common.use_half)
        pose_pred = instantiate(
            cfg.predictors.pose,
            det_model=models["pose_det"],
            det_processor=player_proc,
            pose_model=models["pose_est"],
            pose_processor=pose_proc,
            device=device,
            use_half=cfg.common.use_half,
        )

        if mode == "multi":
            multi_pred = instantiate(
                cfg.predictors.multi,
                ball_predictor=ball_pred,
                court_predictor=court_pred,
                pose_predictor=pose_pred
            )
            if not out_path:
                out_path = inp.with_name(f"{inp.stem}_multi.mp4")
            logger.info(f"[multi] → {out_path}")
            multi_pred.run(inp, out_path)

        else:  # frames
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
            jsonl_path = Path(to_absolute_path(jsonl_cfg)) if jsonl_cfg else out_path / "annotations.json"
            logger.info(f"[frames] Dir: {out_path}, JSONL: {jsonl_path}")
            frames_ann.run(inp, out_path, jsonl_path)

        logger.info("Finished.")
        return

    logger.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
