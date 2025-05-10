import hydra
from pathlib import Path

from hydra.utils import to_absolute_path
from src.multi.frames_predictor import FrameAnnotator
from src.ball.predictor import BallPredictor
from src.court.predictor import CourtPredictor
from src.pose.predictor import PosePredictor
from src.utils.load_model import load_model_weights

@hydra.main(config_path="configs/infers", config_name="frames")
def main(cfg):
    # BallPredictor
    ball = BallPredictor(
        ckpt_path=to_absolute_path(cfg.ball.model.ckpt_path),
        input_size=tuple(cfg.ball.input_size),
        heatmap_size=tuple(cfg.ball.heatmap_size),
        num_frames=cfg.ball.num_frames,
        threshold=cfg.ball.threshold,
        device=cfg.device,
    )

    # CourtPredictor
    court = CourtPredictor(
        model_path=to_absolute_path(cfg.court.model.path),
        device=cfg.device,
        input_size=tuple(cfg.court.input_size),
        num_keypoints=cfg.court.num_keypoints,
        threshold=cfg.court.threshold,
        min_distance=cfg.court.min_distance,
        radius=cfg.court.radius,
        kp_color=tuple(cfg.court.kp_color),
    )

    # PosePredictor
    det_processor = hydra.utils.instantiate(cfg.pose.det_processor)
    det_model     = hydra.utils.instantiate(cfg.pose.det_model)
    det_model     = load_model_weights(det_model, to_absolute_path(cfg.pose.det_checkpoint))

    pose_processor= hydra.utils.instantiate(cfg.pose.pose_processor)
    pose_model    = hydra.utils.instantiate(cfg.pose.pose_model)

    pose = PosePredictor(
        det_model=det_model,
        det_processor=det_processor,
        pose_model=pose_model,
        pose_processor=pose_processor,
        device=cfg.device,
        player_label_id=cfg.pose.player_label_id,
        det_score_thresh=cfg.pose.det_score_thresh,
        pose_score_thresh=cfg.pose.pose_score_thresh
    )

    annotator = FrameAnnotator(
        ball_predictor=ball,
        court_predictor=court,
        pose_predictor=pose,
        intervals=cfg.intervals,
        frame_fmt=cfg.frame_fmt
    )

    # infer_frames.py 内 main() のイメージ
    for vid in cfg.input_videos:
        subdir = Path(cfg.output_root) / vid.name
        frames_dir = subdir / "frames"
        ann_json   = subdir / "annotations.jsonl"

        annotator.run(
            input_path=to_absolute_path(vid.path),
            output_dir=to_absolute_path(frames_dir),
            output_json=to_absolute_path(ann_json),
        )
if __name__ == "__main__":
    main()
