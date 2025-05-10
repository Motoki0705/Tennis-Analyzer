import hydra
from hydra.utils import to_absolute_path
from src.ball.predictor import BallPredictor
from src.court.predictor import CourtPredictor
from src.pose.predictor import PosePredictor
from src.multi.multi_predictor import MultiPredictor
from src.utils.load_model import load_model_weights

@hydra.main(config_path="configs/infers", config_name="multi")
def main(cfg):
    # BallPredictor
    ball_predictor = BallPredictor(
        ckpt_path=to_absolute_path(cfg.ball.model.ckpt_path),
        input_size=tuple(cfg.ball.input_size),
        heatmap_size=tuple(cfg.ball.heatmap_size),
        num_frames=cfg.ball.num_frames,
        threshold=cfg.ball.threshold,
        device=cfg.device,
    )

    # CourtPredictor
    court_predictor = CourtPredictor(
        model_path=to_absolute_path(cfg.court.model.path),
        device=cfg.device,
        input_size=tuple(cfg.court.input_size),
        num_keypoints=cfg.court.num_keypoints,
        threshold=cfg.court.threshold,
        min_distance=cfg.court.min_distance,
        radius=cfg.court.radius,
        kp_color=tuple(cfg.court.kp_color),
    )

    # PosePredictor (内部で人物検出＋姿勢推定)
    det_processor = hydra.utils.instantiate(cfg.pose.det_processor)
    det_model     = hydra.utils.instantiate(cfg.pose.det_model)
    det_model     = load_model_weights(det_model, to_absolute_path(cfg.pose.det_checkpoint))

    pose_processor= hydra.utils.instantiate(cfg.pose.pose_processor)
    pose_model    = hydra.utils.instantiate(cfg.pose.pose_model)

    pose_predictor = PosePredictor(
        det_model=det_model,
        det_processor=det_processor,
        pose_model=pose_model,
        pose_processor=pose_processor,
        device=cfg.device,
        player_label_id=cfg.pose.player_label_id,
        det_score_thresh=cfg.pose.det_score_thresh,
        pose_score_thresh=cfg.pose.pose_score_thresh
    )

    # MultiPredictor
    predictor = MultiPredictor(
        ball_predictor=ball_predictor,
        court_predictor=court_predictor,
        pose_predictor=pose_predictor,
        ball_interval=cfg.intervals.ball,
        court_interval=cfg.intervals.court,
        pose_interval=cfg.intervals.pose
    )

    predictor.run(
        input_path=to_absolute_path(cfg.input_video),
        output_path=to_absolute_path(cfg.output_video),
    )

if __name__ == "__main__":
    main()
