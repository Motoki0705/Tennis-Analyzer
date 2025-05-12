import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.player.predictor import PlayerPredictor
from src.pose.predictor import PosePredictor
from src.utils.load_model import load_model_weights

@hydra.main(config_path="configs/infers", config_name="pose")
def main(cfg):
    ckpt_path = to_absolute_path(cfg.pose.det_checkpoint)
    input_video  = to_absolute_path(cfg.input_video)
    output_video = to_absolute_path(cfg.output_video)

    # ── PlayerPredictor の準備 ──
    det_processor = hydra.utils.instantiate(cfg.pose.det_processor)
    det_model     = hydra.utils.instantiate(cfg.pose.det_model)
    det_model     = load_model_weights(det_model, ckpt_path)

    pose_processor= hydra.utils.instantiate(cfg.pose.pose_processor)
    pose_model    = hydra.utils.instantiate(cfg.pose.pose_model)

    predictor = PosePredictor(
        det_model=det_model,
        det_processor=det_processor,
        pose_model=pose_model,
        pose_processor=pose_processor,
        device=cfg.device,
        player_label_id=cfg.pose.player_label_id,
        det_score_thresh=cfg.pose.det_score_thresh,
        pose_score_thresh=cfg.pose.pose_score_thresh
    )

    predictor.run(
        input_path=  input_video,
        output_path= output_video,
        batch_size=  cfg.batch_size
    )

if __name__ == "__main__":
    main()
