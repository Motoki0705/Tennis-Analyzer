import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.player.predictor import PlayerPredictor
from src.pose.predictor import PosePredictor
from src.utils.load_model import load_model_weights

@hydra.main(config_path="configs/infers", config_name="pose")
def main(cfg):
    # ── PlayerPredictor の準備 ──
    det_processor = hydra.utils.instantiate(cfg.det_processor)
    det_model     = hydra.utils.instantiate(cfg.det_model)
    det_ckpt      = to_absolute_path(cfg.det_checkpoint)
    det_model     = load_model_weights(det_model, det_ckpt)

    player_predictor = PlayerPredictor(
        model=     det_model,
        processor= det_processor,
        label_map= cfg.label_map,
        device=    cfg.device,
        threshold= cfg.det_score_thresh
    )

    # ── PosePredictor の準備 ──
    pose_processor = hydra.utils.instantiate(cfg.pose_processor)
    pose_model     = hydra.utils.instantiate(cfg.pose_model)

    predictor = PosePredictor(
        player_predictor= player_predictor,
        pose_model=       pose_model,
        pose_processor=   pose_processor,
        device=           cfg.device,
        pose_score_thresh=cfg.pose_score_thresh
    )

    input_video  = to_absolute_path(cfg.input_video)
    output_video = to_absolute_path(cfg.output_video)
    predictor.run(
        input_path=  input_video,
        output_path= output_video,
        batch_size=  cfg.batch_size
    )

if __name__ == "__main__":
    main()
