import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.court.predictor import CourtPredictor

@hydra.main(config_path="configs/infers", config_name="court")
def main(cfg):
    model_path   = to_absolute_path(cfg.model.path)
    input_video  = to_absolute_path(cfg.input_video)
    output_video = to_absolute_path(cfg.output_video)

    predictor = CourtPredictor(
        model_path=    model_path,
        device=        cfg.device,
        input_size=    tuple(cfg.input_size),
        num_keypoints= cfg.num_keypoints,
        threshold=     cfg.threshold,
        min_distance=  cfg.min_distance,
        radius=        cfg.radius,
        kp_color=      tuple(cfg.kp_color),
    )
    predictor.run(
        input_path=  input_video,
        output_path= output_video,
        batch_size=  cfg.batch_size
    )

if __name__ == "__main__":
    main()
