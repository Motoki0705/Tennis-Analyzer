import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.player.predictor import PlayerPredictor
from src.utils.load_model import load_model_weights

@hydra.main(config_path="configs/infers", config_name="player")
def main(cfg):
    # instantiate processor & model from config
    processor = hydra.utils.instantiate(cfg.processor)
    model     = hydra.utils.instantiate(cfg.model)

    # checkpoint も絶対パス化
    ckpt = to_absolute_path(cfg.model_checkpoint)
    model = load_model_weights(model, ckpt)

    predictor = PlayerPredictor(
        model=     model,
        processor= processor,
        label_map= cfg.label_map,
        device=    cfg.device,
        threshold= cfg.threshold
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
