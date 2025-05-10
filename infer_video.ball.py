import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.ball.video_predictor import BallPredictor

@hydra.main(config_path="configs/infers", config_name="video_ball")
def main(cfg):
    # 設定ファイル中の相対パスを絶対パスに変換
    ckpt_path    = to_absolute_path(cfg.model.ckpt_path)
    input_video  = to_absolute_path(cfg.input_video)
    output_video = to_absolute_path(cfg.output_video)

    predictor = BallPredictor(
        ckpt_path=ckpt_path,
        input_size=tuple(cfg.input_size),
        heatmap_size=tuple(cfg.heatmap_size),
        base_ch=cfg.base_ch,
        num_frames=cfg.num_frames,
        threshold=cfg.threshold,
        device=cfg.device,
    )
    predictor.run(
        input_path= input_video,
        output_path= output_video,
        mode="heatmap"
    )

if __name__ == "__main__":
    main()
