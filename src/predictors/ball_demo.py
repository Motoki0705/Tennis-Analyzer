from src.predictors.ball_predictor_demo import BallPredictor
from src.ball.lit_module.lit_video_swin_transformer import LitVideoSwinTransformer
import torch

if __name__ == "__main__":
    litmodule = LitVideoSwinTransformer.load_from_checkpoint(
        checkpoint_path="checkpoints/ball/video_swin_transformer/best_model.ckpt",
        map_location=torch.device("cuda")
    )

    input_size = (320, 640)
    heatmap_size = (320, 640)
    num_frames = 16
    threshold = 0.6
    device = "cuda"
    use_half: bool = True


    ball_predictor = BallPredictor(
        litmodule=litmodule,
        input_size=input_size,
        heatmap_size=heatmap_size,
        num_frames=num_frames,
        threshold=threshold,
        device=device,
        use_half=use_half
    )
    

    ball_predictor.run(input_path="datasets/test/video_input2.mp4",
                       output_path="outputs/ball/swin_transformer_v2.mp4",
                       batch_size=4)