import os
import cv2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.utils.visualization.visualization import tensor_to_cv2_image, visualize_peak_valley_heatmap, visualize_standard_heatmap

class HeatmapVisualizerCallback(Callback):
    """
    検証エポックの終了時に、予測ヒートマップを可視化して保存するコールバック。
    モデルのhparamsに応じて、ピーク・谷の可視化を切り替える。
    """
    def __init__(self, output_dir: str = "heatmap_visualizations", num_samples_to_log: int = 1):
        super().__init__()
        self.output_dir = output_dir
        self.num_samples_to_log = num_samples_to_log

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.sanity_checking:
            return

        try:
            val_loader = trainer.val_dataloaders
            batch = next(iter(val_loader))
        except (StopIteration, IndexError):
            print("Warning: Could not get a batch from the validation dataloader. Skipping visualization.")
            return

        images, _, _ = batch
        images = images.to(pl_module.device)

        pl_module.eval()
        with torch.no_grad():
            pred_logits = pl_module(images)
            pred_heatmaps = torch.sigmoid(pred_logits) # -> [0, 1] の範囲
        pl_module.train()

        # --- ここからが修正箇所 ---
        # モデルのハイパーパラメータから、ピーク・谷モードかを確認
        use_peak_valley = getattr(pl_module.hparams, 'use_peak_valley_heatmaps', False)

        # ピーク・谷モードの場合、可視化のため[0, 1]から[-1, 1]にスケール
        if use_peak_valley:
            # `visualize_peak_valley_heatmap` は [-1, 1] のヒートマップを期待する
            display_heatmaps = pred_heatmaps * 2.0 - 1.0
        else:
            # 通常のヒートマップ（[0, 1]のまま）
            display_heatmaps = pred_heatmaps 
        # --- ここまでが修正箇所 ---

        log_dir = trainer.log_dir or "lightning_logs/temp"
        vis_dir = os.path.join(log_dir, self.output_dir)
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"\nSaving visualization heatmaps to: {vis_dir}")

        for i in range(min(self.num_samples_to_log, images.size(0))):
            image_tensor = images[i]
            heatmaps_to_display = display_heatmaps[i]
            image_bgr = tensor_to_cv2_image(image_tensor)
            
            num_keypoints = heatmaps_to_display.shape[0]
            
            for kpt_idx in range(num_keypoints):
                single_heatmap = heatmaps_to_display[kpt_idx]
                
                if use_peak_valley:
                    # ピーク・谷可視化（[-1, 1]範囲）
                    overlay_img = visualize_peak_valley_heatmap(image_bgr.copy(), single_heatmap)
                    filename = f"epoch_{trainer.current_epoch:03d}_sample_{i}_kpt_{kpt_idx:02d}_peak_valley.png"
                else:
                    # 通常のヒートマップ可視化（[0, 1]範囲）
                    overlay_img = visualize_standard_heatmap(image_bgr.copy(), single_heatmap)
                    filename = f"epoch_{trainer.current_epoch:03d}_sample_{i}_kpt_{kpt_idx:02d}_standard.png"
                
                output_path = os.path.join(vis_dir, filename)
                cv2.imwrite(output_path, overlay_img)