import hydra
from hydra.utils import to_absolute_path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.ball.dataset.datamodule import TennisBallDataModule
from src.ball.trainer.video_trainer import KeypointModule
from src.ball.models.cat_frames.lite_tracknet import LiteBallTracker
from src.ball.models.cat_frames.tracknet import BallTrackerNet
from src.utils.load_model import load_model_weights

@hydra.main(version_base="1.1", config_path="configs/train", config_name="ball")
def main(cfg):
    # ─── DataModule の準備 ───
    dm = TennisBallDataModule(
        annotation_file=to_absolute_path(cfg.annotation_file),
        image_root=to_absolute_path(cfg.image_root),
        T=cfg.T,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        input_size=cfg.input_size,
        heatmap_size=cfg.heatmap_size,
        skip_frames_range=cfg.skip_frames_range,
        input_type=cfg.input_type,
        output_type=cfg.output_type
    )
    dm.setup()

    # ─── モデル（MobileNet-U-HeatmapNet） の準備 ───
    tracknet = BallTrackerNet()
    lite_tracknet = LiteBallTracker()
    model_name = ["tracknet", "lite_tracknet"]

    # ─── LightningModule の準備 ───
    lit_model = KeypointModule(
        model=lite_tracknet,
    )

    # ─── コールバック ───
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="lite_tracknet-{epoch:02d}-{val_loss:.4f}"
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # ─── Trainer の起動 ───
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        default_root_dir=to_absolute_path(cfg.trainer.default_root_dir),
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # ─── 学習／検証実行 ───
    trainer.fit(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()
