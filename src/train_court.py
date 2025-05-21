import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.court.dataset.datamodule import CourtDataModule
from src.court.models.lite_tracknet import LiteBallTracker
from src.court.trainer.cnn import CourtLModule


@hydra.main(version_base="1.1", config_path="configs/train", config_name="court")
def main(cfg: DictConfig):
    # データモジュール（データセットパスは空欄でユーザーが指定）

    datamodule = CourtDataModule(
        annotation_root=to_absolute_path(cfg.annotation_root),
        image_root=to_absolute_path(cfg.image_root),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        input_size=[360, 640],
        heatmap_size=[360, 640],
        default_keypoints=cfg.default_keypoints,
        is_each_keypoint=cfg.is_each_keypoint,
        sigma=cfg.sigma,
    )

    # モデル
    model = LiteBallTracker(in_channels=3, heatmap_channels=15)

    # LightningModule
    lmodule = CourtLModule(
        model=model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        optim_t_max=cfg.optim_t_max,
        min_lr=cfg.min_lr,
    )

    # ─── コールバック設定 ─────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.output_dir,
        filename="court-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")
    ealry_stopping = EarlyStopping(monitor="val_loss")

    # ─── Trainer 作成 ──────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        log_every_n_steps=cfg.log_every_n_steps,
        default_root_dir=cfg.output_dir,
        callbacks=[checkpoint_cb, lr_monitor_cb, ealry_stopping],
    )

    # トレーニング実行
    trainer.fit(lmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
