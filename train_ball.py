import hydra
from hydra.utils import to_absolute_path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.ball.dataset.datamodule import TennisBallDataModule
from src.ball.trainer.video_trainer import KeypointModule
from src.ball.models.video.mobile_gru_unet import MobileNetUHeatmapNet, MobileNetUHeatmapWrapper, TemporalHeatmapModel
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
    pretrained = MobileNetUHeatmapNet(
        in_channels=3,
        base_channels=cfg.model.base_ch,
        out_channels=1,
        repeats=cfg.model.repeats,
        expansion=cfg.model.expansion,
        use_se=cfg.model.use_se,
        act=cfg.model.act
    )
    pretrained = load_model_weights(pretrained, ckpt_path=cfg.backbone_ckpt)
    wapper = MobileNetUHeatmapWrapper(pretrained)
    model = TemporalHeatmapModel(wapper, hidden_dim=cfg.model.base_ch * 8)

    # ─── LightningModule の準備 ───
    lit_model = KeypointModule(
        model=model,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        use_visibility_weighting=cfg.model.use_visibility_weighting
    )

    # ─── コールバック ───
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="mobilenet_temporal-{epoch:02d}-{val_loss:.4f}"
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

    # ─── 必要ならテストも実行 ───
    if cfg.trainer.do_test:
        trainer.test(lit_model, datamodule=dm)

if __name__ == "__main__":
    main()
