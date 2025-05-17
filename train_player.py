import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.player.arguments.prepare_transform import prepare_transform
from src.player.dataset.datamodule import CocoDataModule
from src.player.trainers.detr import DetrBasedTrainer


@hydra.main(version_base="1.1", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # ==== モデル・プロセッサ ====
    processor = RTDetrImageProcessor.from_pretrained(cfg.model.model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        ignore_mismatched_sizes=True
    )

    # ==== Transform ====
    train_transform = prepare_transform()

    # ==== DataModule ====
    data_module = CocoDataModule(
        img_folder=cfg.data.img_folder,
        annotation_file=cfg.data.annotation_file,
        cat_id_map=cfg.data.cat_id_map,
        use_original_path=cfg.data.use_original_path,
        processor=processor,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=train_transform,
        val_test_transform=train_transform
    )

    # ==== Lightning Module ====
    lightning_module = DetrBasedTrainer(
        detr_based_model=model,
        lr=cfg.model.lr,
        lr_backbone=cfg.model.lr_backbone,
        weight_decay=cfg.model.weight_decay,
        optim_t_max=cfg.model.optim_t_max,
        min_lr=cfg.model.min_lr,
        num_freeze_epoch=cfg.model.num_freeze_epoch,
    )

    # ==== Trainer ====
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        filename=cfg.checkpoint.filename,
        save_last=cfg.checkpoint.save_last,
        dirpath="checkpoints/player",  # Hydraは作業ディレクトリを変えるので絶対パスが安全
    )
    ealry_stopping = EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, ealry_stopping]
    )

    # ==== トレーニング ====
    trainer.fit(lightning_module, data_module, ckpt_path=cfg.trainer.ckpt_path)



if __name__ == "__main__":
    main()
