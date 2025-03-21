import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.train.dataloader.datamodule import DataModule
from src.train.model_trainer.parent import MNISTLightningModule
from src.train.model_trainer.child import ChildModelTrainer

def get_latest_checkpoint(directory):
    """ 指定したディレクトリ内で最新のチェックポイントを取得 """
    if not os.path.exists(directory):
        return None
    checkpoints = [f for f in os.listdir(directory) if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    checkpoints.sort(reverse=True)  # 一番新しいものを取得
    return os.path.join(directory, checkpoints[0])

def train():
    datamodule = DataModule()

    model_names = ['simple_conv', 'maxpool', 'inception']
    
    # Parent Model の学習をスキップし、child モデルから再開
    print("Parentモデルの学習は完了しているためスキップします。Childモデルのトレーニングを開始します。")

    # Child model training with knowledge distillation
    checkpoint_kd_path = get_latest_checkpoint('checkpoints/child_model_kd/')
    child_trainer_kd = ChildModelTrainer(num_classes=10, use_kd=True)
    child_checkpoint_kd = ModelCheckpoint(
        dirpath='checkpoints/child_model_kd/',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[child_checkpoint_kd],
        logger=pl.loggers.TensorBoardLogger("tb_logs", name="child_model_kd")
    )
    
    if checkpoint_kd_path:
        print(f"蒸留モデルの学習を再開: {checkpoint_kd_path}")
        trainer.fit(child_trainer_kd, datamodule=datamodule, ckpt_path=checkpoint_kd_path)
    else:
        print("蒸留モデルの学習を新規開始")
        trainer.fit(child_trainer_kd, datamodule=datamodule)

    # Child model training without knowledge distillation
    checkpoint_no_kd_path = get_latest_checkpoint('checkpoints/child_model_no_kd/')
    child_trainer_no_kd = ChildModelTrainer(num_classes=10, use_kd=False)
    child_checkpoint_no_kd = ModelCheckpoint(
        dirpath='checkpoints/child_model_no_kd/',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[child_checkpoint_no_kd],
        logger=pl.loggers.TensorBoardLogger("tb_logs", name="child_model_no_kd")
    )

    if checkpoint_no_kd_path:
        print(f"非蒸留モデルの学習を再開: {checkpoint_no_kd_path}")
        trainer.fit(child_trainer_no_kd, datamodule=datamodule, ckpt_path=checkpoint_no_kd_path)
    else:
        print("非蒸留モデルの学習を新規開始")
        trainer.fit(child_trainer_no_kd, datamodule=datamodule)

if __name__ == "__main__":
    train()