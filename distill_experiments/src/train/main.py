import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.train.dataloader.datamodule import DataModule
from src.train.model_trainer.parent import ParentLightningModule
from src.train.model_trainer.child import ChildModelTrainer

def model_train(checkpoint_path, model_trainer, datamodule):
    # checkpoint_path: checkpoints/version_name/model_name.ckpt
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    checkpoint_dir = os.path.dirname(checkpoint_path)
    version_name = os.path.basename(checkpoint_dir)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}' + '-{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    logger = pl.loggers.TensorBoardLogger(
    save_dir="C:/temp/tb_logs",  # tb_logs 以下にログディレクトリを作成
    name=version_name    # 例: mnist
    )
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint],
        logger=logger
    )
    if os.path.exists(checkpoint_path):
        print(f"{model_name} モデルの学習を再開")
        trainer.fit(model_trainer, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        print(f"{model_name} モデルの学習を新規開始")
        trainer.fit(model_trainer, datamodule=datamodule)

def train(version_name, num_classes, in_channels, input_shape):
    datamodule = DataModule()

    # 親モデルの学習
    model_names = ['simple_conv', 'maxpool', 'inception']
    for model_name in model_names:
        model_trainer = ParentLightningModule(model_name,
                                              num_classes=num_classes,
                                              in_channels=in_channels
                                              )
        checkpoint_path = f'checkpoints/{version_name}/{model_name}.ckpt'
        model_train(checkpoint_path, model_trainer, datamodule)
    
    is_kd = True
    for _ in range(2):
        model_trainer = ChildModelTrainer(num_classes=num_classes,
                                          use_kd=is_kd,
                                          in_channels=in_channels,
                                          input_shape=input_shape
                                          )
        if is_kd:
            checkpoint_path = f'checkpoints/{version_name}/child_model_kd.ckpt'
        else:
            checkpoint_path = f'checkpoints/{version_name}/child_model.ckpt'
        model_train(checkpoint_path, model_trainer, datamodule)
        is_kd = False

if __name__ == "__main__":
    version_name='mnist'
    num_classes=10
    in_channels=1
    input_shape=(28, 28)
    train(version_name, num_classes, in_channels, input_shape)