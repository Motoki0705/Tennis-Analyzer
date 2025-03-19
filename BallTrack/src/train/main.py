import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor

from BallTrack.src.train.data.transforms import prepare_transforms
from BallTrack.src.train.model.model_trainer import ModelTrainer
from BallTrack.src.train.data.datamodule import DataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="TrackNet/datasets/Tennis")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--t_max", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="BallTrack/checkpoints")
    parser.add_argument("--tb_dir", type=str, default='BallTrack/tb_logs')
    parser.add_argument("--model_name", type=str, default='seg_simple', help='using model name')
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="チェックポイントから再開")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name, f'version_{args.version}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    #tensorboardを初期化
    logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.model_name, version=args.version)

    # callbacksを初期化
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            monitor='val_loss',
            verbose=True,
            save_last=True,
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        ),
        RichProgressBar(refresh_rate=10, leave=True),
        LearningRateMonitor()
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    train_transform, val_test_transfrom = prepare_transforms(resize_shape=(512, 512))
    model = ModelTrainer(
        orig_channels=9,
        num_keypoints=1,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        t_max=args.t_max,
        extractor='segformer',
        upsampler='simple'
    )
    datamodule = DataModule(
        data_path=args.dataset_dir,
        train_transform=train_transform,
        val_test_transform=val_test_transfrom,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)