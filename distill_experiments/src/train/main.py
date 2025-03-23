import os
import glob

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.train.dataloader.datamodule import DataModule
from src.train.model_trainer.parent import ParentLightningModule
from src.train.model_trainer.child import ChildModelTrainer

def find_latest_checkpoint(checkpoint_dir: str, model_name: str):
    """
    指定ディレクトリ内の model_name に対応する最新（= val_lossが最も小さい）チェックポイントを返す。
    """
    pattern = os.path.join(checkpoint_dir, f"{model_name}-epoch=*-val_loss=*.ckpt")
    candidates = glob.glob(pattern)
    
    if not candidates:
        return None
    
    # val_lossの値を抽出して最小のものを選ぶ
    def extract_val_loss(filepath):
        filename = os.path.basename(filepath)
        try:
            val_loss_str = filename.split("val_loss=")[-1].replace(".ckpt", "")
            return float(val_loss_str)
        except Exception:
            return float("inf")

    best_ckpt = min(candidates, key=extract_val_loss)
    return best_ckpt

def model_train(checkpoint_path, model_trainer, datamodule, max_epochs, log_dir="C:/temp/tb_logs", skip_if_trained=False):
    """
    汎用の学習実行関数です。
    - checkpoint_path が存在し、skip_if_trained=True ならば学習をスキップします。
    - 存在する場合は再開、なければ新規学習を行います。
    """
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    checkpoint_dir = os.path.dirname(checkpoint_path)
    version_name = os.path.basename(checkpoint_dir)
    
    if skip_if_trained and os.path.exists(checkpoint_path):
        print(f"{model_name} のチェックポイントが存在するため、学習をスキップします: {checkpoint_path}")
        return

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}' + '-{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=version_name
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    
    if os.path.exists(checkpoint_path):
        print(f"{model_name} のチェックポイントから学習を再開します: {checkpoint_path}")
        trainer.fit(model_trainer, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        print(f"{model_name} の新規学習を開始します")
        trainer.fit(model_trainer, datamodule=datamodule)

def train_parent_models(new_version_name, prev_version_name, num_classes, in_channels, parent_max_epochs, train_parent=True, log_dir="C:/temp/tb_logs"):
    """
    親モデルの学習を実施またはスキップします。
    ・train_parent=True の場合は、新たに new_version_name 配下へ学習を実施します。
    ・train_parent=False の場合は、prev_version_name 配下のチェックポイントを教師モデルとして利用します。
    
    戻り値として、各親モデルのチェックポイントパスを辞書形式で返します。
    """
    datamodule = DataModule()
    parent_model_names = ['simple_conv', 'maxpool', 'inception']
    teacher_checkpoint_paths = {}
    
    for model_name in parent_model_names:
        if train_parent:
            # 新たなバージョンで親モデルを学習
            checkpoint_path = f'checkpoints/{new_version_name}/{model_name}.ckpt'
            model_trainer = ParentLightningModule(model_name=model_name,
                                                  num_classes=num_classes,
                                                  in_channels=in_channels)
            model_train(checkpoint_path, model_trainer, datamodule, max_epochs=parent_max_epochs, log_dir=log_dir)
            teacher_checkpoint_paths[model_name] = checkpoint_path
        else:
            # 前回のバージョンのチェックポイントを教師モデルとして利用
            checkpoint_path = f'checkpoints/{prev_version_name}'
            find_ckpt = find_latest_checkpoint(checkpoint_path, model_name)
            if not os.path.exists(checkpoint_path):
                print(f"警告: {model_name} の教師モデルのチェックポイントが見つかりません: {checkpoint_path}")
            else:
                teacher_checkpoint_paths[model_name] = find_ckpt
    return teacher_checkpoint_paths

def train_child_models(new_version_name, num_classes, in_channels, input_shape, child_max_epochs, teacher_checkpoint_paths, log_dir="C:/temp/tb_logs"):
    """
    子モデルの学習を実施します。
    ・KD（知識蒸留）ありの場合となしの場合で学習を実施し、それぞれ checkpoint に保存します。
    ・KDありの場合は teacher_checkpoint_paths を ChildModelTrainer に渡します。
    """
    datamodule = DataModule()
    
    # 知識蒸留ありの子モデル
    kd_checkpoint_path = f'checkpoints/{new_version_name}/child_model_kd.ckpt'
    child_model_trainer_kd = ChildModelTrainer(num_classes=num_classes,
                                               use_kd=True,
                                               in_channels=in_channels,
                                               input_shape=input_shape,
                                               teacher_checkpoint_paths=teacher_checkpoint_paths)
    model_train(kd_checkpoint_path, child_model_trainer_kd, datamodule, max_epochs=child_max_epochs, log_dir=log_dir)
    
    # 知識蒸留なしの子モデル
    std_checkpoint_path = f'checkpoints/{new_version_name}/child_model.ckpt'
    child_model_trainer_std = ChildModelTrainer(num_classes=num_classes,
                                                use_kd=False,
                                                in_channels=in_channels,
                                                input_shape=input_shape)
    model_train(std_checkpoint_path, child_model_trainer_std, datamodule, max_epochs=child_max_epochs, log_dir=log_dir)

def train_all(new_version_name, prev_version_name, num_classes, in_channels, input_shape,
              train_parent=True, parent_max_epochs=3, child_max_epochs=3, log_dir="C:/temp/tb_logs"):
    """
    ・new_version_name: 今回学習する際のバージョン名（子モデル含む保存先）
    ・prev_version_name: 既存の親モデルのチェックポイントが保存されているバージョン名
    ・train_parent: True の場合、親モデルを新たに学習し、False の場合は prev_version_name のチェックポイントを使用
    その他、エポック数などはパラメータ化しています。
    """
    teacher_checkpoint_paths = train_parent_models(new_version_name, prev_version_name, num_classes, in_channels, parent_max_epochs, train_parent, log_dir)
    train_child_models(new_version_name, num_classes, in_channels, input_shape, child_max_epochs, teacher_checkpoint_paths, log_dir)

if __name__ == "__main__":
    # 実行例
    new_version_name = 'mnist_add_alpha'      # 今回学習するバージョン
    prev_version_name = 'mnist'            # 既存の親モデルのチェックポイントがあるバージョン
    num_classes = 10
    in_channels = 1
    input_shape = (28, 28)
    train_parent = False        # 親モデルも新たに学習する場合 True、既存のチェックポイントを利用する場合 False
    parent_max_epochs = 5
    child_max_epochs = 5
    
    train_all(new_version_name, prev_version_name, num_classes, in_channels, input_shape,
              train_parent=train_parent, parent_max_epochs=parent_max_epochs, child_max_epochs=child_max_epochs)
