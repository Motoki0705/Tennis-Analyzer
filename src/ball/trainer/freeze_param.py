from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BaseFinetuning

from src.models.cat_frames.swin_448 import SwinCourtUNet
from src.trainer.cat_frames_trainer import CatFramesLModule


class ResumeSafeFreezeBackbone(BaseFinetuning):
    def __init__(self, strategy: dict):
        super().__init__()
        self.freeze_strategy = strategy["freeze"]
        self.unfreeze_strategy = strategy["unfreeze"]

    def freeze_before_training(self, pl_module: pl.LightningModule):
        layer = self._get_layer(pl_module, self.freeze_strategy["before_training"])
        self.freeze(layer)

    def on_fit_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        # 🔁 再開時：過去にunfreezeされた層を再度unfreeze & optimizerに登録
        for epoch, layer_path in self.unfreeze_strategy.items():
            if epoch <= current_epoch:
                layer = self._get_layer(pl_module, layer_path)
                self.unfreeze_and_add_param_group(
                    modules=layer,
                    optimizer=trainer.optimizers[0],
                    train_bn=True,
                )

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.unfreeze_strategy:
            layer_path = self.unfreeze_strategy[epoch]
            layer = self._get_layer(pl_module, layer_path)
            self.unfreeze_and_add_param_group(
                modules=layer,
                optimizer=trainer.optimizers[0],
                train_bn=True,
            )

    def _get_layer(self, pl_module, layer_attr):
        module = pl_module
        for attr in layer_attr.split("."):
            module = getattr(module, attr)
        return module


if __name__ == "__main__":
    strategy = {
        "freeze": {"before_training": "model.backbone"},
        "unfreeze": {
            1: "model.backbone.patch_embed",
            2: "model.backbone.layers_3",
            3: "model.backbone",
        },
    }
    freeze_callback = FreezeBackbone(strategy)
    model = SwinCourtUNet(num_keypoints=1)
    pl_module = CatFramesLModule(model=model)
    freeze_callback.freeze_before_training(pl_module)
    for name, p in pl_module.model.backbone.named_parameters():
        if not p.requires_grad:  # True なら凍結済
            print("frozen :", name)
        else:
            print("train  :", name)

    for unfreeze_epoch, unfreeze_layer in strategy["unfreeze"].items():
        dummy_trainer = SimpleNamespace(
            current_epoch=unfreeze_epoch,
            optimizers=[
                torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
                )
            ],
        )
        freeze_callback.on_train_epoch_start(dummy_trainer, pl_module)
        for name, p in pl_module.model.backbone.named_parameters():
            if not p.requires_grad:  # True なら凍結済
                print("frozen :", name)
            else:
                print("train  :", name)
        print("\n\n")
