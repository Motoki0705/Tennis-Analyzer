import pytest
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
import sys
import os
from unittest.mock import patch, MagicMock

# プロジェクトルートをimportパスに追加（srcなどをimport可能にする）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# テストパラメータの候補
CAT_ID_MAPS = [{2: 0}, {1: 0}]  # カテゴリIDのマッピング（複雑な辞書は避ける）
BATCH_SIZES = [1, 2]                  # バッチサイズ

class DummyCocoDetection:
    """テスト用のダミーCocoDetectionデータセット"""
    def __init__(self, img_folder, annotation_file, cat_id_map, use_original_path, split, transform, size=10):
        self.img_folder = img_folder
        self.annotation_file = annotation_file
        self.cat_id_map = cat_id_map
        self.use_original_path = use_original_path
        self.split = split
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # ダミー画像とアノテーションを返す
        import numpy as np
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # COCOフォーマットのアノテーション
        target = {
            "image_id": idx,
            "annotations": [
                {
                    "id": 100 + idx,
                    "category_id": list(self.cat_id_map.keys())[0],  # 最初のキーを使用
                    "bbox": [50, 50, 100, 100],  # [x, y, width, height]
                    "area": 10000,
                    "iscrowd": 0
                }
            ]
        }
        
        # 変換を適用
        if self.transform:
            transformed = self.transform(image=image, bboxes=[ann["bbox"] for ann in target["annotations"]],
                                        category_id=[ann["category_id"] for ann in target["annotations"]])
            image = transformed["image"]
            
            # 変換後のbboxを更新
            for i, (bbox, cat_id) in enumerate(zip(transformed["bboxes"], transformed["category_id"])):
                target["annotations"][i]["bbox"] = bbox
                target["annotations"][i]["category_id"] = cat_id
        
        return image, target

class DummyProcessor:
    """テスト用のダミープロセッサ"""
    def __call__(self, images, annotations, return_tensors="pt"):
        batch_size = len(images)
        
        # ダミーのエンコーディング結果を返す
        pixel_values = torch.rand(batch_size, 3, 224, 224)
        pixel_mask = torch.ones(batch_size, 224, 224)
        
        # ダミーのラベル情報
        labels = []
        for i, ann in enumerate(annotations):
            boxes = []
            class_labels = []
            
            for a in ann["annotations"]:
                # [x, y, width, height] -> [center_x, center_y, width, height]
                x, y, w, h = a["bbox"]
                cx = x + w/2
                cy = y + h/2
                boxes.append([cx/224, cy/224, w/224, h/224])  # 正規化
                class_labels.append(0)  # 0: player
            
            labels.append({
                "boxes": torch.tensor(boxes),
                "class_labels": torch.tensor(class_labels)
            })
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels
        }

@pytest.mark.parametrize("cat_id_map", CAT_ID_MAPS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_dataset_io(cat_id_map, batch_size):
    """
    CocoDataModuleのI/O仕様（画像、バウンディングボックス）を検証する。
    - pixel_values: 入力画像テンソル
    - pixel_mask: ピクセルマスク
    - labels: バウンディングボックスとクラスラベル
    """

    # hydra設定の上書き
    overrides = [
        f"litdatamodule.batch_size={batch_size}",
    ]

    # DummyCocoDetectionとDummyProcessorを使用してテスト
    with patch('src.player.dataset.coco_dataset.CocoDetection', DummyCocoDetection), \
         patch('transformers.RTDetrImageProcessor.from_pretrained', return_value=DummyProcessor()):
        
        with initialize(version_base="1.3", config_path="../../configs/test/player", job_name="test_dataset_io_instance"):
            cfg = compose(config_name="config_dataset_test.yaml", overrides=overrides)
            
            # cat_id_mapを直接設定
            cfg.litdatamodule.cat_id_map = cat_id_map
            
            # DataModuleをインスタンス化
            datamodule = instantiate(cfg.litdatamodule)
            datamodule.prepare_data()
            datamodule.setup(stage="fit")
            
            # DataLoaderからバッチを取得
            train_loader = datamodule.train_dataloader()
            batch = next(iter(train_loader))

        # バッチの内容を検証
        assert "pixel_values" in batch, "Batch should contain pixel_values"
        assert "pixel_mask" in batch, "Batch should contain pixel_mask"
        assert "labels" in batch, "Batch should contain labels"
        
        # pixel_valuesの形状を検証
        pixel_values = batch["pixel_values"]
        assert pixel_values.dim() == 4, "pixel_values should be 4D: [B, C, H, W]"
        assert pixel_values.shape[0] == batch_size, f"Expected batch size {batch_size}, got {pixel_values.shape[0]}"
        assert pixel_values.shape[1] == 3, "Expected 3 channels"
        
        # pixel_maskの形状を検証
        pixel_mask = batch["pixel_mask"]
        assert pixel_mask.dim() == 3, "pixel_mask should be 3D: [B, H, W]"
        assert pixel_mask.shape[0] == batch_size, f"Expected batch size {batch_size}, got {pixel_mask.shape[0]}"
        
        # labelsの内容を検証
        labels = batch["labels"]
        assert len(labels) == batch_size, f"Expected {batch_size} label entries, got {len(labels)}"
        
        for label_dict in labels:
            assert "boxes" in label_dict, "Label should contain boxes"
            assert "class_labels" in label_dict, "Label should contain class_labels"
            
            boxes = label_dict["boxes"]
            class_labels = label_dict["class_labels"]
            
            assert boxes.dim() == 2, "boxes should be 2D: [N, 4]"
            assert boxes.shape[1] == 4, "boxes should have 4 coordinates"
            assert class_labels.dim() == 1, "class_labels should be 1D"
            assert boxes.shape[0] == class_labels.shape[0], "Number of boxes and labels should match"
            
            # 正規化されたバウンディングボックスの検証
            assert torch.all(boxes >= 0), "Normalized boxes should be >= 0"
            assert torch.all(boxes <= 1), "Normalized boxes should be <= 1"

        print(f"✅ Test passed: cat_id_map={cat_id_map}, batch_size={batch_size}")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 