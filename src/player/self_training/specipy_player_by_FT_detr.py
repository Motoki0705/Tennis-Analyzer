import copy
import json
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from tqdm import tqdm


class _CocoImageDataset(Dataset):
    """DataLoader 用の軽量 Dataset — 画像 PIL と img_info を返す"""

    def __init__(self, image_infos: List[dict], images_root: Union[str, Path]):
        self.image_infos = image_infos
        self.images_root = Path(images_root)

    def __len__(self) -> int:
        return len(self.image_infos)

    def __getitem__(self, idx) -> Tuple[Image.Image, dict]:
        info = self.image_infos[idx]
        img_path = self.images_root / info["original_path"]
        img = Image.open(img_path).convert("RGB")
        return img, info


class PlayerSpecifierByDetr:
    """
    * 元 JSON をコピーして安全に編集
    * FT した conditional DETR をバッチで推論
    * IoU が閾値以上 & is_human_verified==False の bbox を player (=2) に変更
      + is_model_verified=True を付加
    """

    def __init__(
        self,
        input_json_path: Union[str, Path],
        output_json_path: Union[str, Path],
        images_root: Union[str, Path],
        model: torch.nn.Module,
        processor,
        device: torch.device,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        self.input_json_path = Path(input_json_path)
        self.output_json_path = Path(output_json_path)
        self.images_root = Path(images_root)

        # モデル / 推論設定
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

        # COCO 読み込み（コピーを保持して元は破壊しない）
        with open(self.input_json_path, "r") as f:
            self.original_data = json.load(f)
        self.coco_data = copy.deepcopy(self.original_data)

        # 画像とアノテーションを索引
        self.image_id_to_info = {img["id"]: img for img in self.coco_data["images"]}
        self.image_id_to_annotations = {}
        for ann in self.coco_data["annotations"]:
            self.image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

    # ---------- メイン処理 ----------
    def specify_players(self):
        dataset = _CocoImageDataset(
            image_infos=list(self.image_id_to_info.values()),
            images_root=self.images_root,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.coco_collate_fn,  # ([PIL…], [info…])
        )

        for imgs, infos in tqdm(loader, desc="Specifying players", total=len(loader)):
            # PIL → processor（size は各画像ごとに渡す）
            sizes = [(info["height"], info["width"]) for info in infos]
            encoding = self.processor(images=list(imgs), return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(pixel_values=encoding["pixel_values"])

            results_batch = self.processor.post_process_object_detection(
                outputs,
                threshold=self.score_threshold,
                target_sizes=sizes,
            )

            # 画像ごとに IoU マッチング
            for res, info in zip(results_batch, infos, strict=False):
                self._update_annotations(info, res)

    @staticmethod
    def coco_collate_fn(batch):
        return list(zip(*batch, strict=False))

    # ---------- IoU マッチングして category_id 更新 ----------
    def _update_annotations(self, img_info: dict, results: dict):
        image_id = img_info["id"]
        player_boxes = results["boxes"][results["labels"] == 0].cpu()  # player のみ
        if len(player_boxes) == 0:
            return

        anns = self.image_id_to_annotations.get(image_id, [])
        if not anns:
            return

        ex_boxes_xywh = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)
        ex_boxes_xyxy = self._xywh_to_xyxy(ex_boxes_xywh)
        ious = box_iou(player_boxes, ex_boxes_xyxy)  # [num_det, num_ann]

        for det_idx in range(len(player_boxes)):
            best_iou, best_ann_idx = torch.max(ious[det_idx], dim=0)
            if best_iou >= self.iou_threshold:
                ann = anns[best_ann_idx]
                if not ann.get("is_human_verified", False):
                    ann["category_id"] = 2  # player
                    ann["is_model_verified"] = True  # 追加フラグ

    # ---------- ユーティリティ ----------
    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
        boxes_xyxy[:, 3] += boxes_xyxy[:, 1]
        return boxes_xyxy

    def save(self):
        with open(self.output_json_path, "w") as f:
            json.dump(self.coco_data, f, indent=2)
