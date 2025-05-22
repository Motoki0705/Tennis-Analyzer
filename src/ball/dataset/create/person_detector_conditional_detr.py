import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection

# --- 定数 ---
DEFAULT_PERSON_CLASS_ID = 1
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
INITIAL_CATEGORY_ID = 3
INITIAL_CATEGORY_NAME = "non_player_person"
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4

# 描画用定数
BBOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (255, 0, 0)


class PersonCocoDataset(Dataset):
    """
    COCO フォーマットの images リストを受け取り、
    (PIL.Image, image_id, img_info) を返す Dataset。
    """

    def __init__(self, images_info: List[Dict[str, Any]], image_root_dir: str):
        self.images_info = images_info
        self.root = Path(image_root_dir)

    def __len__(self) -> int:
        return len(self.images_info)

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[Image.Image, int, Dict[str, Any]]]:
        img_info = self.images_info[idx]
        image_id = img_info["id"]
        relative_path = img_info.get("original_path", img_info.get("file_name"))
        img_path = self.root / relative_path

        try:
            img = Image.open(img_path).convert("RGB")
            return img, image_id, img_info
        except (FileNotFoundError, UnidentifiedImageError):
            return None


def collate_fn(batch):
    # None を除外
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, image_ids, img_infos = zip(*batch, strict=False)
    return {
        "images": list(images),
        "image_ids": list(image_ids),
        "img_infos": list(img_infos),
    }


class PersonDetectorConditionalDETR:
    """
    COCO 形式の画像エントリをバッチ処理し、
    Conditional DETR で人物検出 → COCO アノテーション出力。
    """

    def __init__(
        self,
        input_coco_path: Optional[str] = None,
        image_root_dir: Optional[str] = None,
        output_coco_path: str = "raw_non_player_annotations_conditional_detr.json",
        model_name: str = "microsoft/conditional-detr-resnet-50",
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        start_annotation_id: int = 1,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        # パス設定
        self.input_coco_path = Path(input_coco_path) if input_coco_path else None
        self.image_root_dir = Path(image_root_dir) if image_root_dir else None
        self.output_coco_path = Path(output_coco_path)

        # 検出・バッチ設定
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.annotation_id_counter = start_annotation_id

        # 出力用データ構造
        self.output_coco_data: Dict[str, Any] = {
            "info": {},
            "licenses": [],
            "images": [],
            "categories": [],
            "annotations": [],
        }
        self.input_images: List[Dict[str, Any]] = []
        self.person_class_id: int = DEFAULT_PERSON_CLASS_ID

        # モデル・プロセッサのロード
        print(f"Loading Conditional DETR model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ConditionalDetrForObjectDetection.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._find_person_class_id()

    def _find_person_class_id(self):
        id2label = getattr(self.model.config, "id2label", {})
        for k, v in id2label.items():
            try:
                if int(k) and v.lower() == "person":
                    self.person_class_id = int(k)
                    print(f"Detected 'person' class_id = {self.person_class_id}")
                    return
            except:
                continue
        print(f"Using default person_class_id = {self.person_class_id}")

    def _load_input_coco(self) -> bool:
        if not self.input_coco_path or not self.input_coco_path.is_file():
            print("No valid input COCO path.")
            return False
        with open(self.input_coco_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.output_coco_data["info"] = data.get("info", self._default_info())
        self.output_coco_data["licenses"] = data.get("licenses", [])
        self.input_images = data.get("images", [])
        print(f"Loaded {len(self.input_images)} images from input COCO.")
        return True

    def _default_info(self) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "description": "Batch Person Detections (Conditional DETR)",
            "version": "1.0",
            "year": now.year,
            "contributor": "PersonDetectorConditionalDETR",
            "date_created": now.isoformat(),
        }

    def _prepare_categories(self):
        self.output_coco_data["categories"] = [
            {
                "id": INITIAL_CATEGORY_ID,
                "name": INITIAL_CATEGORY_NAME,
                "supercategory": "person",
            }
        ]
        print(
            f"Prepared output category id={INITIAL_CATEGORY_ID}, name='{INITIAL_CATEGORY_NAME}'"
        )

    def _create_coco_annotation(
        self, detection: Dict[str, Any], image_id: int, annotation_id: int
    ) -> Dict[str, Any]:
        x, y, w, h = detection["bbox"]
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": INITIAL_CATEGORY_ID,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "score": detection.get("score", None),
        }

    def process_images(self):
        if not self.input_images or not self.image_root_dir:
            print("Error: 入力データまたは画像ディレクトリが未設定です。")
            return

        # DataLoader の準備
        dataset = PersonCocoDataset(self.input_images, str(self.image_root_dir))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

        total_detections = 0
        print(
            f"\nStart batched detection: {len(dataset)} images, batch_size={self.batch_size}, num_workers={self.num_workers}"
        )

        for batch_idx, batch in enumerate(loader, start=1):
            if batch is None:
                print(f"  Batch {batch_idx}: 全ての画像読み込みが失敗したためスキップ")
                continue

            pil_images = batch["images"]
            image_ids = batch["image_ids"]
            img_infos = batch["img_infos"]

            # images エントリを出力に追加
            self.output_coco_data["images"].extend(img_infos)

            # 前処理 → 推論
            inputs = self.processor(images=pil_images, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 後処理
            sizes = [img.size[::-1] for img in pil_images]  # (H, W)
            target_sizes = torch.tensor(sizes, device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.conf_threshold
            )

            # 検出結果をアノテーション化
            for res, img_id in zip(results, image_ids, strict=False):
                for score, label, box in zip(
                    res["scores"], res["labels"], res["boxes"], strict=False
                ):
                    if label.item() == self.person_class_id:
                        xmin, ymin, xmax, ymax = box.tolist()
                        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                        ann = self._create_coco_annotation(
                            {
                                "bbox": [float(x) for x in bbox],
                                "score": float(score.item()),
                            },
                            image_id=img_id,
                            annotation_id=self.annotation_id_counter,
                        )
                        self.output_coco_data["annotations"].append(ann)
                        self.annotation_id_counter += 1
                        total_detections += 1

            print(
                f"  Batch {batch_idx}: 処理完了 {len(pil_images)} 枚, 累計検出数 {total_detections}"
            )

        print(f"\n全バッチ処理完了: 合計検出数 = {total_detections}")

    def save_output_coco(self):
        self.output_coco_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_coco_path, "w", encoding="utf-8") as f:
            json.dump(self.output_coco_data, f, indent=4)
        print(f"Saved output COCO to {self.output_coco_path}")

    def run(self):
        loaded = True
        if self.input_coco_path:
            loaded = self._load_input_coco()
        self._prepare_categories()
        if loaded:
            self.process_images()
            self.save_output_coco()
        else:
            print("Aborting: 入力 COCO の読み込みに失敗しました。")


if __name__ == "__main__":
    # --- 設定 ---
    INPUT_COCO = "data/annotation_jsons/coco_annotations.json"
    IMAGE_DIR = "data/images"
    OUTPUT_COCO = (
        "data/annotation_jsons/raw_non_player_annotations_conditional_detr_batched.json"
    )
    MODEL_NAME = "microsoft/conditional-detr-resnet-50"
    CONF_TH = 0.5
    START_ANN_ID = 1
    BATCH_SIZE = 32
    NUM_WORKERS = 8

    print("--- Person Detection (Conditional DETR, Batched) ---")
    print(f"Input COCO:   {INPUT_COCO}")
    print(f"Image Dir:    {IMAGE_DIR}")
    print(f"Output COCO:  {OUTPUT_COCO}")
    print(f"Model:        {MODEL_NAME}")
    print(f"Threshold:    {CONF_TH}")
    print(f"Batch Size:   {BATCH_SIZE}")
    print(f"Num Workers:  {NUM_WORKERS}")
    print("-" * 40)

    detector = PersonDetectorConditionalDETR(
        input_coco_path=INPUT_COCO,
        image_root_dir=IMAGE_DIR,
        output_coco_path=OUTPUT_COCO,
        model_name=MODEL_NAME,
        conf_threshold=CONF_TH,
        start_annotation_id=START_ANN_ID,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    detector.run()
