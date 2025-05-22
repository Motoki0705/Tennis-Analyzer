import json
import os
from pathlib import Path

# 入力ファイルパス
TRAIN_PATH = "data/court/converted_train.json"
VAL_PATH = "data/court/converted_val.json"
# 出力ファイルパス
OUT_PATH = "data/court/coco_court.json"

# カテゴリ定義
CATEGORIES = [
    {"id": 1, "name": "court"}
]

def load_list_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # train/valを連結
    train_data = load_list_json(TRAIN_PATH)
    val_data = load_list_json(VAL_PATH)
    all_data = train_data + val_data

    images = []
    annotations = []
    image_id_map = {}  # file_name -> image_id
    next_image_id = 1
    next_ann_id = 1

    for item in all_data:
        file_name = item["file_name"]
        if file_name not in image_id_map:
            image_id = next_image_id
            image_id_map[file_name] = image_id
            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": item["width"],
                "height": item["height"]
            })
            next_image_id += 1
        else:
            image_id = image_id_map[file_name]

        annotations.append({
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": item["keypoints"],
            "num_keypoints": item["num_keypoints"]
        })
        next_ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)
    print(f"COCO形式で {OUT_PATH} に保存しました。画像数: {len(images)}, アノテーション数: {len(annotations)}")

if __name__ == "__main__":
    main() 