import json
import os
from pathlib import Path
import copy
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# --- 設定 ---
ORIGINAL_COCO_PATH = "data/annotation_jsons/coco_annotations.json" # ボール情報など元のデータ
PERSON_ANNOTATIONS_PATH = "data/annotation_jsons/raw_non_player_annotations_conditional_detr_batched.json" # 人物検出結果
OUTPUT_MERGED_COCO_PATH = "data/annotation_jsons/coco_annotations_with_all_persons_as_nonplayers.json" # マージ後の出力ファイル

# --- 必要なカテゴリ定義 ---
# これらのカテゴリが出力に含まれるようにする
REQUIRED_CATEGORIES = {
    1: {"name": "ball", "supercategory": "sports"},
    2: {"name": "player", "supercategory": "person"},
    3: {"name": "non_player_person", "supercategory": "person"}
}
# 人物検出結果JSONに含まれるカテゴリID (person_detectorで使ったものと合わせる)
PERSON_DETECTION_CATEGORY_ID = 3


class CocoMerger:
    """
    2つのCOCO形式JSONファイルをマージするクラス。
    - 元データ(ボール等) + 人物検出結果
    - カテゴリ定義を結合・追加
    - アノテーションIDの重複を解決
    """
    def __init__(self, original_path: str, person_path: str, output_path: str):
        self.original_path = Path(original_path)
        self.person_path = Path(person_path)
        self.output_path = Path(output_path)

        self.original_data: Optional[Dict] = None
        self.person_data: Optional[Dict] = None
        self.merged_data: Dict[str, Any] = {
            "info": {}, "licenses": [], "images": [],
            "categories": [], "annotations": []
        }

    def load_data(self) -> bool:
        """入力JSONファイルをロード"""
        print("Loading input JSON files...")
        try:
            if not self.original_path.is_file():
                print(f"Error: Original COCO file not found at {self.original_path}")
                return False
            with open(self.original_path, 'r', encoding='utf-8') as f:
                self.original_data = json.load(f)
            print(f"Loaded original data from {self.original_path}")

            if not self.person_path.is_file():
                 print(f"Error: Person annotations file not found at {self.person_path}")
                 return False
            with open(self.person_path, 'r', encoding='utf-8') as f:
                self.person_data = json.load(f)
            print(f"Loaded person annotations from {self.person_path}")
            return True
        except Exception as e:
            print(f"Error loading JSON files: {e}")
            return False

    def merge_info_licenses_images(self):
        """info, licenses, images セクションをマージ (元データを優先)"""
        print("Merging info, licenses, and images...")
        # 基本的に元のデータをそのまま使う
        self.merged_data["info"] = self.original_data.get("info", {})
        self.merged_data["licenses"] = self.original_data.get("licenses", [])
        self.merged_data["images"] = self.original_data.get("images", []) # 画像情報は元データにあるはず
        print(f"Using info, licenses, and {len(self.merged_data['images'])} images from original data.")

    def merge_categories(self):
        """カテゴリ定義をマージし、不足している必須カテゴリを追加"""
        print("Merging categories...")
        merged_categories: Dict[int, Dict] = {} # IDをキーにして重複を防ぐ

        # 1. 元データのカテゴリを追加
        for cat in self.original_data.get("categories", []):
            if cat.get('id') is not None and cat.get('id') not in merged_categories:
                 merged_categories[cat['id']] = cat

        # 2. 人物検出データのカテゴリを追加 (ID=3 のはず)
        #    人物検出データに含まれるカテゴリIDを特定
        person_cat_id = None
        for cat in self.person_data.get("categories", []):
             cat_id = cat.get('id')
             if cat_id is not None:
                  person_cat_id = cat_id # 人物検出で使ったカテゴリIDを保存
                  if cat_id not in merged_categories:
                       merged_categories[cat_id] = cat
                  break # 通常カテゴリは1つのはず
        # もし person_detector で使ったIDが定数と違う場合は警告を出すか、上書きする
        if person_cat_id is not None and person_cat_id != PERSON_DETECTION_CATEGORY_ID:
             print(f"Warning: Person detection file uses category ID {person_cat_id}, expected {PERSON_DETECTION_CATEGORY_ID}. Using ID {person_cat_id} from file.")
             # 必要なら REQUIRED_CATEGORIES を更新
             if PERSON_DETECTION_CATEGORY_ID in REQUIRED_CATEGORIES:
                 del REQUIRED_CATEGORIES[PERSON_DETECTION_CATEGORY_ID]
             REQUIRED_CATEGORIES[person_cat_id] = {"name": "non_player_person", "supercategory": "person"}


        # 3. 必須カテゴリが不足していれば追加
        for req_id, req_info in REQUIRED_CATEGORIES.items():
            if req_id not in merged_categories:
                print(f"Info: Required category ID {req_id} ('{req_info['name']}') not found. Adding.")
                merged_categories[req_id] = {
                    "id": req_id,
                    "name": req_info["name"],
                    "supercategory": req_info["supercategory"]
                }

        # 最終的なカテゴリリストを作成 (ID順にソート)
        self.merged_data["categories"] = sorted(list(merged_categories.values()), key=lambda x: x['id'])
        print(f"Merged categories (Total: {len(self.merged_data['categories'])}):")
        for cat in self.merged_data["categories"]:
            print(f"  ID: {cat['id']}, Name: {cat['name']}")

    def merge_annotations(self):
        """アノテーションをマージし、IDの衝突を解決"""
        print("Merging annotations...")
        original_annotations = self.original_data.get("annotations", [])
        person_annotations = self.person_data.get("annotations", [])

        # 1. 元のアノテーションの最大IDを見つける
        max_original_id = 0
        if original_annotations:
            max_original_id = max(ann.get('id', 0) for ann in original_annotations)
        print(f"Max annotation ID in original data: {max_original_id}")

        # 2. 新しいアノテーションIDの開始番号を決定
        next_annotation_id = max_original_id + 1

        # 3. 元のアノテーションをそのまま追加
        merged_annotations = copy.deepcopy(original_annotations) # コピーしておく

        # 4. 人物検出アノテーションのIDを変更して追加
        print(f"Re-assigning IDs for {len(person_annotations)} person annotations starting from {next_annotation_id}...")
        reassigned_count = 0
        for ann in person_annotations:
            original_ann_id = ann.get('id') # 元のIDはログ用に保持しても良い
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = next_annotation_id # 新しいIDを割り当て
            # category_id が正しいか確認 (PersonDetectorで作ったIDと一致するはず)
            if new_ann.get('category_id') != PERSON_DETECTION_CATEGORY_ID:
                 print(f"Warning: Annotation ID {original_ann_id} has unexpected category ID {new_ann.get('category_id')}, expected {PERSON_DETECTION_CATEGORY_ID}. Keeping original category ID.")
                 # 強制的に上書きする場合:
                 # new_ann['category_id'] = PERSON_DETECTION_CATEGORY_ID

            merged_annotations.append(new_ann)
            next_annotation_id += 1
            reassigned_count += 1

        print(f"Re-assigned IDs for {reassigned_count} person annotations.")
        self.merged_data["annotations"] = merged_annotations
        print(f"Total annotations after merge: {len(self.merged_data['annotations'])}")

    def save_merged_coco(self):
        """マージ結果をJSONファイルとして保存"""
        print(f"\nSaving merged COCO data to: {self.output_path}")
        try:
            # 更新日時をinfoに追加
            self.merged_data["info"]["date_merged"] = datetime.now().isoformat()
            self.output_path.parent.mkdir(parents=True, exist_ok=True) # ディレクトリ作成
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.merged_data, f, indent=4)
            print("Successfully saved the merged COCO JSON file.")
        except Exception as e:
            print(f"Error saving merged COCO file: {e}")

    def run(self):
        """マージプロセス全体を実行"""
        if not self.load_data():
            return
        self.merge_info_licenses_images()
        self.merge_categories()
        self.merge_annotations()
        self.save_merged_coco()

# --- 実行部分 ---
if __name__ == "__main__":
    print("--- COCO Dataset Merger Script ---")
    print(f"Original COCO: {ORIGINAL_COCO_PATH}")
    print(f"Person Annotations: {PERSON_ANNOTATIONS_PATH}")
    print(f"Output Merged COCO: {OUTPUT_MERGED_COCO_PATH}")
    print("-" * 30)

    merger = CocoMerger(
        original_path=ORIGINAL_COCO_PATH,
        person_path=PERSON_ANNOTATIONS_PATH,
        output_path=OUTPUT_MERGED_COCO_PATH
    )
    merger.run()

    print("\n--- Merge Script Finished ---")