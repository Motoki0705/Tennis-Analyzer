# generate_final_dataset.py

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# --- 設定 ---
INPUT_MERGED_COCO = "data/annotation_jsons/coco_annotations_with_all_persons_as_nonplayers.json"  # マージ済み中間JSON
SPECIFICATION_DIR = "data/player_specification"  # 分類結果保存ディレクトリ
OUTPUT_FINAL_COCO = "data/annotation_jsons/coco_annotations_final.json"  # 最終出力JSON

# カテゴリID定義 (入力JSONと一致させる)
CAT_ID_BALL = 1
CAT_ID_PLAYER = 2
CAT_ID_NON_PLAYER = 3

# カスタムメタデータフィールド名
META_HUMAN_VERIFIED = "is_human_verified"


class FinalDatasetGenerator:
    """
    分類結果ファイル群と中間COCO JSONを結合し、最終データセットJSONを生成するクラス。
    手動検証フラグを追加 (Player指定されたアノテーションの画像IDを検証済みとする)。
    """

    def __init__(self, input_merged_coco: str, spec_dir: str, output_final_coco: str):
        self.input_merged_coco_path = Path(input_merged_coco)
        self.spec_dir = Path(spec_dir)
        self.output_final_coco_path = Path(output_final_coco)

        self.master_coco_data: Optional[Dict] = None
        self.all_player_ids: Set[int] = set()
        self.all_ignored_ids: Set[int] = set()
        self.verified_image_ids: Set[int] = set()  # 検証済み画像IDのセット

    def load_data(self) -> bool:
        """中間COCO JSONと分類結果ファイル群をロード"""
        # --- 中間COCO JSON ロード ---
        if not self.input_merged_coco_path.is_file():
            print(
                f"Error: Input merged COCO file not found at {self.input_merged_coco_path}"
            )
            return False
        print(f"Loading merged COCO data from: {self.input_merged_coco_path}")
        try:
            with open(self.input_merged_coco_path, "r", encoding="utf-8") as f:
                self.master_coco_data = json.load(f)
            print("Merged COCO data loaded.")
        except Exception as e:
            print(f"Error reading merged COCO file: {e}")
            return False

        # --- カテゴリ存在チェック (必須) ---
        if "categories" not in self.master_coco_data:
            print("Error: 'categories' key not found in the input COCO JSON.")
            return False

        existing_categories = {
            cat["id"]: cat for cat in self.master_coco_data["categories"]
        }
        required_categories = {
            CAT_ID_BALL: {"name": "ball", "supercategory": "sports"},
            CAT_ID_PLAYER: {"name": "player", "supercategory": "person"},
            CAT_ID_NON_PLAYER: {"name": "non_player_person", "supercategory": "person"},
        }

        missing_category_names = []
        for req_id, req_info in required_categories.items():
            if req_id not in existing_categories:
                missing_category_names.append(f"{req_info['name']} (ID {req_id})")

        if missing_category_names:
            print(
                f"Error: Input COCO JSON must contain the following categories: {', '.join(missing_category_names)}"
            )
            return False

        print("Required categories found in input COCO.")

        # --- ★ 追加: annotation_id から image_id を引けるマップを構築 ---
        annotation_id_to_image_id: Dict[int, int] = {}
        if self.master_coco_data and "annotations" in self.master_coco_data:
            for ann in self.master_coco_data["annotations"]:
                ann_id = ann.get("id")
                img_id = ann.get("image_id")
                if ann_id is not None and img_id is not None:
                    annotation_id_to_image_id[ann_id] = img_id
        print(
            f"Built annotation_id_to_image_id map for {len(annotation_id_to_image_id)} annotations."
        )

        # --- 分類結果ファイル群 ロードと集計 ---
        print(f"Loading classification results from: {self.spec_dir}")
        if not self.spec_dir.is_dir():
            print(
                f"Warning: Specification directory not found at {self.spec_dir}. No player/ignore/verification classifications will be applied."
            )
            return True  # COCOファイルは読めたので処理は続行できる

        spec_files = list(self.spec_dir.glob("game*_clip*.json"))
        print(f"Found {len(spec_files)} specification files.")

        for spec_file in spec_files:
            try:
                with open(spec_file, "r") as f:
                    spec_data = json.load(f)

                # player, ignored の ID をセットに追加
                player_ids_in_file = spec_data.get("players", [])
                ignored_ids_in_file = spec_data.get("ignored", [])
                self.all_player_ids.update(set(player_ids_in_file))
                self.all_ignored_ids.update(set(ignored_ids_in_file))

                # ★ 修正: verified_image_ids の集計方法を変更
                # ファイルの verified_image_ids は使わず、ファイル内の Player ID から画像IDを特定
                # spec_data.get("verified_image_ids", []) は無視される

                # ファイル内のPlayer IDに対応する画像IDをverified_image_idsに追加
                for ann_id in player_ids_in_file:
                    img_id = annotation_id_to_image_id.get(ann_id)
                    if img_id is not None:
                        self.verified_image_ids.add(img_id)
                    else:
                        # Player ID がマスターアノテーションに見つからない場合
                        print(
                            f"Warning: Player annotation ID {ann_id} found in {spec_file} but not in master COCO annotations. Cannot determine image_id for verification."
                        )

            except Exception as e:
                print(
                    f"Warning: Could not load or parse specification file {spec_file}: {e}. Skipping."
                )

        # Ignoreされたアノテーションの画像も検証済みとしたい場合は、
        # ここで self.all_ignored_ids に対応する画像IDも self.verified_image_ids に追加する。
        # 今回の要件「プレーヤーと判断された人が1人でもいるフレーム」に合わせて Player ID のみから集計する。

        print(
            f"Loaded total {len(self.all_player_ids)} player IDs and {len(self.all_ignored_ids)} ignored IDs."
        )
        print(
            f"Determined {len(self.verified_image_ids)} image IDs as human verified (based on Player annotations)."
        )
        return True

    def generate_final_annotations(self) -> List[Dict]:
        """
        元のannotationsリストをフィルタリングし、カテゴリIDを修正、メタデータを追加して最終リストを生成
        """
        if not self.master_coco_data or "annotations" not in self.master_coco_data:
            print("Error: Master COCO data not loaded or has no annotations.")
            return []

        original_annotations = self.master_coco_data["annotations"]
        final_annotations: List[Dict] = []
        discarded_count = 0
        category_changed_to_player_count = 0  # 変数名変更
        kept_as_nonplayer_count = 0
        annotations_marked_verified_count = 0  # 変数名変更

        print(
            f"\nGenerating final annotations from {len(original_annotations)} candidates..."
        )

        for ann in original_annotations:
            ann_id = ann.get("id")
            cat_id = ann.get("category_id")
            img_id = ann.get("image_id")

            # 必須フィールドのチェック
            if ann_id is None or cat_id is None or img_id is None:
                print(
                    f"Warning: Skipping annotation with missing id, category_id, or image_id: {ann}. "
                )
                discarded_count += 1  # これも破棄とカウント
                continue

            # ボールアノテーションは常に維持 (メタデータは追加しない)
            if cat_id == CAT_ID_BALL:
                final_annotations.append(copy.deepcopy(ann))  # コピーしておく
                continue

            # 人物候補 (初期カテゴリID=3 または元々2だったもの) のアノテーションを処理
            # 元データにカテゴリID 2 が存在する可能性も考慮し、処理対象とする
            if cat_id == CAT_ID_NON_PLAYER or cat_id == CAT_ID_PLAYER:
                # 無視対象かチェック
                if ann_id in self.all_ignored_ids:
                    discarded_count += 1
                    continue  # 破棄

                # 無視対象でない場合
                modified_ann = copy.deepcopy(ann)  # コピーして修正/メタデータを追加

                # ★ 修正: 人間による検証済みフラグを設定
                # このアノテーションが属する画像のIDが verified_image_ids セットに含まれているかチェック
                is_verified = img_id in self.verified_image_ids
                modified_ann[META_HUMAN_VERIFIED] = is_verified
                if is_verified:
                    annotations_marked_verified_count += 1

                # プレーヤーかチェック (無視されなかったものの中から)
                if ann_id in self.all_player_ids:
                    # Player としてマークされた -> category_id を 2 に変更
                    modified_ann["category_id"] = CAT_ID_PLAYER
                    final_annotations.append(modified_ann)
                    category_changed_to_player_count += 1
                else:
                    # プレーヤーとしても無視としてもマークされなかった -> NonPlayer (カテゴリID 3) のまま維持
                    # 元々カテゴリID 3 の場合と、元々カテゴリID 2 だったが Player 指定されなかった場合
                    # ここでは分類結果に従い、Player指定されていなければ全てカテゴリID 3 とする
                    modified_ann["category_id"] = CAT_ID_NON_PLAYER  # 明示的に3に設定
                    final_annotations.append(modified_ann)
                    kept_as_nonplayer_count += 1

            # その他のカテゴリは破棄
            else:
                print(
                    f"Warning: Skipping annotation with unexpected category ID {cat_id}: {ann_id}. "
                )
                discarded_count += 1

        print("\nFinished generating final annotations.")
        print(f"  Original total annotations: {len(original_annotations)}")
        print(f"  Final total annotations: {len(final_annotations)}")
        print(
            f"  Category changed to Player ({CAT_ID_PLAYER}): {category_changed_to_player_count}"
        )
        print(f"  Kept as NonPlayer ({CAT_ID_NON_PLAYER}): {kept_as_nonplayer_count}")
        print(f"  Discarded (Ignored or other): {discarded_count}")
        print(
            f"  Total annotations marked as human verified: {annotations_marked_verified_count} (belonging to {len(self.verified_image_ids)} images)"
        )  # ログ表示を修正

        return final_annotations

    def save_final_coco(self, final_annotations: List[Dict]):
        """最終アノテーションリストを含むCOCO JSONを保存"""
        if not self.master_coco_data:
            print("Error: Master COCO data not loaded. Cannot save.")
            return
        final_coco_data = copy.deepcopy(self.master_coco_data)
        final_coco_data["annotations"] = final_annotations
        # info セクションに最終更新日時を追加または更新
        if "info" not in final_coco_data:
            final_coco_data["info"] = {}
        final_coco_data["info"]["date_modified"] = datetime.now().isoformat()

        print(f"\nSaving final COCO data to: {self.output_final_coco_path}")
        try:
            self.output_final_coco_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_final_coco_path, "w", encoding="utf-8") as f:
                json.dump(final_coco_data, f, indent=4)
            print("Successfully saved the final COCO JSON file.")
        except Exception as e:
            print(f"Error saving final COCO file: {e}")

    def run(self):
        """プロセス全体を実行"""
        if not self.load_data():
            return

        final_annotations = self.generate_final_annotations()

        # generate_final_annotations がアノテーションリストを返した場合のみ保存
        if final_annotations is not None:
            self.save_final_coco(final_annotations)
        else:
            print("Final annotations list is empty or None. Skipping save.")


# --- 実行部分 ---
if __name__ == "__main__":
    # datetime は save_final_coco でインポート済み

    print("--- Final Dataset Generation Script ---")
    print(f"Input Merged COCO: {INPUT_MERGED_COCO}")
    print(f"Specification Dir: {SPECIFICATION_DIR}")
    print(f"Output Final COCO: {OUTPUT_FINAL_COCO}")
    print("-" * 30)

    generator = FinalDatasetGenerator(
        input_merged_coco=INPUT_MERGED_COCO,
        spec_dir=SPECIFICATION_DIR,
        output_final_coco=OUTPUT_FINAL_COCO,
    )
    generator.run()

    print("\n--- Script Finished ---")
