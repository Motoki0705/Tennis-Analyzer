# specify_players_clickable.py

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

# --- 定数 ---
INPUT_COCO_JSON = r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_with_all_persons_as_nonplayers.json"
IMAGE_ROOT_DIR = r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\images"
SPECIFICATION_DIR = (
    r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\player_specification"
)
PROGRESS_FILE = r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\player_specification\player_spec_last_worked.json"  # ファイル名変更推奨

# カテゴリID定義 (入力JSONと一致させる)
CAT_ID_BALL = 1
CAT_ID_PLAYER = 2
CAT_ID_NON_PLAYER = 3

# 状態定義 (Ignoredを削除)
STATE_PLAYER: Literal["player"] = "player"
STATE_NON_PLAYER: Literal["non_player"] = "non_player"  # デフォルト状態

# 色定義 (BGR) - Ignored関連を削除
COLOR_PLAYER = (0, 255, 0)  # 緑
COLOR_NON_PLAYER = (0, 0, 255)  # 赤 (デフォルト色)
COLOR_BALL = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG_PLAYER = (0, 128, 0)
COLOR_BG_NON_PLAYER = (0, 0, 128)
COLOR_INFO = (255, 200, 0)
COLOR_UNSAVED = (0, 255, 255)

# キー定義 (p, n を削除)
KEY_D = ord("d")  # 次フレーム
KEY_F = ord("f")  # >>
KEY_A = ord("a")  # <<
KEY_W = ord("w")  # クリップ保存してメニューへ戻る
KEY_Q = ord("q")  # ツール全体を終了


class PlayerSpecifierClickable:  # クラス名変更推奨
    """
    COCOアノテーション内の
    人物候補(初期カテゴリ3)を対話的に分類するツール。
    マウスクリックで Player/NonPlayer をトグルする。
    クリップ選択機能付き。
    """

    def __init__(
        self, coco_path: str, image_root: str, spec_dir: str, progress_file: str
    ):
        self.coco_path = Path(coco_path)
        self.image_root = Path(image_root)
        self.spec_dir = Path(spec_dir)
        self.progress_file = Path(progress_file)
        self.spec_dir.mkdir(parents=True, exist_ok=True)

        self.master_coco_data: Optional[Dict] = None
        self.image_id_to_image_info: Dict[int, Dict] = {}
        self.image_id_to_annotations: Dict[int, List[Dict]] = defaultdict(list)
        self.clip_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.sorted_clips: List[Tuple[int, int]] = []

        self.current_clip_id: Optional[Tuple[int, int]] = None
        self.current_clip_image_ids: List[int] = []
        self.current_clip_specifications: Dict[int, Literal["player", "non_player"]] = (
            {}
        )
        self.current_frame_index: int = 0
        self.clip_modified: bool = False
        self.last_worked_clip_id: Optional[Tuple[int, int]] = None

    # --- 修正関数: フレーム番号取得 ---
    def _get_frame_num_for_sort(self, img_id: int) -> float:
        """画像IDからファイル名を抽出し、数値フレーム番号を返す (ソート用)"""
        img_info = self.image_id_to_image_info.get(img_id)
        if not img_info:
            return float("inf")  # 画像情報が見つからない場合は最後に

        fname = img_info.get("file_name")
        if not fname:
            return float("inf")  # ファイル名がない場合も最後に

        try:
            # 例: '0000.jpg' -> 0, '0010.jpg' -> 10
            return int(Path(fname).stem)
        except ValueError:
            # ファイル名から数字部分を抽出できない場合 (例: 'frame_abc.png')
            print(
                f"Warning: Could not extract numeric frame number from filename '{fname}'. Sorting order may be affected."
            )
            return float("inf")  # 数字以外のファイル名も最後に

    def load_coco_data(self) -> bool:
        """メインのCOCO JSONファイルをロードし、内部マッピングを構築"""
        if not self.coco_path.is_file():
            print(f"Error: Input COCO file not found at {self.coco_path}")
            return False
        print(f"Loading COCO data from: {self.coco_path}")
        try:
            with open(self.coco_path, "r", encoding="utf-8") as f:
                self.master_coco_data = json.load(f)

            print("Building internal mappings...")
            for img_info in self.master_coco_data["images"]:
                self.image_id_to_image_info[img_info["id"]] = img_info

            for ann_info in self.master_coco_data["annotations"]:
                self.image_id_to_annotations[ann_info["image_id"]].append(ann_info)

            temp_clip_map = defaultdict(list)
            for img_id, img_info in self.image_id_to_image_info.items():
                if "game_id" in img_info and "clip_id" in img_info:
                    temp_clip_map[(img_info["game_id"], img_info["clip_id"])].append(
                        img_id
                    )
                else:
                    print(
                        f"Warning: Image ID {img_id} ('{img_info.get('file_name')}') missing game_id or clip_id."
                    )

            for gc_id, img_ids in temp_clip_map.items():
                # ★ 修正: 上で定義した通常の関数を key に指定
                self.clip_map[gc_id] = sorted(img_ids, key=self._get_frame_num_for_sort)

            self.sorted_clips = sorted(self.clip_map.keys())

            category_ids = {
                cat["id"] for cat in self.master_coco_data.get("categories", [])
            }
            if not all(
                cat_id in category_ids
                for cat_id in [CAT_ID_BALL, CAT_ID_PLAYER, CAT_ID_NON_PLAYER]
            ):
                print(
                    f"Error: Input COCO JSON must contain categories for Ball ({CAT_ID_BALL}), Player ({CAT_ID_PLAYER}), and NonPlayer ({CAT_ID_NON_PLAYER})."
                )
                return False

            print(f"Mappings built. Found {len(self.sorted_clips)} clips.")
            return True

        except Exception as e:
            print(f"Error loading or parsing COCO JSON: {e}")
            return False

    # --- load_progress, save_progress, get_spec_file_path, load_clip_specification, save_clip_specification は変更なし ---
    def load_progress(self) -> Optional[Tuple[int, int]]:
        try:
            if self.progress_file.is_file():
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    last_clip_id_list = data.get("last_worked_clip_id", [])
                if isinstance(last_clip_id_list, list) and len(last_clip_id_list) == 2:
                    self.last_worked_clip_id = tuple(last_clip_id_list)
                    print(
                        f"Progress file found. Last worked clip ID: {self.last_worked_clip_id}"
                    )
                    return self.last_worked_clip_id
            print("Progress file not found or invalid.")
            return None
        except Exception as e:
            print(f"Error reading progress file '{self.progress_file}': {e}.")
            return None

    def save_progress(self, clip_id: Tuple[int, int]):
        try:
            data = {
                "last_worked_clip_id": list(clip_id),
                "last_saved_time": datetime.now().isoformat(),
            }
            with open(self.progress_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Progress updated. Last worked clip ID: {clip_id}")
            self.last_worked_clip_id = clip_id
        except Exception as e:
            print(f"Error saving progress to '{self.progress_file}': {e}")

    def get_spec_file_path(self, clip_id: Tuple[int, int]) -> Path:
        game_id, clip_num = clip_id
        return self.spec_dir / f"game{game_id}_clip{clip_num}.json"

    def load_clip_specification(self):
        spec_file = self.get_spec_file_path(self.current_clip_id)
        self.current_clip_specifications = {}
        loaded_spec = {"players": []}

        if spec_file.is_file():
            try:
                with open(spec_file, "r") as f:
                    loaded_spec = json.load(f)
                print(f"Loaded existing specification from {spec_file}")
            except Exception as e:
                print(f"Warning: Could not load spec file {spec_file}: {e}")
                loaded_spec = {"players": []}

        player_ids = set(loaded_spec.get("players", []))

        for img_id in self.current_clip_image_ids:
            for ann in self.image_id_to_annotations.get(img_id, []):
                ann_id = ann["id"]
                cat_id = ann["category_id"]
                if cat_id == CAT_ID_NON_PLAYER or cat_id == CAT_ID_PLAYER:
                    if ann_id in player_ids:
                        self.current_clip_specifications[ann_id] = STATE_PLAYER
                    else:
                        self.current_clip_specifications[ann_id] = STATE_NON_PLAYER

    def save_clip_specification(self) -> bool:
        """現在のクリップの分類仕様をファイルに保存 (Player ID と verified_image_ids を修正)"""
        spec_file = self.get_spec_file_path(self.current_clip_id)

        # ★ 修正: verified_image_ids の生成ロジックを変更
        verified_image_ids_to_save = []
        for img_id in self.current_clip_image_ids:
            # この画像にPlayerとして分類されたアノテーションがあるかチェック
            annotations_in_image = self.image_id_to_annotations.get(img_id, [])
            is_player_frame = False
            for ann in annotations_in_image:
                ann_id = ann["id"]
                # 人物アノテーションであり、かつ作業用データでPlayerになっているか
                if (
                    ann.get("category_id") == CAT_ID_NON_PLAYER
                    or ann.get("category_id") == CAT_ID_PLAYER
                ) and self.current_clip_specifications.get(ann_id) == STATE_PLAYER:
                    is_player_frame = True
                    break  # Playerが見つかったらこの画像はおしまい

            if is_player_frame:
                verified_image_ids_to_save.append(img_id)

        output_spec = {
            "players": [
                ann_id
                for ann_id, state in self.current_clip_specifications.items()
                if state == STATE_PLAYER
            ],
            "verified_image_ids": verified_image_ids_to_save,  # ★ 新しいリストを設定
        }

        print(
            f"\nSaving specification for clip {self.current_clip_id} to {spec_file}..."
        )
        # ... (以降のファイル保存処理は同じ) ...
        try:
            with open(spec_file, "w") as f:
                json.dump(output_spec, f, indent=4)
            print("Specification saved successfully.")
            self.clip_modified = False
            return True
        except Exception as e:
            print(f"Error saving specification file: {e}")
            return False

    # --- _draw_annotations は変更なし ---
    def _draw_annotations(self, img: np.ndarray, image_id: int):
        annotations = self.image_id_to_annotations.get(image_id, [])
        for ann in annotations:
            ann_id = ann["id"]
            cat_id = ann["category_id"]
            bbox = ann.get("bbox")
            if not bbox:
                continue
            xmin, ymin, w, h = map(int, bbox)
            xmax, ymax = xmin + w, ymin + h
            current_state = self.current_clip_specifications.get(ann_id)
            color, bg_color, label_text = None, None, None
            if cat_id == CAT_ID_BALL:
                color, label_text = COLOR_BALL, "Ball"
            elif cat_id == CAT_ID_PLAYER or cat_id == CAT_ID_NON_PLAYER:
                if current_state == STATE_PLAYER:
                    color, bg_color, label_text = (
                        COLOR_PLAYER,
                        COLOR_BG_PLAYER,
                        f"Player ({ann_id})",
                    )
                else:
                    color, bg_color, label_text = (
                        COLOR_NON_PLAYER,
                        COLOR_BG_NON_PLAYER,
                        f"NonPlayer ({ann_id})",
                    )
            if color:
                border_color = color  # No selected highlight
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), border_color, 2)
                if label_text:
                    (tw, th), bl = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    tx, ty = xmin, (
                        ymin - bl - 3 if ymin - bl - 3 > 0 else ymin + th + bl
                    )
                    final_bg_color = bg_color if bg_color else border_color
                    cv2.rectangle(
                        img, (tx, ty - th - bl), (tx + tw, ty + bl), final_bg_color, -1
                    )
                    cv2.putText(
                        img,
                        label_text,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLOR_TEXT,
                        1,
                        cv2.LINE_AA,
                    )

    # --- _handle_mouse_click は変更なし ---
    def _handle_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_image_id = self.current_clip_image_ids[self.current_frame_index]
            annotations = self.image_id_to_annotations.get(current_image_id, [])
            clicked_ann_id, min_area = None, float("inf")
            for ann in reversed(annotations):
                if (
                    ann["category_id"] == CAT_ID_NON_PLAYER
                    or ann["category_id"] == CAT_ID_PLAYER
                ):
                    bbox = ann.get("bbox")
                    if bbox:
                        xmin, ymin, w, h = map(int, bbox)
                        if xmin <= x < xmin + w and ymin <= y < ymin + h:
                            area = w * h
                            if area < min_area:
                                clicked_ann_id, min_area = ann["id"], area
            if clicked_ann_id is not None:
                current_state = self.current_clip_specifications.get(
                    clicked_ann_id, STATE_NON_PLAYER
                )
                new_state = (
                    STATE_NON_PLAYER if current_state == STATE_PLAYER else STATE_PLAYER
                )
                self.current_clip_specifications[clicked_ann_id] = new_state
                self.clip_modified = True
                print(f"Clicked ID: {clicked_ann_id}. Toggled state to: {new_state}")

    # --- _process_key_input は変更なし ---
    def _process_key_input(self, key: int) -> Optional[Literal["saved", "quit"]]:
        num_frames = len(self.current_clip_image_ids)
        action_taken = False
        if key == KEY_D:
            if self.current_frame_index < num_frames - 1:
                self.current_frame_index += 1
                action_taken = True
        elif key == KEY_F:
            if self.current_frame_index < num_frames - 1:
                self.current_frame_index = min(
                    self.current_frame_index + 30, num_frames - 1
                )
                action_taken = True
        elif key == KEY_A:
            if self.current_frame_index > 0:
                self.current_frame_index = max(self.current_frame_index - 5, 0)
                action_taken = True
        elif key == KEY_W:
            if self.save_clip_specification():
                self.save_progress(self.current_clip_id)
                return "saved"
            else:
                print("Save failed.")
                action_taken = True
        # --- 終了 ---
        elif key == KEY_Q:
            # 退出メッセージを組み立て
            quit_msg = "Quit requested."
            # ★ 修正: if文を次の行に分ける
            if self.clip_modified:
                quit_msg += " Unsaved changes will be lost."

            # ユーザーに確認
            print(f"\n{quit_msg}")
            confirm = input("Are you sure? (y/N): ").lower()

            if confirm == "y":
                # ツール全体を終了
                return "quit"
            else:
                # 終了取り消し
                print("Quit cancelled.")
                action_taken = (
                    True  # キャンセル後も画面を再描画するため必要に応じてフラグを立てる
                )
        return "redraw" if action_taken else None

    # --- specify_clip は変更なし ---
    def specify_clip(self, clip_id: Tuple[int, int]):
        self.current_clip_id = clip_id
        self.current_clip_image_ids = self.clip_map[clip_id]
        self.current_frame_index = 0
        self.clip_modified = False
        self.load_clip_specification()
        game_id, clip_num = clip_id
        num_frames = len(self.current_clip_image_ids)
        window_title = f"Specify Players (Click Toggle) - G{game_id} C{clip_num}"
        print(
            f"\n--- Specifying Clip: Game {game_id}, Clip {clip_num} ({num_frames} frames) ---"
        )
        print("Click on box to toggle Player/NonPlayer.")
        print("Keys: [d]Next [f]>> [a]<< | [w]Save & Back [q]Quit Tool")
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, self._handle_mouse_click)
        while True:
            # --- 画面描画 ---
            current_image_id = self.current_clip_image_ids[self.current_frame_index]
            image_info = self.image_id_to_image_info[current_image_id]
            rel_path = image_info.get("original_path", image_info.get("file_name"))
            img_path = self.image_root / rel_path
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                img_cv = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    img_cv,
                    f"Error loading {rel_path}",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                img_cv = img_cv.copy()
            self._draw_annotations(img_cv, current_image_id)
            h, w = img_cv.shape[:2]
            info1 = f"G{game_id} C{clip_num} | Frame {self.current_frame_index + 1}/{num_frames} ({image_info['file_name']})"
            cv2.putText(
                img_cv,
                info1,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_INFO,
                1,
                cv2.LINE_AA,
            )
            if self.clip_modified:
                cv2.putText(
                    img_cv,
                    "* Unsaved *",
                    (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_UNSAVED,
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_title, img_cv)
            # --- キー入力処理 ---
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            result = self._process_key_input(key)
            if result == "saved":
                cv2.destroyWindow(window_title)
                return "back_to_menu"
            elif result == "quit":
                cv2.destroyWindow(window_title)
                return "quit"
            # Redraw logic is implicitly handled by state changes and navigation keys triggering a new loop iteration

    # --- run は変更なし ---
    def run(self):
        if not self.load_coco_data():
            return
        self.load_progress()
        while True:
            print("\n--- Clip Selection Menu ---")
            print("Available Clips ([X] = Spec file exists, [ ] = No spec file):")
            unfinished_clips = []
            spec_exists_map = {}
            for idx, clip_id_tuple in enumerate(self.sorted_clips):
                spec_file = self.get_spec_file_path(clip_id_tuple)
                exists = spec_file.exists()
                spec_exists_map[clip_id_tuple] = exists
                status = "[X]" if exists else "[ ]"
                prefix = "->" if clip_id_tuple == self.last_worked_clip_id else "  "
                print(
                    f"{prefix}{status} {idx + 1}: Game {clip_id_tuple[0]}, Clip {clip_id_tuple[1]}"
                )
                if not exists:
                    unfinished_clips.append((idx, clip_id_tuple))
            print(
                "\nEnter clip number, 'n' for next unfinished, 'f <game_id>', or 'q' to quit."
            )
            user_input = input("> ").lower().strip()
            parts = user_input.split()
            command = parts[0]
            selected_clip_index = -1
            if command == "q":
                print("Exiting.")
                break
            elif command == "n":
                if unfinished_clips:
                    next_unfinished_idx = -1
                    last_worked_index = -1
                    if (
                        self.last_worked_clip_id
                        and self.last_worked_clip_id in self.sorted_clips
                    ):
                        last_worked_index = self.sorted_clips.index(
                            self.last_worked_clip_id
                        )
                    for idx, clip_id in unfinished_clips:
                        if idx > last_worked_index:
                            next_unfinished_idx = idx
                            break
                    if next_unfinished_idx == -1:
                        next_unfinished_idx = unfinished_clips[0][0]
                    selected_clip_index = next_unfinished_idx
                    print(f"Selected next unfinished: {selected_clip_index + 1}")
                else:
                    print("No unfinished clips.")
                    continue
            elif command == "f":
                if len(parts) == 2 and parts[1].isdigit():
                    filter_game_id = int(parts[1])
                    print(f"\n--- Clips for Game {filter_game_id} ---")
                    game_clips = []
                    for idx, clip_id_tuple in enumerate(self.sorted_clips):
                        if clip_id_tuple[0] == filter_game_id:
                            status = "[X]" if spec_exists_map[clip_id_tuple] else "[ ]"
                            prefix = (
                                "->"
                                if clip_id_tuple == self.last_worked_clip_id
                                else "  "
                            )
                            print(
                                f"{prefix}{status} {idx + 1}: G{clip_id_tuple[0]} C{clip_id_tuple[1]}"
                            )
                            game_clips.append((idx, clip_id_tuple))
                    if not game_clips:
                        print(f"No clips for Game {filter_game_id}.")
                        continue
                    print(f"\nEnter number for Game {filter_game_id}:")
                    filter_input = input("> ").lower().strip()
                    try:
                        selected_clip_index = int(filter_input) - 1
                        if (
                            not (0 <= selected_clip_index < len(self.sorted_clips))
                            or self.sorted_clips[selected_clip_index][0]
                            != filter_game_id
                        ):
                            print("Invalid number for this game.")
                            continue
                    except ValueError:
                        print("Invalid input.")
                        continue
                else:
                    print("Invalid filter command. Use 'f <game_id>'.")
                    continue
            else:
                try:
                    selected_clip_index = int(command) - 1
                    if not (0 <= selected_clip_index < len(self.sorted_clips)):
                        print("Invalid clip number.")
                        continue
                except ValueError:
                    print("Invalid input.")
                    continue
            target_clip_id = self.sorted_clips[selected_clip_index]
            outcome = self.specify_clip(target_clip_id)
            if outcome == "quit":
                break
            elif outcome == "back_to_menu":
                continue
        print("\nSpecification Tool finished.")
        cv2.destroyAllWindows()


# --- 実行部分 ---
if __name__ == "__main__":
    print("--- Player Specification Tool (Click Toggle) ---")
    print(f"Input COCO JSON: {INPUT_COCO_JSON}")
    print(f"Image Root: {IMAGE_ROOT_DIR}")
    print(f"Specification Dir: {SPECIFICATION_DIR}")
    print("-" * 30)

    specifier = PlayerSpecifierClickable(
        coco_path=INPUT_COCO_JSON,
        image_root=IMAGE_ROOT_DIR,
        spec_dir=SPECIFICATION_DIR,
        progress_file=PROGRESS_FILE,
    )
    specifier.run()

    print("\n--- Script Finished ---")
