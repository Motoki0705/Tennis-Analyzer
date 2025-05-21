import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


def visualize_clip(
    coco_json_path: str,
    images_root: str,
    game_id: int,
    clip_id: int,
    output_video_path: str,
    output_size: tuple = (1280, 720),
    fps: int = 5,
):
    # アノテーション読み込み
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # imagesとannotationsをindex化
    img_id_to_img = {
        img["id"]: img
        for img in coco["images"]
        if img["game_id"] == game_id and img["clip_id"] == clip_id
    }
    annotations_by_image_id = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["image_id"] in img_id_to_img and ann["category_id"] in [2, 3]:
            annotations_by_image_id[ann["image_id"]].append(ann)

    # クリップ内の画像をfile_nameでソート
    sorted_images = sorted(img_id_to_img.values(), key=lambda x: x["file_name"])

    # 動画ライター初期化
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    for img_info in tqdm(sorted_images, desc=f"Rendering clip {clip_id}"):
        img_path = os.path.join(images_root, img_info["original_path"])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Image not found: {img_path}")
            continue

        img = cv2.resize(img, output_size)
        anns = annotations_by_image_id[img_info["id"]]

        # 描画
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            color = (0, 255, 0) if ann["category_id"] == 2 else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label = "Player" if ann["category_id"] == 2 else "Non-Player"
            pid = ann.get("player_id", -1)
            text = f"{label} ID={pid}"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(img)

    writer.release()
    print(f"Video saved to {output_video_path}")


# -------- 使用例 --------
if __name__ == "__main__":
    visualize_clip(
        coco_json_path=r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_globally_tracked.json",
        images_root=r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\images",
        game_id=2,
        clip_id=1,
        output_video_path=r"C:\Users\kamim\code\Tennis-Analyzer\outputs\clip_1_visualized.mp4",
        output_size=(1280, 720),
        fps=15,
    )
