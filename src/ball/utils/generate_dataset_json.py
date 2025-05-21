import csv
import json
import os
import random


def generate_coco_splits(
    root_dir, out_dir, num_frames=3, train_ratio=0.7, val_ratio=0.15
):
    os.makedirs(out_dir, exist_ok=True)

    # COCO-like data containers
    all_samples = []

    image_id = 0
    ann_id = 0

    print(f"🔄 Processing dataset from {root_dir} with T={num_frames}...")

    for game in sorted(os.listdir(root_dir)):
        for clip in sorted(os.listdir(os.path.join(root_dir, game))):
            clip_path = os.path.join(root_dir, game, clip)
            image_files = sorted(
                [f for f in os.listdir(clip_path) if f.endswith(".jpg")]
            )
            label_path = os.path.join(clip_path, "Label.csv")

            if not os.path.exists(label_path) or len(image_files) < num_frames:
                continue

            # ラベル読み込み
            with open(label_path, newline="") as f:
                reader = csv.reader(f)
                next(reader)
                labels = {row[0]: row[1:] for row in reader}

            for i in range(num_frames - 1, len(image_files)):
                frames = image_files[i - num_frames + 1 : i + 1]
                target_frame = frames[-1]
                if target_frame not in labels:
                    continue
                if int(labels[target_frame][0]) == 0:
                    continue

                try:
                    x = float(labels[target_frame][1])
                    y = float(labels[target_frame][2])
                    visibility = int(labels[target_frame][0])
                    status = int(labels[target_frame][3])
                except:
                    continue

                sample = {
                    "image_id": image_id,
                    "video_id": f"{game}_{clip}",
                    "video_path": f"{root_dir}/{game}/{clip}",
                    "frames": frames,
                    "label": {
                        "x": x,
                        "y": y,
                        "visibility": visibility,
                        "status": status,
                    },
                }
                all_samples.append(sample)
                image_id += 1

    # シャッフルして分割
    random.shuffle(all_samples)
    N = len(all_samples)
    N_train = int(N * train_ratio)
    N_val = int(N * val_ratio)

    splits = {
        "train": all_samples[:N_train],
        "val": all_samples[N_train : N_train + N_val],
        "test": all_samples[N_train + N_val :],
    }

    print(
        f"📊 Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    for split_name, split_samples in splits.items():
        coco = {"images": [], "annotations": [], "videos": []}

        added_video_ids = set()

        for sample in split_samples:
            coco["images"].append(
                {
                    "id": sample["image_id"],
                    "file_names": sample["frames"],
                    "video_id": sample["video_id"],
                }
            )

            coco["annotations"].append(
                {
                    "image_id": sample["image_id"],
                    "keypoints": [
                        sample["label"]["x"],
                        sample["label"]["y"],
                        sample["label"]["visibility"],
                    ],
                    "status": sample["label"]["status"],
                }
            )

            if sample["video_id"] not in added_video_ids:
                coco["videos"].append(
                    {"id": sample["video_id"], "path": sample["video_path"]}
                )
                added_video_ids.add(sample["video_id"])

        out_path = os.path.join(out_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"✅ Saved {split_name} set to {out_path}")


if __name__ == "__main__":
    generate_coco_splits(
        root_dir="data/images",
        out_dir="data/annotation_jsons",
        num_frames=3,
        train_ratio=0.7,
        val_ratio=0.15,
    )
