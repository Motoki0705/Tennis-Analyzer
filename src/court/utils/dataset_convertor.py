import json

from utils.utils import line_intersection

with open(r"TennisCourtDetector\data\data_val.json", "r") as f:
    raw_data = json.load(f)

converted_data = []

for entry in raw_data:
    kps = entry["kps"]
    keypoints = []
    for x, y in kps:
        keypoints.extend([x, y, 2])  # 2 = visible

    # 中心点の追加（交差点）
    x_ct, y_ct = line_intersection(
        (kps[0][0], kps[0][1], kps[3][0], kps[3][1]),
        (kps[1][0], kps[1][1], kps[2][0], kps[2][1]),
    )

    # float に変換
    keypoints.extend([float(x_ct), float(y_ct), 2])

    converted_data.append(
        {
            "image_id": entry["id"] + ".png",
            "file_name": entry["id"] + ".png",
            "image_path": f"./data/images/{entry['id']}.png",
            "width": 1280,
            "height": 720,
            "keypoints": keypoints,
            "num_keypoints": 15,
        }
    )

with open(r"TennisCourtDetector\data\converted_val.json", "w") as f:
    json.dump(converted_data, f, indent=2)
