import json
from collections import Counter

# アノテーションファイルのパスを指定してください
annotation_path = "data/court/converted_train.json"

with open(annotation_path, "r") as f:
    data = json.load(f)

# 各サンプルのnum_keypointsを集計
num_kp_list = [sample.get("num_keypoints", None) for sample in data]

# Noneを除外
num_kp_list = [n for n in num_kp_list if n is not None]

# 集計
stats = Counter(num_kp_list)

print("=== num_keypoints ごとのサンプル数 ===")
for n in sorted(stats):
    print(f"{n}個: {stats[n]}件")
