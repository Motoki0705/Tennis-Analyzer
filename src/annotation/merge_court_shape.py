
#!/usr/bin/env python
"""Merge court_shapes into an existing COCO JSON.

Usage:
    python merge_court_shapes.py \
        --coco_json tennis_dataset.json \
        --court_shapes court_shapes.json \
        --output_json tennis_dataset_merged.json

If the original COCO JSON already contains a `court_shapes` key, new
entries are appended (game_id collision will overwrite the old entry).
"""

import argparse
import json
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Merge court_shapes into COCO JSON")
    p.add_argument("--coco_json", required=False, default=r"data\ball\coco_annotations_pose.json", help="Path to original COCO JSON")
    p.add_argument("--court_shapes", required=False, default=r"data\ball\coco_annotations_all.json", help="Path to court_shapes JSON")
    p.add_argument("--output_json", required=False, default=r"data\ball\coco_annotations_all_merged.json", help="Path to write merged JSON")
    return p.parse_args()

def main():
    args = parse_args()

    coco_path = Path(args.coco_json)
    court_path = Path(args.court_shapes)
    out_path = Path(args.output_json)

    if not coco_path.exists():
        sys.exit(f"COCO JSON not found: {coco_path}")
    if not court_path.exists():
        sys.exit(f"court_shapes JSON not found: {court_path}")

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    with court_path.open("r", encoding="utf-8") as f:
        court_shapes = json.load(f)

    # ensure court_shapes is a list
    if not isinstance(court_shapes, list):
        sys.exit("court_shapes JSON must be a list of objects")

    # attach / merge
    if 'court_shapes' not in coco:
        coco['court_shapes'] = []

    # create dict for fast overwrite if same game_id
    existing = {c['game_id']: c for c in coco['court_shapes']}

    for item in court_shapes:
        gid = item['game_id']
        existing[gid] = item  # overwrite or insert

    coco['court_shapes'] = list(existing.values())

    # save
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"Merged JSON saved to {out_path}")

if __name__ == "__main__":
    main()
