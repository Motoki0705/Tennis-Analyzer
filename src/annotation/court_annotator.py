
#!/usr/bin/env python
"""Manual Court Keypoint Annotator (OpenCV)

Usage:
    python manual_court_annotator.py \
        --coco_json dataset.json \
        --image_root path/to/images \
        --output_json court_shapes.json

Controls (while annotation window is active):
    - Left click : add keypoint
    - 'u'        : undo last keypoint
    - 'r'        : reset current keypoints
    - 'n' or 'ENTER': save current game's keypoints and move to next game
    - 'q' or 'ESC'  : quit (saves progress so far)

The script will iterate over each unique `game_id` in the COCO JSON,
display the first image of that game, and let you click any number of
keypoints in order. For each game, the clicked points are stored as
[x, y, 2] triplets (v=2 means visible) and written to
`court_shapes` in the output JSON.
"""

import argparse
import json
import os
from pathlib import Path
import cv2
import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Manual Court Keypoint Annotator")
    p.add_argument('--coco_json', required=False, default=r"data\ball\coco_annotations_pose.json", help='Path to existing COCO JSON')
    p.add_argument('--image_root', required=False, default=r"data\ball\images", help='Root directory where image files are stored')
    p.add_argument('--output_json', required=False, default=r"data\ball\coco_annotations_all.json", help='Path to output court_shapes JSON')
    return p.parse_args()

def load_game_representatives(coco_json):
    """Return dict {game_id: first_image_dict}."""
    games = {}
    for img in coco_json['images']:
        gid = img['game_id']
        if gid not in games:
            games[gid] = img
    return games

def main():
    args = parse_args()

    with open(args.coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    games = load_game_representatives(coco)
    court_shapes = []

    # Load existing if resuming
    if Path(args.output_json).exists():
        with open(args.output_json, 'r', encoding='utf-8') as f:
            court_shapes = json.load(f)
        done_game_ids = {c['game_id'] for c in court_shapes}
    else:
        done_game_ids = set()

    window = 'Annotate Court (click points, n:next, u:undo, r:reset, q:quit)'
    cv2.namedWindow(window)

    keypoints = []  # current game's clicks

    def on_mouse(event, x, y, flags, param):
        nonlocal keypoints
        if event == cv2.EVENT_LBUTTONDOWN:
            keypoints.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    game_ids = sorted(games.keys())
    idx = 0
    while idx < len(game_ids):
        gid = game_ids[idx]
        if gid in done_game_ids:
            idx += 1
            continue

        img_info = games[gid]
        img_path = Path(args.image_root) / img_info.get('original_path', img_info['file_name'])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image for game {gid}: {img_path}")
            idx += 1
            continue

        keypoints.clear()
        while True:
            disp = img.copy()
            # draw existing points
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(disp, str(i+1), (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(window, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('u'), ord('U')):
                if keypoints:
                    keypoints.pop()
            elif k in (ord('r'), ord('R')):
                keypoints.clear()
            elif k in (ord('n'), ord('N'), 13):  # Enter key
                if not keypoints:
                    print("No points clicked. Please annotate before saving.")
                    continue
                flat = []
                for x, y in keypoints:
                    flat.extend([int(x), int(y), 2])  # v=2
                court_shapes.append({
                    "game_id": gid,
                    "keypoints": flat
                })
                done_game_ids.add(gid)
                # write incremental save
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(court_shapes, f, indent=2)
                print(f"Saved game {gid} with {len(keypoints)} pts. Moving to next.")
                idx += 1
                break
            elif k in (ord('q'), ord('Q'), 27):  # ESC
                print("Quitting annotation.")
                cv2.destroyAllWindows()
                # Save before exit
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(court_shapes, f, indent=2)
                return

    cv2.destroyAllWindows()
    print("All games annotated. Output saved to", args.output_json)

if __name__ == '__main__':
    main()
