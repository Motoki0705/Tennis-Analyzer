#!/usr/bin/env python
"""Annotation Viewer: play dataset frames like a video with overlays.

Controls
--------
SPACE           : pause/resume
RIGHT or 'd'    : next frame (when paused)
LEFT  or 'a'    : previous frame (when paused)
c               : skip to next clip
g               : skip to next game
q / ESC         : quit

Required fields
---------------
images[].game_id, images[].clip_id
annotations[].event_status (for ball)
court_shapes[].game_id, court_shapes[].keypoints

Usage
-----
python annotation_viewer.py \
    --coco_json tennis_dataset_merged.json \
    --image_root /path/to/images \
    --fps 30
"""

import argparse
import json
import cv2
from pathlib import Path
from collections import defaultdict

BALL_ID = 1
PLAYER_ID = 2
COURT_ID = 3   # optional

# colors (B, G, R)
COLORS = {
    "ball_default": (0, 215, 255),   # orange
    "ball_event":   (0, 0, 255),     # red
    "bbox":         (0, 255, 0),     # green
    "pose":         (255, 255, 0),   # cyan
    "court":        (255, 0, 255),   # magenta
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco_json', default=r"data\ball\coco_annotations_ball_pose.json", required=False)
    p.add_argument('--image_root', default=r"data\ball\images", required=False)
    p.add_argument('--fps', type=float, default=30.0)
    return p.parse_args()

def safe_get(d, key, default=None):
    """安全に辞書のキーを取得する関数"""
    return d.get(key, default)

def safe_list(lst, idx, default=None):
    """安全にリストのインデックスを取得する関数"""
    return lst[idx] if idx < len(lst) else default

def build_index(coco):
    ann_by_image = defaultdict(list)
    for ann in coco['annotations']:
        ann_by_image[ann['image_id']].append(ann)
    return ann_by_image

def build_skeleton_map(coco):
    for cat in coco.get('categories', []):
        if cat.get('id') == PLAYER_ID:
            return cat.get('skeleton', [])
    return []

def build_court_map(coco):
    court_map = {}
    if 'court_shapes' in coco:
        for c in coco['court_shapes']:
            court_map[c['game_id']] = c['keypoints']
    return court_map

def main():
    args = parse_args()
    with open(args.coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    ann_by_image = build_index(coco)
    skeleton = build_skeleton_map(coco)
    court_map = build_court_map(coco)
    img_root = Path(args.image_root)

    images = sorted(coco['images'], key=lambda x: x['id'])
    total = len(images)
    idx = 0
    paused = False
    win = 'Annotation Viewer'
    cv2.namedWindow(win)

    delay = int(1000 / args.fps) if args.fps > 0 else 1

    while 0 <= idx < total:
        img_info = images[idx]
        img_path = img_root / img_info.get('original_path', img_info['file_name'])
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f'Cannot read {img_path}')
            idx += 1
            continue

        game_id = img_info.get('game_id')
        clip_id = img_info.get('clip_id')

        # overlay court keypoints
        kp_vec = court_map.get(game_id)
        if kp_vec and len(kp_vec) >= 3:
            pts = [(kp_vec[i], kp_vec[i+1]) for i in range(0, len(kp_vec), 3)]
            for x, y in pts:
                cv2.circle(frame, (int(x), int(y)), 3, COLORS['court'], -1)

        # overlay annotations
        for ann in ann_by_image.get(img_info['id'], []):
            if ann['category_id'] == BALL_ID:
                # キーポイント
                keypoints = ann.get('keypoints', [])
                if len(keypoints) >= 3:
                    x, y, v = keypoints[:3]
                    color = COLORS['ball_event'] if ann.get('event_status', 0) else COLORS['ball_default']
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)

                # BBox
                bbox = ann.get('bbox')
                if bbox and len(bbox) == 4:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), COLORS['bbox'], 2)

            elif ann['category_id'] == PLAYER_ID:
                # BBox
                bbox = ann.get('bbox')
                if bbox and len(bbox) == 4:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), COLORS['bbox'], 2)

                # Pose
                keypoints = ann.get('keypoints', [])
                if len(keypoints) >= 3:
                    kps = [(keypoints[i], keypoints[i + 1], keypoints[i + 2]) for i in range(0, len(keypoints), 3)]
                    # draw lines
                    for a, b in skeleton:
                        if a < len(kps) and b < len(kps):
                            xa, ya, va = kps[a]
                            xb, yb, vb = kps[b]
                            if va > 0 and vb > 0:
                                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), COLORS['pose'], 1)

                    # draw joints
                    for x, y, v in kps:
                        if v > 0:
                            cv2.circle(frame, (int(x), int(y)), 2, COLORS['pose'], -1)

        # text overlay
        txt = f'Game {game_id}  Clip {clip_id}  Frame {idx+1}/{total}'
        cv2.putText(frame, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(win, frame)

        key = cv2.waitKey(delay if not paused else 0) & 0xFF

        if key == ord(' ') :  # pause/resume
            paused = not paused
        elif key in (ord('d'), 83):  # right
            idx += 1
        elif key in (ord('a'), 81):  # left
            idx = max(0, idx - 1)
        elif key == ord('c'):  # skip clip
            current_clip = clip_id
            while idx < total and images[idx]['clip_id'] == current_clip and images[idx]['game_id'] == game_id:
                idx += 1
        elif key == ord('g'):  # skip game
            current_game = game_id
            while idx < total and images[idx]['game_id'] == current_game:
                idx += 1
        elif key in (ord('q'), 27):  # quit
            break
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
