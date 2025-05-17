#!/usr/bin/env python
"""Annotation Viewer  –  show Ball / Court / Pose overlays.

Controls
--------
SPACE           : pause / resume
RIGHT or 'd'    : next frame (when paused)
LEFT  or 'a'    : previous frame (when paused)
c               : skip to next clip
g               : skip to next game
q / ESC         : quit
"""

import json
import cv2
from pathlib import Path
from collections import defaultdict

# ─────────── ハイパーパラメータ (必要に応じて変更) ───────────
PARENT_DIR = Path("outputs/frames")   # 解析済みフォルダ親
FPS        = 30.0                     # 再生 FPS
MAX_WIDTH  = 640                     # 表示サイズ上限
MAX_HEIGHT = 360

# 閾値
BALL_THRESH  = 0.60   # ball score
COURT_THRESH = 0.60   # court keypoint score
POSE_THRESH  = 0.60   # pose keypoint score

# カテゴリ ID
BALL_ID   = 1
PLAYER_ID = 2
COURT_ID  = 3

# 色 (B, G, R)
COLORS = {
    "ball":    (  0, 215, 255),
    "ball_ev": (  0,   0, 255),
    "court":   (255,   0, 255),
    "bbox":    (  0, 255,   0),
    "pose":    (255, 255,   0),
}

# ──────────────────────────────────────────────────────────
def build_index(coco):
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[ann["image_id"]].append(ann)
    return by_img

def build_skeleton(coco):
    for cat in coco.get("categories", []):
        if cat["id"] == PLAYER_ID:
            return cat.get("skeleton", [])
    return []

def build_court_gt(coco):
    # court_shapes があれば GT として可視化
    m = {}
    for shp in coco.get("court_shapes", []):
        m[shp["game_id"]] = shp.get("keypoints", [])
    return m

def latest_output_dir(parent: Path) -> Path:
    subdirs = sorted([d for d in parent.iterdir() if d.is_dir()])
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {parent}")
    return subdirs[-1]

# ──────────────────────────────────────────────────────────
def main():
    out_dir = latest_output_dir(PARENT_DIR)
    img_root = out_dir
    coco_json = out_dir / "annotations.json"

    with open(coco_json, "r", encoding="utf-8") as fp:
        coco = json.load(fp)

    anns   = build_index(coco)
    skel   = build_skeleton(coco)
    court_gt = build_court_gt(coco)

    images = sorted(coco["images"], key=lambda x: x["id"])
    total  = len(images)
    idx, paused = 0, False
    delay = int(1000 / FPS) if FPS > 0 else 1

    cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)

    while 0 <= idx < total:
        info = images[idx]
        frame = cv2.imread(str(img_root / info.get("original_path", info["file_name"])))
        if frame is None:
            print("Cannot read image:", info["file_name"])
            idx += 1
            continue

        game_id = info.get("game_id")
        # ── Court GT (任意)────────
        gt_vec = court_gt.get(game_id, [])
        for i in range(0, len(gt_vec), 3):
            x, y = gt_vec[i], gt_vec[i+1]
            cv2.circle(frame, (int(x), int(y)), 3, COLORS["court"], -1)

        # ── Annotations ───────────
        for ann in anns.get(info["id"], []):
            cid = ann["category_id"]

            # ---------- BALL ----------
            if cid == BALL_ID:
                bx, by, bw, bh = ann.get("bbox", [0,0,0,0])
                score = ann.get("score", 0.0)
                if score >= BALL_THRESH:
                    cv2.rectangle(frame, (int(bx),int(by)), (int(bx+bw), int(by+bh)), COLORS["bbox"], 2)
                    kps = ann.get("keypoints", [])
                    if len(kps) >= 3:
                        x, y, _ = kps[:3]
                        color = COLORS["ball_ev"] if ann.get("event_status", 0) else COLORS["ball"]
                        cv2.circle(frame, (int(x),int(y)), 5, color, -1)

            # ---------- COURT ----------
            elif cid == COURT_ID:
                kps   = ann.get("keypoints", [])
                scores = ann.get("keypoints_scores", [])
                for i in range(0, len(kps), 3):
                    x, y, v = kps[i:i+3]
                    conf = scores[i//3] if i//3 < len(scores) else 1.0
                    if (v > 0) and (conf >= COURT_THRESH):
                        cv2.circle(frame, (int(x), int(y)), 3, COLORS["court"], -1)

            # ---------- PLAYER / POSE ----------
            elif cid == PLAYER_ID:
                # bbox
                bx, by, bw, bh = ann.get("bbox", [0,0,0,0])
                cv2.rectangle(frame, (int(bx),int(by)), (int(bx+bw), int(by+bh)), COLORS["bbox"], 2)
                # pose
                kps = ann.get("keypoints", [])
                scr = ann.get("keypoints_scores", [])
                pts = [(kps[i], kps[i+1], kps[i+2],
                        scr[i//3] if i//3 < len(scr) else 1.0)
                       for i in range(0, len(kps), 3)]
                # joints
                for x,y,v,c in pts:
                    if v>0 and c>=POSE_THRESH:
                        cv2.circle(frame,(int(x),int(y)),2,COLORS["pose"],-1)
                # skeleton
                for a,b in skel:
                    if a < len(pts) and b < len(pts):
                        xa,ya,va,ca = pts[a]
                        xb,yb,vb,cb = pts[b]
                        if va>0 and vb>0 and ca>=POSE_THRESH and cb>=POSE_THRESH:
                            cv2.line(frame,(int(xa),int(ya)),(int(xb),int(yb)),COLORS["pose"],1)

        # title
        txt = f"Game {info.get('game_id')}  Clip {info.get('clip_id')}  Frame {idx+1}/{total}"
        cv2.putText(frame, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # resize
        h,w = frame.shape[:2]
        s = min(MAX_WIDTH/w, MAX_HEIGHT/h, 1.0)
        disp = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        cv2.imshow("Viewer", disp)

        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        if key == ord(' '):            paused = not paused
        elif key in (ord('d'), 83):    idx += 1
        elif key in (ord('a'), 81):    idx = max(0, idx-1)
        elif key == ord('c'):          # next clip
            cur_clip = info.get("clip_id")
            while idx<total and images[idx]["clip_id"]==cur_clip and images[idx]["game_id"]==game_id: idx+=1
        elif key == ord('g'):          # next game
            cur_game = game_id
            while idx<total and images[idx]["game_id"]==cur_game: idx+=1
        elif key in (ord('q'), 27):    break
        else:
            if not paused: idx += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
