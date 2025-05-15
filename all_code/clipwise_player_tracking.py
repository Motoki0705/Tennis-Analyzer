import json
import copy
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def bbox_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w/2, y + h/2])

def bbox_iou(boxA, boxB):
    xa1, ya1, wa, ha = boxA
    xb1, yb1, wb, hb = boxB

    xa2, ya2 = xa1 + wa, ya1 + ha
    xb2, yb2 = xb1 + wb, yb1 + hb

    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    boxA_area = wa * ha
    boxB_area = wb * hb

    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area else 0

class ClipwisePlayerTracker:
    def __init__(self, annotations, images, iou_thresh=0.3, center_thresh=50):
        self.annotations = annotations
        self.images = images
        self.iou_thresh = iou_thresh
        self.center_thresh = center_thresh
        self.tracks = defaultdict(list)
        self.next_player_id = 1

    def track_clip(self, clip_images):
        clip_images.sort(key=lambda x: x['file_name'])

        previous_boxes = []
        previous_ids = []

        for img_info in clip_images:
            img_id = img_info['id']
            anns = [ann for ann in self.annotations if ann['image_id'] == img_id and ann['category_id'] in [2,3]]

            current_boxes = [ann['bbox'] for ann in anns]
            current_centers = [bbox_center(bbox) for bbox in current_boxes]

            matched_ids = [-1] * len(current_boxes)

            # matching with previous frame
            for idx, curr_box in enumerate(current_boxes):
                curr_center = current_centers[idx]
                best_match = -1
                best_score = 0

                for p_idx, prev_box in enumerate(previous_boxes):
                    prev_center = bbox_center(prev_box)
                    iou = bbox_iou(curr_box, prev_box)
                    center_dist = np.linalg.norm(curr_center - prev_center)

                    if iou > self.iou_thresh and center_dist < self.center_thresh and iou > best_score:
                        best_score = iou
                        best_match = previous_ids[p_idx]

                if best_match != -1:
                    matched_ids[idx] = best_match
                    self.tracks[best_match].append(anns[idx])
                else:
                    matched_ids[idx] = self.next_player_id
                    self.tracks[self.next_player_id].append(anns[idx])
                    self.next_player_id += 1

            previous_boxes = current_boxes
            previous_ids = matched_ids

    def assign_player_category(self):
        for pid, track in self.tracks.items():
            if any(ann['category_id'] == 2 for ann in track):
                for ann in track:
                    ann['category_id'] = 2
                    ann['player_id'] = pid
                    ann['is_track_verified'] = True
            else:
                for ann in track:
                    ann['player_id'] = pid
                    ann['is_track_verified'] = False

def main(
    input_json_path: str,
    output_json_path: str,
    iou_thresh: float = 0.3,
    center_thresh: float = 50
):
    with open(input_json_path, 'r') as f:
        coco = json.load(f)

    new_coco = copy.deepcopy(coco)
    annotations = new_coco['annotations']
    images = new_coco['images']

    clip_to_images = defaultdict(list)
    for img in images:
        key = (img['game_id'], img['clip_id'])
        clip_to_images[key].append(img)

    tracker = ClipwisePlayerTracker(
        annotations=annotations,
        images=images,
        iou_thresh=iou_thresh,
        center_thresh=center_thresh
    )

    for (game_id, clip_id), clip_imgs in tqdm(clip_to_images.items(), desc='Tracking clips'):
        tracker.track_clip(clip_imgs)

    tracker.assign_player_category()

    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f, indent=2)

if __name__ == '__main__':
    input_json = r'C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_final.json'
    output_json = r'C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_tracked.json'

    main(
        input_json_path=input_json,
        output_json_path=output_json,
        iou_thresh=0.2,
        center_thresh=50
    )
