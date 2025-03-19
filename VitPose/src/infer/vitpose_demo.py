import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

def draw_predictions(image, person_boxes, keypoints_results):
    """推論結果を画像に描画する"""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # バウンディングボックスを描画
    for box in person_boxes:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # キーポイントを描画
    for person_result in keypoints_results:
        person_keypoints = person_result[0]['keypoints'].cpu().numpy()
        confidence_scores = person_result[0]['scores'].cpu().numpy()
        
        for (x, y), confidence in zip(person_keypoints, confidence_scores):
            if confidence > 0.5:  # 信頼度が高い場合のみ描画
                draw.ellipse((x-3, y-3, x+3, y+3), fill="blue")
    
    return image

# デバイスの選択
device = "cuda" if torch.cuda.is_available() else "cpu"

# 画像の取得
url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# ------------------------------------------------------------------------
# Stage 1. Detect humans on the image
# ------------------------------------------------------------------------
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

inputs = person_image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]

# 人物クラス (COCOのクラスID 0) のバウンディングボックスを取得
person_boxes = result["boxes"][result["labels"] == 0].cpu().numpy()

# VOCフォーマット (x1, y1, x2, y2) → COCOフォーマット (x1, y1, w, h)
person_boxes[:, 2] -= person_boxes[:, 0]
person_boxes[:, 3] -= person_boxes[:, 1]

# ------------------------------------------------------------------------
# Stage 2. Detect keypoints for each person found
# ------------------------------------------------------------------------
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)

inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = pose_model(**inputs)

pose_results = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
print(f'person_boxes: {person_boxes}')
print(f'pose_results: {pose_results}')

# 推論結果を画像に描画
drawn_image = draw_predictions(image, person_boxes, pose_results)

# 結果を表示
plt.figure(figsize=(10, 10))
plt.imshow(drawn_image)
plt.axis("off")
plt.show()