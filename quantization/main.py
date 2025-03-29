import torch
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation
)
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import os


# ========================== UTILS ==========================

def draw_predictions(image, person_boxes, keypoints_array):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # バウンディングボックス描画
    for box in person_boxes:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # キーポイント描画
    for person_keypoints in keypoints_array[0]:  # batch = 1 の場合
        for x, y, score in person_keypoints:
            if score > 0.5:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")

    return image


# ========================== SETUP ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 入力画像を読み込み
url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# ========================== STEP 1: RT-DETR 人物検出 ==========================

person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

inputs = person_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    detection_outputs = person_model(**inputs)

results = person_processor.post_process_object_detection(
    detection_outputs,
    target_sizes=torch.tensor([(image.height, image.width)]),
    threshold=0.3
)
boxes_all = results[0]["boxes"]
labels_all = results[0]["labels"]

# 人物（label == 0）のバウンディングボックスを抽出
person_boxes = boxes_all[labels_all == 0].cpu().numpy()
person_boxes[:, 2] -= person_boxes[:, 0]
person_boxes[:, 3] -= person_boxes[:, 1]

# ========================== STEP 2: ViTPose モデル → ONNX に変換 ==========================

class ONNXVitPoseWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.heatmaps

model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)
wrapper = ONNXVitPoseWrapper(model).eval()

# ダミー入力（エクスポート用）
# ダミー入力（エクスポート用）
dummy_image = torch.randn(1, 3, 256, 192)

if not os.path.exists("vitpose.onnx"):
    torch.onnx.export(
        wrapper,
        (dummy_image,),  # boxes を削除
        f="vitpose.onnx",
        input_names=["pixel_values"],
        output_names=["heatmaps"],
        opset_version=13,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "heatmaps": {0: "batch", 1: "num_boxes"}  # ※出力形式が変わらない場合
        }
    )
    print("✅ ONNX export complete.")


# ========================== STEP 3: 量子化 ==========================

if not os.path.exists("vitpose_quant.onnx"):
    quantize_dynamic(
        model_input="vitpose.onnx",
        model_output="vitpose_quant.onnx",
        weight_type=QuantType.QInt8
        )
    print("✅ Quantized model saved.")

# ========================== STEP 4: ONNX Runtime による推論 ==========================

pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
processed = pose_processor(image, boxes=[person_boxes], return_tensors="np")

pixel_values = processed["pixel_values"].astype(np.float32)

session = ort.InferenceSession("vitpose_quant.onnx", providers=["CPUExecutionProvider"])
onnx_inputs = {
    "pixel_values": pixel_values,
}

onnx_outputs = session.run(None, onnx_inputs)
keypoints_result = onnx_outputs[0]  # shape: (1, num_boxes, num_keypoints, 3)
print(keypoints_result.shape)

# ========================== STEP 5: 可視化 ==========================

drawn = draw_predictions(image, person_boxes, keypoints_result)
plt.figure(figsize=(10, 10))
plt.imshow(drawn)
plt.axis("off")
plt.show()
