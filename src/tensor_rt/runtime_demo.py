import onnxruntime as ort
import numpy as np

# 推論セッション作成（GPU 上で動くように CUDAExecutionProvider を指定）
session = ort.InferenceSession(
    "onnx_fp16/ball/lite_tracknet/lite_balltracker_fp16.onnx",       # or "lite_balltracker_int8.onnx"
    providers=["CUDAExecutionProvider"]
)

# ダミー入力
x = np.random.randn(1, 9, 360, 640).astype(np.float32)

# 推論
output = session.run(None, {"input": x})  # 入出力は通常どおり
print("output shape:", output[0].shape)
