import torch

from src.ball.models.lite_tracknet import LiteBallTracker
from src.utils.model_utils import load_model_weights

# 1. モデル構築＆重みロード
model = LiteBallTracker(in_channels=9, heatmap_channels=1)
ckpt_path = "checkpoints/ball/lite_tracknet/lite_tracknet-epoch=46-val_loss=0.0000.ckpt"
model = load_model_weights(model, ckpt_path)
model.eval()

# 2. ダミー入力
dummy = torch.randn(1, 9, 360, 640)

# 3. ONNXエクスポート
torch.onnx.export(
    model,
    dummy,
    "onnx/ball/lite_tracknet/lite_balltrack.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print("✅ Exported ONNX: lite_balltrack.onnx")
