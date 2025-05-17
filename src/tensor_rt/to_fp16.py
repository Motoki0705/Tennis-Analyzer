import onnx
from onnxconverter_common import float16

# 1) ONNX モデル読み込み
model = onnx.load("onnx/ball/lite_tracknet/lite_balltrack.onnx")

# 2) FP32 → FP16 変換
#    keep_io_types=True にすると入出力だけは float32 のままにできます
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True
)

# 3) FP16 モデルを保存
onnx.save(model_fp16, "onnx_fp16/ball/lite_tracknet/lite_balltrack_fp16.onnx")
print("✅ FP16 モデル生成完了")
