import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


# キャリブ用データリーダーの例（代表的なクリップを用意）
class BallCalibReader(CalibrationDataReader):
    def __init__(self, data_list):
        self.data_list = data_list
        self.idx = 0

    def get_next(self):
        if self.idx < len(self.data_list):
            inp = {"input": self.data_list[self.idx]}
            self.idx += 1
            return inp
        return None


# 例: 50クリップ分の NumPy 配列を用意（shape=(1,9,360,640)）
calib_data = [np.random.randn(1, 9, 360, 640).astype(np.float32) for _ in range(50)]
reader = BallCalibReader(calib_data)

quantize_static(
    model_input="onnx/ball/lite_tracknet/lite_balltrack.onnx",
    model_output="onnx/ball/lite_tracknet/lite_balltrack_statint8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QOperator,  # ★QLinearConv 形式
    weight_type=QuantType.QInt8,  # 重みも
    activation_type=QuantType.QInt8,  # 活性化も量子化
)
print("✅ 静的量子化モデル生成完了: lite_balltracker_statint8.onnx")
