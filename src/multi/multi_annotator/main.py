# multi_flow_annotator/main.py

from pathlib import Path

# --- ダミーのPredictorクラス ---
# 実際のプロジェクトでは、これらのクラスを実際のモデルラッパーに置き換えます。
class DummyPredictor:
    """実際の予測器の代わりとなるダミークラス。"""
    def __init__(self, name="dummy", num_frames=8):
        self.name = name
        self.num_frames = num_frames
        print(f"{name.capitalize()} Predictor (Dummy) initialized.")

    def preprocess(self, data): return data, [d.shape[:2] for d in data]
    def inference(self, data): return [{"message": f"inferred by {self.name}"}] * len(data)
    def postprocess(self, data, shapes): return [{}] * len(data), None
    def preprocess_detection(self, data): return data
    def inference_detection(self, data): return data
    def postprocess_detection(self, data, frames): return [], [], [], []
    def preprocess_pose(self, images, boxes): return images
    def inference_pose(self, data): return data
    def postprocess_pose(self, outputs, boxes, scores, valid, num_frames): return [[] for _ in range(num_frames)]


def main():
    """メイン関数: アノテーターをセットアップして実行します。"""
    # --- 設定 ---
    # プロジェクトのルートディレクトリからの相対パスで指定
    # このスクリプト(main.py)が `multi_flow_annotator` フォルダ内にあることを想定
    project_root = Path(__file__).parent.parent 
    input_directory = project_root / "sample_data" # サンプルデータディレクトリを指定
    output_file = project_root / "output" / "annotations.json"
    
    print(f"入力ディレクトリ: {input_directory.resolve()}")
    print(f"出力ファイル: {output_file.resolve()}")

    # ダミーデータディレクトリとファイルの作成 (もしなければ)
    if not input_directory.exists():
        print("サンプルデータディレクトリが存在しないため、作成します。")
        (input_directory / "game1" / "clip1").mkdir(parents=True, exist_ok=True)
        # ここにダミーの.jpg画像を配置してください。
        # 例: (input_directory / "game1" / "clip1" / "frame001.jpg").touch()

    # --- 予測器のインスタンス化 (ダミーを使用) ---
    ball_predictor = DummyPredictor("ball")
    court_predictor = DummyPredictor("court")
    pose_predictor = DummyPredictor("pose")
    
    # --- アノテーターのセットアップ ---
    annotator = MultiFlowAnnotator(
        ball_predictor=ball_predictor,
        court_predictor=court_predictor,
        pose_predictor=pose_predictor,
        batch_sizes={"ball": 8, "court": 16, "pose": 4},
        vis_thresholds={"ball": 0.5, "court": 0.6, "pose": 0.5},
        max_queue_size=16,
        debug=False  # デバッグログを詳細表示する場合はTrue
    )
    
    # --- 実行 ---
    annotator.run(
        input_dir=input_directory,
        output_json=output_file
    )


if __name__ == "__main__":
    # このスクリプトを実行するには、まずサンプルデータディレクトリを作成し、
    # その中に `game1/clip1/frame001.jpg` のような構造で画像を配置する必要があります。
    main()