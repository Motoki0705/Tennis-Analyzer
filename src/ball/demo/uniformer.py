import torch
from huggingface_hub import hf_hub_download
from uniformer import (
    uniformer_small,  # 公式リポジトリから（インストールされている場合）
)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのインスタンス化
model = uniformer_small()

# Hugging Face Hubから事前学習済み重みをダウンロード
model_path = hf_hub_download(
    repo_id="Sense-X/uniformer_image", filename="uniformer_small_in1k.pth"
)
state_dict = torch.load(model_path, map_location="cpu")

# 状態辞書をモデルにロード
model.load_state_dict(state_dict)

# モデルをデバイスに移動し、評価モードに設定
model = model.to(device)
model.eval()

# ここでモデルを使用できます
print("事前学習済みUniformerモデルがロードされました。")
