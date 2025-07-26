import torch

def analyze_checkpoint(path: str):
    """
    PyTorch Lightningのチェックポイントファイルを読み込み、中身を解析して表示する。

    Args:
        path (str): チェックポイントファイルのパス (.ckpt)
    """
    # GPUがない環境でもエラーが出ないように、CPUにマッピングして読み込む
    try:
        # PyTorch 2.0.0以上を推奨
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        # 古いPyTorchバージョンの場合
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

    print(f"✅ チェックポイントファイルをロードしました: {path}")
    print("-" * 50)

    # --- 基本情報の表示 ---
    print("🔑 全てのキー:")
    print(list(checkpoint.keys()))
    print("-" * 50)

    print("ℹ️ 基本情報:")
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"  - Global Step: {checkpoint['global_step']}")
    if 'pytorch-lightning_version' in checkpoint:
        print(f"  - PyTorch Lightning Version: {checkpoint['pytorch-lightning_version']}")
    if 'hyper_parameters' in checkpoint and checkpoint['hyper_parameters']:
        print("  - Hyperparameters:")
        for key, value in checkpoint['hyper_parameters'].items():
            print(f"    - {key}: {value}")
    print("-" * 50)

    # --- state_dict (モデルの重み) の解析 ---
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"📦 state_dict には {len(state_dict)} 個のパラメータが含まれています。")
        print("以下に最初の10個のレイヤー名とテンソルの形状を表示します。")
        
        count = 0
        for layer_name, params in state_dict.items():
            print(f"  - {layer_name}: {list(params.shape)}")
            count += 1
            if count >= 10:
                break
    else:
        print("⚠️ state_dictが見つかりませんでした。")

# --- 使用例 ---
if __name__ == '__main__':
    try:
        # ▼▼▼ 解析したいチェックポイントファイルのパスを指定してください ▼▼▼
        checkpoint_path = 'checkpoints/ball/conv3d_tsm_fpn/best_model.ckpt' 
        # ▲▲▲ ここを書き換える ▲▲▲

        analyze_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print("="*50)
        print("🚨エラー: 指定されたファイルが見つかりません。")
        print("`checkpoint_path` を実際の.ckptファイルのパスに書き換えてください。")
        print("例: checkpoint_path = 'lightning_logs/version_0/checkpoints/sample.ckpt'")
        print("="*50)
    except Exception as e:
        print(f"エラーが発生しました: {e}")