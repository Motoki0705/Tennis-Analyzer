import argparse
import os
import cv2
import torch
import numpy as np

# CourtDataset クラスをインポート（パスは環境に合わせてください）
# from src.court.dataset.court_dataset import CourtDataset
# テスト用に同じディレクトリにあると仮定
from src.court.dataset.court_dataset import CourtDataset


def tensor_to_cv2_image(tensor: torch.Tensor) -> np.ndarray:
    """
    正規化された画像テンソル (C, H, W) をOpenCV画像 (H, W, C) に変換する。
    """
    # 逆正規化 (albumentations.Normalizeのデフォルト値)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # テンソルをnumpyに変換し、チャンネル順を (H, W, C) に変更
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    
    # 逆正規化処理
    image_np = (image_np * std) + mean
    image_np = np.clip(image_np, 0, 1)
    
    # 0-255の整数値に変換し、BGR形式にする
    image_np = (image_np * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_bgr

def visualize_keypoints_with_id(
    image: np.ndarray, 
    keypoints: torch.Tensor, 
    radius: int = 5, 
    font_scale: float = 0.5
) -> np.ndarray:
    """
    画像上にキーポイントのID（インデックス）を描画する。
    """
    img_copy = image.copy()
    num_keypoints = len(keypoints)
    # IDごとに安定した色を生成
    colors = [tuple(map(int, c)) for c in np.random.randint(50, 255, size=(num_keypoints, 3))]

    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # visibilityが1以上の場合のみ描画
            px, py = int(x), int(y)
            color = colors[i]
            
            # 点を描画
            cv2.circle(img_copy, (px, py), radius, color, -1, cv2.LINE_AA)
            # IDを描画 (白背景付き)
            cv2.putText(
                img_copy, str(i), (px + 5, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                img_copy, str(i), (px + 5, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA
            )
    return img_copy

def visualize_peak_valley_heatmap(
    image: np.ndarray, 
    heatmap: torch.Tensor, 
    alpha: float = 0.6
) -> np.ndarray:
    """
    単一のヒートマップ（ピークと谷を含む）を画像にオーバーレイ表示する。
    ピーク（+1に近い値）は赤、谷（-1に近い値）は青で表現する。
    """
    img_copy = image.copy()
    
    heatmap_np = heatmap.cpu().numpy()

    heatmap_scaled = ((heatmap_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    # --- この行を修正 ---
    # COLORMAP_COOLWARMの代わりに、広く利用可能なCOLORMAP_JETを使用します。
    # これにより、谷は青、背景は緑、ピークは赤で表現されます。
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)

    overlay_image = cv2.addWeighted(img_copy, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay_image

def main():
    parser = argparse.ArgumentParser(description="Visualize CourtDataset outputs with peak-valley heatmaps.")
    parser.add_argument("--json_path", required=True, help="Path to the annotation JSON file.")
    parser.add_argument("--image_root", required=True, help="Path to the root directory of images.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of the sample to visualize.")
    parser.add_argument("--output_dir", default="visualization_output", help="Directory to save visualization images.")
    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------
    # データセットをマルチチャンネル・ヒートマップモードで初期化
    # -------------------------------------------------------------------
    print("Initializing dataset in multi-channel heatmap mode with peaks and valleys...")
    dataset = CourtDataset(
        annotation_path=args.json_path,
        image_root=args.image_root,
        is_each_keypoint=True,  # マルチチャンネルモードを有効化
        use_peak_valley_heatmaps=True
    )

    if args.sample_index >= len(dataset):
        print(f"Error: sample_index {args.sample_index} is out of range. "
              f"Dataset size is {len(dataset)}.")
        return

    # -------------------------------------------------------------------
    # 指定されたサンプルを取得
    # -------------------------------------------------------------------
    print(f"Loading sample at index: {args.sample_index}")
    image_tensor, heatmaps_tensor, scaled_keypoints = dataset[args.sample_index]
    
    # 画像テンソルをOpenCV形式に変換
    image_bgr = tensor_to_cv2_image(image_tensor)

    # -------------------------------------------------------------------
    # 1. キーポイントIDの可視化 (1枚の画像)
    # -------------------------------------------------------------------
    print("\nVisualizing keypoints with their IDs...")
    keypoints_img = visualize_keypoints_with_id(image_bgr.copy(), scaled_keypoints)
    keypoints_output_path = os.path.join(args.output_dir, f"sample_{args.sample_index}_keypoints.png")
    cv2.imwrite(keypoints_output_path, keypoints_img)
    print(f"  -> Saved to: {keypoints_output_path}")

    # -------------------------------------------------------------------
    # 2. 個別のヒートマップのオーバーレイ可視化 (15枚の画像)
    # -------------------------------------------------------------------
    print("\nVisualizing each heatmap channel (peak=red, valley=blue)...")
    num_keypoints = heatmaps_tensor.shape[0]
    for i in range(num_keypoints):
        # i番目のキーポイントに対応するヒートマップを取得
        single_heatmap = heatmaps_tensor[i]

        # 可視化関数を呼び出し
        heatmap_overlay_img = visualize_peak_valley_heatmap(image_bgr.copy(), single_heatmap)
        
        # 保存
        heatmap_output_path = os.path.join(
            args.output_dir, 
            f"sample_{args.sample_index}_heatmap_kpt_{i}.png"
        )
        cv2.imwrite(heatmap_output_path, heatmap_overlay_img)
        print(f"  -> Saved heatmap for keypoint {i} to: {heatmap_output_path}")

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()