import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time

def visualize_img_with_heatmap(frames, heatmap, is_cat_frames):
    # 最終画像を取得
    if is_cat_frames:
        third_frame_tensor = frames[-3:, :, :]  # [3, H, W]
    else:
        third_frame_tensor = frames[-1, :, :, :] # [3, H, W]

    # denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    third_frame_tensor = third_frame_tensor * std + mean
    
    third_frame_np = TF.to_pil_image(third_frame_tensor).convert("RGB")
    third_frame_np = np.array(third_frame_np)

    # --- ヒートマップをスケーリング＆カラー化 ---
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_uint8 = np.uint8(255 * (heatmap_np / np.max(heatmap_np) + 1e-8))  # 正規化
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # --- サイズ合わせ（念のため） ---
    heatmap_color = cv2.resize(heatmap_color, (third_frame_np.shape[1], third_frame_np.shape[0]))

    # --- 合成（α合成） ---
    alpha = 0.5
    overlay = cv2.addWeighted(third_frame_np, 1 - alpha, heatmap_color, alpha, 0)

    # --- 表示 or 保存 ---
    cv2.imwrite("heatmap_overlay.png", overlay)
    print("Saved overlay as heatmap_overlay.png")


def play_overlay_sequence(frames: np.ndarray, heatmaps: np.ndarray, batch_idx: int = 0, fps: float = 5.0):
    """
    バッチ内の連続フレームとヒートマップを動画のように表示します。

    Parameters
    ----------
    frames : np.ndarray or torch.Tensor
        形状 [B, 3, N, H, W]
    heatmaps : np.ndarray or torch.Tensor
        形状 [B, 1, N, H, W]
    batch_idx : int
        表示するサンプルのインデックス
    fps : float
        フレームレート（1秒あたりのフレーム数）
    """
    # Tensor→numpy
    if hasattr(frames, 'cpu'):
        frames = frames.cpu().numpy()
    if hasattr(heatmaps, 'cpu'):
        heatmaps = heatmaps.cpu().numpy()

    B, C, N, H, W = frames.shape

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6 * H / W))
    ax.axis('off')
    im = ax.imshow(np.zeros((H, W, 3), dtype=np.float32))

    interval = 1.0 / fps
    cmap = plt.get_cmap('jet')

    for t in range(N):
        # 元画像
        img = frames[batch_idx, :, t]        # [3, H, W]
        img = np.transpose(img, (1, 2, 0))   # [H, W, 3]

        # ヒートマップ
        hm = heatmaps[batch_idx, 0, t]       # [H, W]
        hm_color = cmap(hm)[:, :, :3]        # [H, W, 3]

        # オーバーレイ（半透明）
        overlay = img * 0.6 + hm_color * 0.4
        overlay = np.clip(overlay, 0, 1)

        im.set_data(overlay)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval)

    plt.ioff()
    plt.show()

def play_overlay_sequence_xy(frames, coords, vis, fps=5.0, r=4):
    """
    frames : [3,N,H,W]
    coords : [N,2]  (正規化 x,y)
    vis    : [N]
    """
    frames = frames.cpu().numpy() if hasattr(frames,'cpu') else frames
    coords = coords.cpu().numpy() if hasattr(coords,'cpu') else coords
    vis    = vis.cpu().numpy()    if hasattr(vis,'cpu')    else vis

    _, N, H, W = frames.shape
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6*H/W))
    ax.axis("off")
    im = ax.imshow(np.zeros((H,W,3),dtype=np.float32))
    circ = plt.Circle((0,0), r, color='red')
    ax.add_patch(circ)

    for t in range(N):
        frame = np.transpose(frames[:,t], (1,2,0))
        im.set_data(frame)
        if vis[t] > 0:
            cx = coords[t,0] * W
            cy = coords[t,1] * H
            circ.center = (cx, cy)
            circ.set_visible(True)
        else:
            circ.set_visible(False)
        fig.canvas.draw()
        plt.pause(1.0/fps)

    plt.ioff()
    plt.show()