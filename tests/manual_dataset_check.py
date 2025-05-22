import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from src.data.datamodule import TennisDataModule
import numpy as np # For overlay

# 可視化画像の保存先
SAVE_ROOT = "dataset_samples"
os.makedirs(SAVE_ROOT, exist_ok=True)

# タスクごとの設定例
TASK_CONFIGS = {
    "ball": {
        "dataset_kwargs": {
            "annotation_file": "data/ball/coco_annotations_ball_pose_court.json",
            "image_root": "data/ball/images",
            "T": 3,
            "input_size_ball": (360, 640), # For resizing heatmap
            "heatmap_size_ball": (90, 160), # Example, ensure this matches dataset
            "input_type": "cat",
            "output_type": "all",
            "skip_frames_range": (1, 1),
        },
        "num_samples": 2,
    },
    "court": {
        "dataset_kwargs": {
            "annotation_file": "data/court/coco_court.json",
            "image_root": "data/court/images",
            "input_size_court": (256, 256),
            "heatmap_size_court": (64, 64),
            "default_num_keypoints": 15,
            "sigma": 2.0,
            "is_each_keypoint_heatmap": True,
        },
        "num_samples": 2,
    },
    "player": {
        "dataset_kwargs": {
            "annotation_file": "data/ball/coco_annotations_ball_pose_court.json",
            "image_root": "data/ball/images",
            # 必要に応じて他のパラメータも追加
        },
        "num_samples": 2,
    },
}

def visualize_and_save_ball(sample, idx, save_dir):
    image_tensor, target = sample
    if image_tensor is None or (isinstance(image_tensor, torch.Tensor) and image_tensor.numel() == 0):
        print(f"[ball] Skipping empty sample at idx={idx}")
        return
    print(f"[ball] image_tensor shape: {getattr(image_tensor, 'shape', None)}")
    if not hasattr(image_tensor, 'ndim'):
        print(f"[ball] image_tensor has no ndim, skipping idx={idx}")
        return

    processed_imgs = []  # 画像を保存するリストを追加

    # [B, C*T, H, W] (cat) の場合
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        img_seq = image_tensor[0]  # [C*T, H, W]
        C = 3
        T = img_seq.shape[0] // C
        for t in range(T):
            img = img_seq[t*C:(t+1)*C].cpu()
            processed_imgs.append(img)  # 画像をリストに追加
            plt.figure()
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title(f"Ball Sample {idx} Frame {t}")
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"ball_{idx}_frame{t}.png"))
            plt.close()
    # [B, T, C, H, W] (stack) の場合
    elif image_tensor.ndim == 5 and image_tensor.shape[0] == 1:
        for t in range(image_tensor.shape[1]):
            img = image_tensor[0, t].cpu()
            processed_imgs.append(img)  # 画像をリストに追加
            plt.figure()
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title(f"Ball Sample {idx} Frame {t}")
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"ball_{idx}_frame{t}.png"))
            plt.close()
    else:
        print(f"[ball] Unsupported shape: {image_tensor.shape} at idx={idx}, skipping.")
        return
    # ヒートマップも同様に対応
    if "heatmaps" in target:
        heatmaps_tensor = target["heatmaps"]
        if isinstance(heatmaps_tensor, torch.Tensor) and heatmaps_tensor.numel() == 0:
            print(f"[ball] Skipping empty heatmap at idx={idx}")
            return

        # Process heatmaps based on their dimension
        # Assuming batch size is 1 for visualization
        # For [B, T, H_hm, W_hm] (output_type='all' with stack/cat after dataloader)
        # or [T, H_hm, W_hm] (output_type='all' from dataset directly if batch_size=1 in dataloader and input_type='stack')
        # or [B, H_hm, W_hm] (output_type='last')
        # or [H_hm, W_hm] (output_type='last' from dataset directly)

        processed_heatmaps = []
        if heatmaps_tensor.ndim == 4: # [B, T, H_hm, W_hm]
            processed_heatmaps = [heatmaps_tensor[0, t] for t in range(heatmaps_tensor.shape[1])]
        elif heatmaps_tensor.ndim == 3: # [T, H_hm, W_hm]
            processed_heatmaps = [heatmaps_tensor[t] for t in range(heatmaps_tensor.shape[0])]
        elif heatmaps_tensor.ndim == 2: # [H_hm, W_hm] (output_type='last')
             # Need to associate with the last frame image.
             # This logic assumes we have 'processed_imgs' available from image processing part.
            if processed_imgs: # Ensure we have images to overlay on
                processed_heatmaps = [heatmaps_tensor]
            else:
                print(f"[ball] Heatmap is 2D but no processed images found to overlay for sample {idx}.")
        else:
            print(f"[ball] Unsupported heatmap ndim={heatmaps_tensor.ndim} at idx={idx}, skipping heatmap processing.")

        # Ensure number of heatmaps matches number of processed images for 'all' case
        if target.get("visibility") is not None and target["visibility"].ndim == 2 : # B, T
            vis_flags = target["visibility"][0] # Assuming B=1
        elif target.get("visibility") is not None and target["visibility"].ndim == 1: # T
            vis_flags = target["visibility"]
        else:
            vis_flags = [1] * len(processed_heatmaps) # Assume visible if not provided


        for t_idx, hmap_single_frame in enumerate(processed_heatmaps):
            if t_idx >= len(processed_imgs): # Safety check
                print(f"[ball] More heatmaps than images for sample {idx}, frame {t_idx}. Skipping extra heatmap.")
                break

            original_img_for_overlay = processed_imgs[t_idx] # This is [C,H,W] tensor
            
            # Save individual heatmap
            plt.figure()
            plt.imshow(hmap_single_frame.cpu().numpy(), cmap="hot")
            plt.title(f"Ball Heatmap {idx} Frame {t_idx} (Vis: {vis_flags[t_idx].item() if hasattr(vis_flags[t_idx], 'item') else vis_flags[t_idx]})")
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"ball_{idx}_heatmap_frame{t_idx}.png"))
            plt.close()

            # Overlay heatmap on image
            if hmap_single_frame.max() > 0.1: # Only overlay if heatmap is significant
                h_img, w_img = original_img_for_overlay.shape[1], original_img_for_overlay.shape[2]
                
                # Resize heatmap to image size
                heatmap_resized = TF.resize(hmap_single_frame.unsqueeze(0), (h_img, w_img), interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)
                heatmap_resized_np = heatmap_resized.cpu().numpy()
                
                # Normalize heatmap for colormap
                heatmap_norm = (heatmap_resized_np - heatmap_resized_np.min()) / (heatmap_resized_np.max() - heatmap_resized_np.min() + 1e-8)
                
                cmap = plt.cm.get_cmap('jet')
                heatmap_colored = cmap(heatmap_norm)[:, :, :3] # Take RGB, discard Alpha from cmap
                heatmap_colored_tensor = torch.from_numpy(heatmap_colored).permute(2, 0, 1).float()

                img_for_display_np = original_img_for_overlay.cpu().permute(1,2,0).numpy()
                # Ensure it's in 0-1 range if it was normalized, or handle other ranges if necessary
                # Assuming img_for_display_np is already suitable for plt.imshow
                
                alpha = 0.5
                overlayed_img_np = (1 - alpha) * img_for_display_np + alpha * heatmap_colored_tensor.permute(1,2,0).cpu().numpy()
                overlayed_img_np = np.clip(overlayed_img_np, 0, 1) # Clip to valid range for imshow

                plt.figure()
                plt.imshow(overlayed_img_np)
                plt.title(f"Ball Overlay {idx} Frame {t_idx}")
                plt.axis("off")
                plt.savefig(os.path.join(save_dir, f"ball_{idx}_overlay_frame{t_idx}.png"))
                plt.close()
            else:
                print(f"[ball] Heatmap max for frame {t_idx} is too low ({hmap_single_frame.max().item()}), skipping overlay.")

def visualize_and_save_court(sample, idx, save_dir):
    image_tensor, heatmaps = sample
    img = image_tensor.cpu()
    plt.figure()
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(f"Court Sample {idx}")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"court_{idx}.png"))
    plt.close()
    # ヒートマップも保存
    if heatmaps.ndim == 3:
        for k in range(heatmaps.shape[0]):
            plt.figure()
            plt.imshow(heatmaps[k].cpu().numpy(), cmap="hot")
            plt.title(f"Court Heatmap {idx} Keypoint {k}")
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"court_{idx}_heatmap{k}.png"))
            plt.close()

def visualize_and_save_player(sample, idx, save_dir):
    image_tensor, target = sample
    img = image_tensor.cpu()
    plt.figure()
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(f"Player Sample {idx}")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"player_{idx}.png"))
    plt.close()
    # バウンディングボックスを重ねて保存
    if "boxes" in target and target["boxes"].numel() > 0:
        boxes = target["boxes"]
        h, w = img.shape[1:]
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0).numpy())
        for box in boxes:
            cx, cy, bw, bh = box.tolist()
            # 正規化を元に戻す
            cx *= w
            cy *= h
            bw *= w
            bh *= h
            x = cx - bw / 2
            y = cy - bh / 2
            rect = plt.Rectangle((x, y), bw, bh, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.title(f"Player Sample {idx} with Boxes")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"player_{idx}_boxes.png"))
        plt.close()

def main():
    for task, config in TASK_CONFIGS.items():
        print(f"Processing task: {task}")
        save_dir = os.path.join(SAVE_ROOT, task)
        os.makedirs(save_dir, exist_ok=True)
        dm = TennisDataModule(
            task=task,
            dataset_kwargs=config["dataset_kwargs"],
            batch_size=1,
            num_workers=0,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        for idx, sample in enumerate(loader):
            if task == "ball":
                visualize_and_save_ball(sample, idx, save_dir)
            elif task == "court":
                visualize_and_save_court(sample, idx, save_dir)
            elif task == "player":
                visualize_and_save_player(sample, idx, save_dir)
            if idx + 1 >= config["num_samples"]:
                break
        print(f"Saved {config['num_samples']} samples for {task} in {save_dir}")

if __name__ == "__main__":
    main() 