import os
import cv2
import torch
import numpy as np

from BallTrack.src.models.composed_model import ComposedModel
from BallTrack.src.train.data.dataset import TennisDataset
from BallTrack.src.train.data.transforms import prepare_transforms

class VisualizeInferFrom3Frames:
    def __init__(self, checkpoint_path, extractor='segformer', upsampler='simple'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルのロード
        self.model = ComposedModel(
            orig_channels=9,  # 3フレーム分のRGB画像を結合するため 3*3=9
            num_keypoints=1,
            extractor=extractor,
            upsampler=upsampler
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        # データセットの準備
        _, val_test_transform = prepare_transforms(resize_shape=(512, 512))
        self.dataset = TennisDataset(
            root_dir='TrackNet/datasets/Tennis',
            transform=val_test_transform,
            mode='test',
            train_ratio=0,
            val_ratio=0
        )
        self.num_samples = len(self.dataset)

        # 正規化パラメータの取得（transform から逆正規化）
        self.mean = np.array([0.485, 0.456, 0.406])  # 事前学習モデルの標準的な値
        self.std = np.array([0.229, 0.224, 0.225])  # 事前学習モデルの標準的な値

    def unnormalize(self, img):
        """ 正規化された画像を元に戻す """
        img = img * self.std + self.mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def visualize(self):
        for idx in range(self.num_samples):
            input_frames, original_frames = self.get_3frames(idx)
            inferd_coordinates = self.infer(input_frames)

            # 2フレーム目を可視化用に取得
            inferd_frame = original_frames[1].copy()  # OpenCV 用にコピー
            inferd_frame = self.unnormalize(inferd_frame)  # 正規化を元に戻す
            inferd_frame = cv2.cvtColor(inferd_frame, cv2.COLOR_RGB2BGR)  # OpenCV 用に変換

            # ヒートマップの推論座標を可視化
            y, x = inferd_coordinates
            cv2.circle(inferd_frame, (x, y), 5, (0, 0, 255), -1)  # 赤丸を描画

            # 画像を表示（連続再生用に waitKey(1)）
            cv2.imshow("Inference Result", inferd_frame)
            key = cv2.waitKey(1000)  # 1ms 待機
            if key == 27:  # ESCキーで終了
                break

        cv2.destroyAllWindows()

    def infer(self, input_frames):
        input_frames = input_frames.clone().detach().float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            heatmap = self.model(input_frames)  # 推論実行
        
        heatmap = heatmap.squeeze().cpu().numpy()
        inferd_coordinates = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        return inferd_coordinates

    def get_3frames(self, idx):
        """
        指定されたインデックスの3フレーム分のデータを取得する。
        - input_frames: モデル入力用のテンソル
        - original_frames: 可視化用のオリジナル画像
        """
        frames, _ = self.dataset.__getitem__(idx)

        # `clone().detach()` で安全にコピー
        input_frames = frames.clone().detach().float()

        # 可視化用に画像を復元
        original_frames = []
        for i in range(3):
            frame = frames[i * 3:(i + 1) * 3]  # RGBチャンネルごとに分割
            frame = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) に変換
            original_frames.append(frame)

        return input_frames, original_frames

if __name__ == '__main__':
    visualizer = VisualizeInferFrom3Frames(
        checkpoint_path='BallTrack/checkpoints/seg_simple/version_0/last.ckpt',
    )
    visualizer.visualize()
