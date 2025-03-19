import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

class CSVVideoVisualizer:
    def __init__(self, input_video, input_csv, output_video=None, point_color=(0, 0, 255), point_radius=5):
        """
        Parameters:
          input_video (str): 入力動画のパス
          input_csv (str): ボール検出結果のCSVのパス
          output_video (str, optional): 出力動画のパス（Noneの場合は再生のみ）
          point_color (tuple): 描画する点の色 (B, G, R)
          point_radius (int): 描画する点の半径
        """
        self.input_video = input_video
        self.input_csv = input_csv
        self.output_video = output_video
        self.point_color = point_color
        self.point_radius = point_radius
        self.data = pd.read_csv(input_csv)
        
    def visualize(self):
        cap = cv2.VideoCapture(self.input_video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video, fourcc, fps, (frame_width, frame_height))
        else:
            out = None

        pbar = tqdm(total=total_frames, desc="Processing video")
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 該当フレームの検出データを取得
            frame_data = self.data[self.data['frame_idx'] == frame_idx]
            for _, row in frame_data.iterrows():
                if not np.isnan(row['x']) and not np.isnan(row['y']):
                    x, y = int(row['x']), int(row['y'])
                    cv2.circle(frame, (x, y), self.point_radius, self.point_color, -1)
            
            cv2.imshow('Ball Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if out:
                out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if self.output_video:
            print(f"Output video saved to {self.output_video}")

if __name__ == '__main__':
    input_video = r"BallTrack\data\raw\input_video.mp4"
    input_csv = r"BallTrack\outputs\output_detections.csv"
    output_video = r"BallTrack\outputs\visualized_output.mp4"  # 出力動画を保存したい場合
    
    visualizer = CSVVideoVisualizer(input_video, input_csv, output_video)
    visualizer.visualize()
