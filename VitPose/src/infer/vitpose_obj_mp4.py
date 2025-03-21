import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from PIL import Image

class VideoPoseEstimator:
    def __init__(self, input_video, output_csv, max_persons=6, detection_threshold=0.3):
        """
        コンストラクタ
        
        Parameters:
          input_video (str): 入力動画ファイルのパス
          output_csv (str): 出力CSVファイルのパス
          max_persons (int): フレームごとに処理する最大人数（デフォルトは6）
          detection_threshold (float): 人物検出の閾値（デフォルトは0.3）
        """
        self.input_video = input_video
        self.output_csv = output_csv
        self.max_persons = max_persons
        self.detection_threshold = detection_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_keypoints = None  # 後で決定
        
        self._load_models()
    
    def _load_models(self):
        """
        人物検出とポーズ推定の各モデルとプロセッサをロードします。
        """
        # 人物検出モデル
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(self.device)
        # ポーズ推定モデル
        self.pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(self.device)
    
    def process_frame(self, image):
        """
        1フレーム分の画像に対して、人物検出とポーズ推定を実施します。
        
        Parameters:
          image (PIL.Image): 入力画像
        
        Returns:
          list[dict]: 各人物のポーズ情報（キーポイント、スコア、バウンディングボックス）
        """
        # 人物検出
        inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        results_detect = self.person_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=self.detection_threshold
        )
        result = results_detect[0]
        # ラベル 0 を人物と仮定
        person_boxes = result["boxes"][result["labels"] == 0].cpu().numpy()
        if len(person_boxes) == 0:
            return []  # 検出なし
        
        # VOC形式 (x1, y1, x2, y2) から COCO形式 (x1, y1, w, h) へ変換
        person_boxes[:, 2] -= person_boxes[:, 0]
        person_boxes[:, 3] -= person_boxes[:, 1]
        
        # ポーズ推定
        inputs_pose = self.pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)
        pose_results = self.pose_processor.post_process_pose_estimation(outputs_pose, boxes=[person_boxes])
        
        persons = []
        for person_result in pose_results[0]:
            keypoints = person_result['keypoints'].cpu().numpy()  # (num_keypoints, 2)
            scores = person_result['scores'].cpu().numpy()        # (num_keypoints,)
            bbox = person_result['bbox'].cpu().numpy()             # (4,)
            persons.append({
                'keypoints': keypoints,
                'scores': scores,
                'bbox': bbox
            })
        
        # 最大 max_persons 件までに制限
        persons = persons[:self.max_persons]
        return persons
    
    def create_csv_row(self, frame_idx, persons):
        """
        1フレーム分のCSV用の1行のデータを作成します。
        人物数が max_persons 未満の場合は NaN で埋めます。
        
        Parameters:
          frame_idx (int): フレーム番号
          persons (list[dict]): 検出された人物のポーズ情報
        
        Returns:
          list: CSV出力用の1行データ
        """
        row = [frame_idx]
        for i in range(self.max_persons):
            if i < len(persons):
                kp = persons[i]['keypoints']
                sc = persons[i]['scores']
                bb = persons[i]['bbox']
                row.extend(kp.flatten().tolist())
                row.extend(sc.flatten().tolist())
                row.extend(bb.flatten().tolist())
            else:
                # キーポイント: num_keypoints * 2, スコア: num_keypoints, bbox: 4要素
                row.extend([np.nan] * (self.num_keypoints * 2))
                row.extend([np.nan] * (self.num_keypoints))
                row.extend([np.nan] * 4)
        return row
    
    def get_csv_columns(self):
        """
        CSVのヘッダー（カラム名）を作成します。
        
        Returns:
          list: カラム名リスト
        """
        columns = ['frame_idx']
        for i in range(1, self.max_persons + 1):
            # キーポイント座標 (x, y)
            for j in range(self.num_keypoints):
                columns.append(f'person{i}_kp_{j}_x')
                columns.append(f'person{i}_kp_{j}_y')
            # キーポイントのスコア
            for j in range(self.num_keypoints):
                columns.append(f'person{i}_score_{j}')
            # バウンディングボックス (x, y, w, h)
            columns.extend([f'person{i}_bbox_x', f'person{i}_bbox_y', f'person{i}_bbox_w', f'person{i}_bbox_h'])
        return columns
    
    def process_video(self):
        """
        入力動画の全フレームに対して人物検出とポーズ推定を実施し、
        結果をCSVに出力します。
        """
        cap = cv2.VideoCapture(self.input_video)
        frame_idx = 0
        all_rows = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV (BGR) から PIL (RGB) へ変換
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            persons = self.process_frame(image)
            
            # 初回またはまだ num_keypoints が設定されていない場合は、検出結果から決定
            if self.num_keypoints is None and len(persons) > 0:
                self.num_keypoints = persons[0]['keypoints'].shape[0]
            
            # もし検出結果がなく、かつまだ num_keypoints が設定されていなければ、デフォルト値を設定
            if self.num_keypoints is None:
                self.num_keypoints = 17
            
            row = self.create_csv_row(frame_idx, persons)
            all_rows.append(row)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # 動画全体で一度もフレームで人物が検出されなかった場合に備え、最終的に num_keypoints をデフォルト値に設定
        if self.num_keypoints is None:
            self.num_keypoints = 17
        
        columns = self.get_csv_columns()
        df = pd.DataFrame(all_rows, columns=columns)
        df.to_csv(self.output_csv, index=False)
        print(f"CSV saved to {self.output_csv}")

if __name__ == '__main__':
    # 設定値を直接指定
    input_video = "TrackNet/data/raw/input_video.mp4"
    output_csv = "VitPose/outputs/output.csv"
    max_persons = 6
    detection_threshold = 0.7

    estimator = VideoPoseEstimator(input_video, output_csv, max_persons, detection_threshold)
    estimator.process_video()
