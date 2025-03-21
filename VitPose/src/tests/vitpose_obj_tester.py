import cv2
import pandas as pd

class OutputVideoTester:
    def __init__(self, input_video, output_csv, max_persons=6):
        """
        コンストラクタ
        
        Parameters:
          input_video (str): 入力動画ファイルのパス
          output_csv (str): 出力CSVファイルのパス
          max_persons (int): フレーム毎の最大人物数（CSV作成時と同じ値）
        """
        self.input_video = input_video
        self.output_csv = output_csv
        self.max_persons = max_persons
        
        # CSVの読み込み
        self.df = pd.read_csv(self.output_csv)
        # 例: "person1_kp_0_x" などのカラムから person1 のキーポイント数を算出
        person1_kp_columns = [col for col in self.df.columns if col.startswith("person1_kp_") and col.endswith("_x")]
        self.num_keypoints = len(person1_kp_columns)
        print(f"Detected number of keypoints: {self.num_keypoints}")
    
    def play_video(self):
        """
        入力動画を読み込み、各フレームに CSV のキーポイント情報を描画して再生します。
        """
        cap = cv2.VideoCapture(self.input_video)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # CSV の該当フレームの行が存在する場合
            if frame_idx < len(self.df):
                row = self.df.iloc[frame_idx]
                # 各人物毎にキーポイントを描画
                for person in range(1, self.max_persons + 1):
                    for kp_idx in range(self.num_keypoints):
                        x_col = f'person{person}_kp_{kp_idx}_x'
                        y_col = f'person{person}_kp_{kp_idx}_y'
                        x = row[x_col]
                        y = row[y_col]
                        # キーポイントが存在する場合（NaNでない場合）
                        if pd.notna(x) and pd.notna(y):
                            # 小さい円を描画（緑色）
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            cv2.imshow("Output Video with Keypoints", frame)
            # 30ms待機、'q'キーで終了
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 設定値を直接指定
    input_video = "TrackNet/data/raw/input_video.mp4"
    output_csv = "VitPose/outputs/output.csv"
    max_persons = 6

    tester = OutputVideoTester(input_video, output_csv, max_persons)
    tester.play_video()
