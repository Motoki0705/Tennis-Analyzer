import cv2
import os
import csv
import re
from config import Config

class TennisAnnotationTool:
    def __init__(self, scale_factor=2, config: Config=None):
        self.config = config
        self.frames_folder = config.frame_directory
        self.current_scene = None  # 現在のシーンフォルダ
        self.output_csv = None  # 出力CSVファイル
        self.frame_files = []  # 現在のシーンのフレームファイル
        self.current_frame_idx = 0
        self.annotations = []  # 現在のシーンのアノテーション
        self.states = config.states  # 試合状況
        self.events = config.events  # ショット種類
        self.selected_state = self.states[0]
        self.selected_event = self.events[0]
        self.button_positions = {}
        self.scale_factor = scale_factor  # 表示倍率
        self.scene_folders = sorted(
            [os.path.join(config.frame_directory, d) for d in os.listdir(config.frame_directory) if os.path.isdir(os.path.join(config.frame_directory, d))],
            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) if re.search(r'\d+', os.path.basename(x)) else float('inf')        
            )

    def load_scene(self, scene_folder):
        """
        シーンフォルダをロード
        """
        self.current_scene = scene_folder
        self.output_csv = os.path.join(self.current_scene, f"annotations_{os.path.basename(scene_folder)}.csv")
        self.frame_files = sorted(
            [os.path.join(scene_folder, f) for f in os.listdir(scene_folder) if f.endswith('.jpg')],
            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) if re.search(r'\d+', os.path.basename(x)) else float('inf')
        )
        self.current_frame_idx = 0
        self.annotations = []

        if os.path.exists(self.output_csv):
            print(f"{self.output_csv} を読み込んでいます...")
            with open(self.output_csv, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.annotations.append({
                        'frame': int(row['frame']),
                        'state': row['state'],
                        'event': row['event']
                    })
            self.current_frame_idx = len(self.annotations)
            print(f"続きから開始します。現在のフレーム: {self.current_frame_idx}")
        else:
            print(f"{self.output_csv} が見つかりません。新しいアノテーションを開始します。")

    def save_annotations_to_csv(self):
        """
        現在のシーンのアノテーションをCSVに保存
        """
        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'state', 'event'])
            for idx, annotation in enumerate(self.annotations):
                row = [
                    idx,
                    annotation['state'],
                    annotation['event']
                ]
                writer.writerow(row)
        print(f"アノテーションを {self.output_csv} に保存しました。")

    def save_frame_annotation(self):
        """
        現在のフレームのアノテーションを保存
        """
        annotation = {
            'frame': self.current_frame_idx,
            'state': self.selected_state,
            'event': self.selected_event
        }

        if self.current_frame_idx < len(self.annotations):
            self.annotations[self.current_frame_idx] = annotation
        else:
            self.annotations.append(annotation)

        self.current_frame_idx += 1
        return True

    def draw_buttons(self, frame):
        """
        ボタンをフレーム上に描画
        """
        self.button_positions = {}
        x, y = 5, 5

        # 状態 (states) ボタン
        for state in self.states:
            color = (0, 255, 0) if state == self.selected_state else (255, 255, 255)
            text_size = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x2, y2 = x + text_size[0] + 10, y + text_size[1] + 10
            cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
            cv2.putText(frame, state, (x + 5, y + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            self.button_positions[state] = (x, y, x2, y2)
            y += 40

        y += 20

        # イベント (events) ボタン
        for event in self.events:
            color = (0, 255, 0) if event == self.selected_event else (255, 255, 255)
            text_size = cv2.getTextSize(event, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x2, y2 = x + text_size[0] + 10, y + text_size[1] + 10
            cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
            cv2.putText(frame, event, (x + 5, y + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            self.button_positions[event] = (x, y, x2, y2)
            y += 40

    def mouse_callback(self, event, x, y, flags, param):
        """
        マウスイベント処理
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # 左クリック
            for label, (x1, y1, x2, y2) in self.button_positions.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if label in self.states:
                        self.selected_state = label
                        print(f"Stateが選択されました: {self.selected_state}")
                    elif label in self.events:
                        self.selected_event = label
                        print(f"Eventが選択されました: {self.selected_event}")
                    return

    def run(self):
        """
        全シーンを処理
        """
        for scene_folder in self.scene_folders:
            print(f"シーンを処理中: {os.path.basename(scene_folder)}")
            self.load_scene(scene_folder)

            cv2.namedWindow('Frame')
            cv2.setMouseCallback('Frame', self.mouse_callback)

            while self.current_frame_idx < len(self.frame_files):
                frame_path = self.frame_files[self.current_frame_idx]
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"フレーム {frame_path} を読み込めません。")
                    self.current_frame_idx += 1
                    continue

                frame_resized = cv2.resize(frame, (frame.shape[1] * self.scale_factor, frame.shape[0] * self.scale_factor))

                while True:
                    temp_frame = frame_resized.copy()
                    self.draw_buttons(temp_frame)
                    cv2.imshow('Frame', temp_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('d'):  # 次のフレーム
                        if self.save_frame_annotation():
                            break
                    elif key == ord('q'):  # 終了
                        self.save_annotations_to_csv()
                        cv2.destroyAllWindows()
                        return

            self.save_annotations_to_csv()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = TennisAnnotationTool(frames_folder='frames')
    tool.run()
