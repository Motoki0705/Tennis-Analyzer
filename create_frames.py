import cv2
import os
from config import Config
config = Config()
# 動画ファイルのパス
video_path = config.video_directory

# フレームを保存するルートフォルダ
output_root = config.frame_directory
os.makedirs(output_root, exist_ok=True)

# 動画を読み込む
cap = cv2.VideoCapture(video_path)

# 動画が開けない場合のエラーハンドリング
if not cap.isOpened():
    print("動画を開けませんでした")
    exit()

# 動画のFPSと総フレーム数を取得
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"動画のFPS: {fps}, 総フレーム数: {total_frames}")

# 10秒ごとに分けるためのフレーム数
frames_per_scene = fps * 10

frame_count = 0
scene_count = 1

# 最初のシーンフォルダを作成
scene_folder = os.path.join(output_root, f'scene{scene_count}')
os.makedirs(scene_folder, exist_ok=True)

# フレームを順次読み込む
while True:
    ret, frame = cap.read()
    if not ret:
        break  # フレームが取得できなければ終了

    # 10秒ごとにシーンフォルダを切り替え
    if frame_count % frames_per_scene == 0 and frame_count != 0:
        scene_count += 1
        scene_folder = os.path.join(output_root, f'scene{scene_count}')
        os.makedirs(scene_folder, exist_ok=True)
        print(f"新しいフォルダを作成しました: {scene_folder}")

    # フレームを保存
    frame_filename = os.path.join(scene_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    print(f"フレーム {frame_count} を保存しました: {frame_filename}")

    frame_count += 1

# リソース解放
cap.release()
print("すべてのフレームを保存しました。")
