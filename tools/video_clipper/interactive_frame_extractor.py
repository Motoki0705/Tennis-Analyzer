#!/usr/bin/env python
"""
動画を再生しながら手動でタイムスタンプを指定してクリップを作成し、フレームを抽出するインタラクティブなツール

使用方法:
    python -m tools.video_clipper.interactive_frame_extractor \
        --input_video data/videos/match1.mp4 \
        --output_dir datasets/ball/unlabeled/images/game1 \
        --json_output datasets/ball/unlabeled/coco_annotations_unlabeled.json \
        --fps 5

操作方法:
    スペース: 再生/一時停止
    M: 開始タイムスタンプをマーク
    N: 終了タイムスタンプをマーク
    C: マークしたタイムスタンプの範囲でクリップを作成
    E: フレームを抽出
    Q: 終了
    ←: 5秒戻る
    →: 5秒進む
    ↑/W: 再生速度アップ
    ↓/S: 再生速度ダウン
    H: ヘルプ表示
"""

import argparse
import cv2
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys

from PIL import Image, ImageDraw, ImageFont

# 同じディレクトリのモジュールをインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_frames_to_unlabeled import extract_frames_to_dir

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 日本語フォント設定（適宜パスを修正してください）
# Windows の標準フォントを指定
FONT_PATH = r"C:\Windows\Fonts\meiryo.ttc"
FONT_SIZE = 16

class InteractiveFrameExtractor:
    """
    動画を再生しながら手動でタイムスタンプを指定してクリップを作成し、フレームを抽出するインタラクティブなツール
    """
    
    def __init__(
        self,
        input_video: Union[str, Path],
        output_dir: Union[str, Path],
        json_output: Union[str, Path],
        fps: Optional[float] = None,
        quality: int = 95,
        game_id: Optional[str] = None,
        target_resolution: Tuple[int, int] = (640, 360),
    ):
        """
        初期化

        Args:
            input_video: 入力動画のパス
            output_dir: 出力ディレクトリのパス
            json_output: COCO形式のJSONメタデータの出力パス
            fps: 抽出するフレームのFPS（Noneの場合は動画のFPSを使用）
            quality: JPEGの品質（0-100）
            game_id: ゲームID（Noneの場合は出力ディレクトリの最後の部分を使用）
            target_resolution: 出力画像の解像度 (幅, 高さ)
        """
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.json_output = Path(json_output)
        self.quality = quality
        self.game_id = game_id if game_id else self.output_dir.name
        self.target_resolution = target_resolution

        # 動画キャプチャの初期化
        self.cap = cv2.VideoCapture(str(self.input_video))
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {self.input_video}")

        # 動画の情報を取得
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.video_fps
        
        # 抽出用FPSの設定
        self.extract_fps = fps if fps is not None else self.video_fps
        self.fps = self.extract_fps  # 互換性のために残す

        # 再生制御用の変数
        self.is_playing = False
        self.current_frame = 0
        self.current_time = 0.0
        self.play_speed = 1.0

        # タイムスタンプマーク用の変数
        self.start_time = None
        self.end_time = None
        self.time_ranges: List[List[float]] = []
        self.clip_count = 0

        # COCO形式のデータ辞書を初期化または既存のデータを読み込む
        self.coco_data = self._load_or_init_coco_data()

        # 表示用のウィンドウ名
        self.window_name = "Interactive Frame Extractor"

        # 日本語描画用フォント
        self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

        logger.info(f"動画情報: {self.input_video}")
        logger.info(f"  FPS: {self.video_fps}")
        logger.info(f"  抽出FPS: {self.extract_fps}")
        logger.info(f"  フレーム数: {self.frame_count}")
        logger.info(f"  元の解像度: {self.width}x{self.height}")
        logger.info(f"  出力解像度: {self.target_resolution[0]}x{self.target_resolution[1]}")
        logger.info(f"  長さ: {self.duration:.2f}秒")

    def _load_or_init_coco_data(self) -> Dict:
        """
        COCO形式のデータ辞書を初期化または既存のデータを読み込む

        Returns:
            COCO形式のデータ辞書
        """
        if self.json_output.exists():
            try:
                with open(self.json_output, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
                logger.info(f"既存のJSONファイルを読み込みました: {self.json_output}")
                return coco_data
            except json.JSONDecodeError:
                logger.warning(f"JSONファイルの読み込みに失敗しました: {self.json_output}")

        # 新規作成
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "tennis_ball",
                    "supercategory": "ball",
                    "keypoints": ["center"],
                    "skeleton": []
                }
            ]
        }

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームに情報をオーバーレイする (Pillow を使って日本語も描画)
        """
        # NumPy→Pillow
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # 各種テキストと位置
        texts = [
            (f"Time: {self.current_time:.2f}s / {self.duration:.2f}s", (10, 20), (0,255,0)),
            (f"Frame: {self.current_frame}/{self.frame_count}",     (10, 40), (0,255,0)),
            (f"Speed: {self.play_speed:.1f}x",                     (10, 60), (0,255,0)),
            (f"開始: {self.start_time:.2f}s" if self.start_time else "開始: 未設定", (10, 80), (255,255,0)),
            (f"終了: {self.end_time:.2f}s"   if self.end_time   else "終了: 未設定", (10,100), (255,255,0)),
            (f"クリップ数: {self.clip_count}",                       (10,120), (255,255,0)),
        ]
        for text, pos, color in texts:
            draw.text(pos, text, font=self.font, fill=color)

        # コントロール説明（右側に配置）
        controls = [
            "Space: 再生/一時停止",
            "M: 開始点マーク",
            "N: 終了点マーク",
            "C: クリップ作成",
            "E: フレーム抽出",
            "Q: 終了",
            "←/→: -/+ 5秒",
            "↑/W: 速度アップ",
            "↓/S: 速度ダウン",
            "H: ヘルプ表示"
        ]
        w, h = img_pil.size
        line_height = 25  # 行間（ピクセル）
        for i, ctl in enumerate(controls):
            y = 20 + i * line_height
            draw.text((w - 180, y), ctl, font=self.font, fill=(0,255,255))

        # Pillow→NumPy
        return np.array(img_pil)

    def create_clip(self):
        """
        マークしたタイムスタンプの範囲でクリップを作成する
        """
        if self.start_time is None or self.end_time is None:
            logger.warning("開始と終了の両方をマークしてください")
            return
        if self.start_time >= self.end_time:
            logger.warning("開始は終了より前にしてください")
            return

        self.time_ranges.append([self.start_time, self.end_time])
        self.clip_count += 1
        logger.info(f"クリップ {self.clip_count}: {self.start_time:.2f}s - {self.end_time:.2f}s")
        self.start_time = None
        self.end_time = None

    def extract_frames(self):
        """
        マークしたクリップからフレームを抽出する
        """
        if not self.time_ranges:
            logger.warning("クリップがありません")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coco_data = extract_frames_to_dir(
            input_video=self.input_video,
            output_dir=self.output_dir,
            time_ranges=self.time_ranges,
            clip_prefix="clip",
            fps=self.fps,
            quality=self.quality,
            coco_data=self.coco_data,
            game_id=self.game_id,
            target_resolution=self.target_resolution,
        )

        os.makedirs(os.path.dirname(self.json_output), exist_ok=True)
        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(self.coco_data, f, ensure_ascii=False, indent=2)
        logger.info(f"COCOメタデータを保存: {self.json_output} (画像数: {len(self.coco_data['images'])})")

        self.time_ranges = []
        self.clip_count = 0

    def run(self):
        """
        インタラクティブなフレーム抽出を実行
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 360)

        last_update = time.time()
        frame_time = 1.0 / self.video_fps

        logger.info("起動しました。Hキーでヘルプ表示")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                # 終了後に最初に戻す
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                self.current_time = 0.0
                continue

            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.current_time = self.current_frame / self.video_fps

            # 情報をオーバーレイ
            frame = self._draw_info_overlay(frame)
            frame = cv2.resize(frame, (640, 360))
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKeyEx(10)
            # Unicode 対応で文字に
            char = chr(key) if 0 <= key <= 0x10FFFF else ''
            c = char.lower()

            # 終了
            if key == ord('q'):
                break
            # 再生/一時停止
            elif key == ord(' '):
                self.is_playing = not self.is_playing
                logger.info("再生" if self.is_playing else "一時停止")
            # タイムマーク
            elif key == ord('m'):
                self.start_time = self.current_time
                logger.info(f"開始マーク: {self.start_time:.2f}s")
            elif key == ord('n'):
                self.end_time = self.current_time
                logger.info(f"終了マーク: {self.end_time:.2f}s")
            # クリップ & 抽出
            elif key == ord('c'):
                self.create_clip()
            elif key == ord('e'):
                self.extract_frames()
            # ヘルプ
            elif key == ord('h'):
                self.show_help()
            # シーク (←/→)
            elif key in [81, 83, 97, 99, 65361, 65363]:
                offset = -5 if key in [81, 97, 65361] else 5
                new_frame = np.clip(self.current_frame + int(offset * self.video_fps), 0, self.frame_count-1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                logger.info(f"{offset:+}秒移動 -> {new_frame/self.video_fps:.2f}s")
            # 速度調整 (↑/W, ↓/S)
            elif key in [82, 0, 101, 65362] or c == 'w':
                old = self.play_speed
                self.play_speed = min(4.0, old + 0.25)
                logger.info(f"速度: {old:.2f}x → {self.play_speed:.2f}x")
            elif key in [84, 1, 98, 65364] or c == 's':
                old = self.play_speed
                self.play_speed = max(0.25, old - 0.25)
                logger.info(f"速度: {old:.2f}x → {self.play_speed:.2f}x")

            # デバッグ用キーコード表示
            if key not in (-1, 0):
                logger.debug(f"key code: {key}")

            # 再生中フレーム送り
            if self.is_playing:
                now = time.time()
                elapsed = now - last_update
                target = frame_time / self.play_speed
                time.sleep(max(0.001, target - elapsed))
                last_update = time.time()
                if self.play_speed > 1.0:
                    skip = max(1, int((self.play_speed - 1)*2))
                    nf = min(self.frame_count-1, self.current_frame + skip)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, nf)
            else:
                time.sleep(0.05)

        # 終了時の後処理
        self.cap.release()
        cv2.destroyAllWindows()
        if self.time_ranges:
            self.extract_frames()

    def show_help(self):
        """
        ヘルプ画面を表示
        """
        help_text = """
操作方法:
    スペース: 再生/一時停止
    M: 開始タイムスタンプをマーク
    N: 終了タイムスタンプをマーク
    C: クリップを作成
    E: フレームを抽出
    Q: 終了
    ←/→: 5秒戻る/進む
    ↑/W: 再生速度アップ
    ↓/S: 再生速度ダウン
    H: このヘルプを表示
        """
        logger.info(help_text)
        # ヘルプ画像表示
        help_img = np.zeros((360, 640, 3), dtype=np.uint8)
        img_pil = Image.fromarray(help_img)
        draw = ImageDraw.Draw(img_pil)
        for i, line in enumerate(help_text.strip().split('\n')):
            draw.text((20, 30 + i*25), line, font=self.font, fill=(255,255,255))
        help_np = np.array(img_pil)
        cv2.namedWindow("Help", cv2.WINDOW_NORMAL)
        cv2.imshow("Help", help_np)
        cv2.waitKey(0)
        cv2.destroyWindow("Help")

def main():
    parser = argparse.ArgumentParser(description="インタラクティブフレーム抽出ツール")
    parser.add_argument("--input_video", type=str, required=True, help="入力動画のパス")
    parser.add_argument("--output_dir",  type=str, required=True, help="出力ディレクトリのパス")
    parser.add_argument("--json_output", type=str, required=True, help="COCO形式JSONの出力パス")
    parser.add_argument("--fps",         type=float, help="抽出FPS（未指定で動画FPS）")
    parser.add_argument("--quality",     type=int, default=95, help="JPEG品質 (0-100)")
    parser.add_argument("--game_id",     type=str, help="ゲームID (未指定で出力ディレクトリ名) ")
    parser.add_argument("--resolution",  type=str, default="640,360",
                        help="出力解像度 幅,高さ (例: '640,360')")
    args = parser.parse_args()

    # 解像度パース
    try:
        w, h = map(int, args.resolution.split(','))
        res = (w, h)
    except ValueError:
        logger.warning(f"解像度形式不正: {args.resolution} → デフォルト640x360")
        res = (640, 360)

    extractor = InteractiveFrameExtractor(
        input_video=args.input_video,
        output_dir=args.output_dir,
        json_output=args.json_output,
        fps=args.fps,
        quality=args.quality,
        game_id=args.game_id,
        target_resolution=res,
    )
    extractor.run()

if __name__ == "__main__":
    main()
