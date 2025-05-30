#!/usr/bin/env python
"""
複数の動画から自動的にフレームを抽出するバッチ処理スクリプト

使用例:
    python -m tools.video_clipper.batch_extract_frames \
        --input_dir data/videos \
        --output_root datasets/ball/unlabeled/images \
        --json_output datasets/ball/unlabeled/coco_annotations_unlabeled.json \
        --clip_duration 10 \
        --clip_overlap 2 \
        --fps 5
"""

import argparse
import cv2
import json
import logging
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import glob

from extract_frames_to_unlabeled import extract_frames_to_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def generate_time_ranges(duration: float, clip_duration: float, clip_overlap: float) -> List[List[float]]:
    """
    動画の長さに基づいて時間範囲を自動的に生成する

    Args:
        duration: 動画の長さ（秒）
        clip_duration: 各クリップの長さ（秒）
        clip_overlap: クリップ間のオーバーラップ（秒）

    Returns:
        時間範囲のリスト [[start1, end1], [start2, end2], ...]
    """
    if clip_duration <= clip_overlap:
        raise ValueError("クリップの長さはオーバーラップより長くする必要があります")

    time_ranges = []
    start_time = 0
    step = clip_duration - clip_overlap

    while start_time < duration:
        end_time = min(start_time + clip_duration, duration)
        if end_time - start_time >= 1.0:  # 少なくとも1秒以上のクリップを抽出
            time_ranges.append([start_time, end_time])
        start_time += step
        if start_time >= duration:
            break

    return time_ranges

def process_video(
    video_path: Union[str, Path],
    output_root: Union[str, Path],
    json_output: Union[str, Path],
    clip_duration: float,
    clip_overlap: float,
    fps: Optional[float] = None,
    quality: int = 95,
    append: bool = True,
) -> None:
    """
    1つの動画を処理してフレームを抽出する

    Args:
        video_path: 入力動画のパス
        output_root: 出力ルートディレクトリのパス
        json_output: COCO形式のJSONメタデータの出力パス
        clip_duration: 各クリップの長さ（秒）
        clip_overlap: クリップ間のオーバーラップ（秒）
        fps: 抽出するフレームのFPS（Noneの場合は動画のFPSを使用）
        quality: JPEGの品質（0-100）
        append: 既存のJSONファイルに追記するかどうか
    """
    video_path = Path(video_path)
    
    # 動画の情報を取得
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"動画を開けませんでした: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    cap.release()

    # ゲームIDとして動画名を使用（拡張子なし）
    game_id = video_path.stem
    output_dir = Path(output_root) / game_id

    # 時間範囲を生成
    time_ranges = generate_time_ranges(duration, clip_duration, clip_overlap)

    # 既存のJSONファイルを読み込む（追記モードの場合）
    coco_data = None
    if append and os.path.exists(json_output):
        try:
            with open(json_output, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
            logger.info(f"既存のJSONファイルを読み込みました: {json_output}")
        except json.JSONDecodeError:
            logger.warning(f"JSONファイルの読み込みに失敗しました: {json_output}")
            coco_data = None

    # フレームを抽出してCOCO形式のメタデータを生成
    coco_data = extract_frames_to_dir(
        input_video=video_path,
        output_dir=output_dir,
        time_ranges=time_ranges,
        clip_prefix="clip",
        fps=fps,
        quality=quality,
        coco_data=coco_data,
        game_id=game_id,
    )

    # JSONファイルを保存
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"COCO形式のメタデータを保存しました: {json_output}")
    logger.info(f"画像数: {len(coco_data['images'])}")

def main():
    parser = argparse.ArgumentParser(description="複数の動画から自動的にフレームを抽出するバッチ処理")
    parser.add_argument("--input_dir", type=str, required=True, help="入力動画のディレクトリ")
    parser.add_argument("--output_root", type=str, required=True, help="出力ルートディレクトリのパス")
    parser.add_argument("--json_output", type=str, required=True, help="COCO形式のJSONメタデータの出力パス")
    parser.add_argument("--clip_duration", type=float, default=10.0, help="各クリップの長さ（秒）")
    parser.add_argument("--clip_overlap", type=float, default=2.0, help="クリップ間のオーバーラップ（秒）")
    parser.add_argument("--fps", type=float, help="抽出するフレームのFPS（指定なしの場合は動画のFPSを使用）")
    parser.add_argument("--quality", type=int, default=95, help="JPEGの品質（0-100）")
    parser.add_argument("--video_pattern", type=str, default="*.mp4", help="動画ファイルのパターン")

    args = parser.parse_args()

    # 入力ディレクトリから動画ファイルを取得
    input_dir = Path(args.input_dir)
    video_files = sorted(list(input_dir.glob(args.video_pattern)))
    
    if not video_files:
        logger.error(f"動画ファイルが見つかりません: {input_dir}/{args.video_pattern}")
        return

    logger.info(f"処理対象の動画ファイル: {len(video_files)}個")
    
    # 各動画を処理
    for i, video_path in enumerate(video_files):
        logger.info(f"動画 {i+1}/{len(video_files)}: {video_path}")
        process_video(
            video_path=video_path,
            output_root=args.output_root,
            json_output=args.json_output,
            clip_duration=args.clip_duration,
            clip_overlap=args.clip_overlap,
            fps=args.fps,
            quality=args.quality,
            append=(i > 0),  # 最初の動画は追記しない
        )

    logger.info("すべての動画の処理が完了しました")

if __name__ == "__main__":
    main() 