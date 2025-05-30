#!/usr/bin/env python
"""
動画から指定した秒数範囲でクリップに分割し、フレームとして抽出してCOCO形式のメタデータを生成するスクリプト

使用例:
    python -m tools.video_clipper.extract_frames_to_unlabeled \
        --input_video data/videos/match1.mp4 \
        --output_dir datasets/ball/unlabeled/images/game1 \
        --time_ranges "[[3, 15], [25, 44], [67, 88]]" \
        --json_output datasets/ball/unlabeled/coco_annotations_unlabeled.json \
        --fps 5
"""

import argparse
import cv2
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import ast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def extract_frames_to_dir(
    input_video: Union[str, Path],
    output_dir: Union[str, Path],
    time_ranges: List[List[float]],
    clip_prefix: str = "clip",
    fps: Optional[float] = None,
    quality: int = 95,
    coco_data: Optional[Dict] = None,
    game_id: str = None,
    target_resolution: Tuple[int, int] = (640, 360),  # 幅, 高さ
) -> Dict:
    """
    動画から指定した秒数範囲でクリップを抽出し、フレームとして保存する

    Args:
        input_video: 入力動画のパス
        output_dir: 出力ディレクトリのパス
        time_ranges: 抽出するクリップの時間範囲のリスト [[start1, end1], [start2, end2], ...]
        clip_prefix: クリップディレクトリの接頭辞
        fps: 抽出するフレームのFPS（Noneの場合は動画のFPSを使用）
        quality: JPEGの品質（0-100）
        coco_data: COCO形式のデータ辞書（追記モードの場合）
        game_id: ゲームID（Noneの場合は出力ディレクトリの最後の部分を使用）
        target_resolution: 出力画像の解像度 (幅, 高さ)

    Returns:
        COCO形式のデータ辞書
    """
    input_video = Path(input_video)
    output_dir = Path(output_dir)

    if not input_video.exists():
        raise FileNotFoundError(f"入力動画が見つかりません: {input_video}")

    # 出力ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # ゲームIDが指定されていない場合は出力ディレクトリの最後の部分を使用
    if game_id is None:
        game_id = output_dir.name

    # COCO形式のデータ辞書を初期化または既存のデータを使用
    if coco_data is None:
        coco_data = {
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

    # 次のimage_idを取得
    next_image_id = 1
    if coco_data["images"]:
        next_image_id = max([img["id"] for img in coco_data["images"]]) + 1

    # 動画を開く
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {input_video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / video_fps

    logger.info(f"動画情報: {input_video}")
    logger.info(f"  FPS: {video_fps}")
    logger.info(f"  フレーム数: {frame_count}")
    logger.info(f"  元の解像度: {width}x{height}")
    logger.info(f"  出力解像度: {target_resolution[0]}x{target_resolution[1]}")
    logger.info(f"  長さ: {duration:.2f}秒")

    # 抽出するFPSを設定
    extract_fps = fps if fps is not None else video_fps
    frame_interval = int(video_fps / extract_fps)
    if frame_interval < 1:
        frame_interval = 1
        logger.warning(f"指定されたFPS({fps})が元の動画FPS({video_fps})より高いため、すべてのフレームを抽出します")

    # 各時間範囲ごとに処理
    for clip_idx, (start_sec, end_sec) in enumerate(time_ranges):
        clip_idx += 1  # 1-indexed
        clip_dir = output_dir / f"{clip_prefix}{clip_idx}"
        clip_dir.mkdir(exist_ok=True)

        logger.info(f"クリップ {clip_idx}: {start_sec}秒 - {end_sec}秒 を抽出中...")

        # 開始フレームと終了フレームを計算
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)

        if start_frame < 0:
            start_frame = 0
        if end_frame >= frame_count:
            end_frame = frame_count - 1

        # シークして開始フレームに移動
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = 0
        extracted_count = 0
        current_frame = start_frame

        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # 指定したフレーム間隔でフレームを抽出
            if frame_idx % frame_interval == 0:
                # フレームをリサイズ
                resized_frame = cv2.resize(frame, target_resolution)
                
                frame_file = clip_dir / f"frame_{extracted_count:05d}.jpg"
                cv2.imwrite(str(frame_file), resized_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

                # COCO形式のデータに画像情報を追加
                rel_path = f"{game_id}/{clip_prefix}{clip_idx}/frame_{extracted_count:05d}.jpg"
                image_entry = {
                    "id": next_image_id,
                    "file_name": rel_path,
                    "width": target_resolution[0],
                    "height": target_resolution[1],
                    "license": 1,
                    "game_id": game_id,
                    "clip_id": f"{clip_prefix}{clip_idx}",
                    "frame_idx": extracted_count,
                    "frame_idx_in_video": current_frame,
                    "source_video": str(input_video.name)
                }
                coco_data["images"].append(image_entry)
                next_image_id += 1
                extracted_count += 1

            frame_idx += 1
            current_frame += 1

        logger.info(f"  抽出完了: {extracted_count}フレーム")

    cap.release()
    return coco_data

def parse_time_ranges(time_ranges_str: str) -> List[List[float]]:
    """
    文字列形式の時間範囲をリストに変換する

    Args:
        time_ranges_str: 時間範囲の文字列（例: "[[3, 15], [25, 44], [67, 88]]"）

    Returns:
        時間範囲のリスト
    """
    try:
        return ast.literal_eval(time_ranges_str)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"時間範囲の形式が不正です: {e}")

def main():
    parser = argparse.ArgumentParser(description="動画からフレームを抽出してCOCO形式のメタデータを生成する")
    parser.add_argument("--input_video", type=str, required=True, help="入力動画のパス")
    parser.add_argument("--output_dir", type=str, required=True, help="出力ディレクトリのパス")
    parser.add_argument("--time_ranges", type=str, required=True, 
                        help="抽出するクリップの時間範囲のリスト（例: '[[3, 15], [25, 44], [67, 88]]'）")
    parser.add_argument("--clip_prefix", type=str, default="clip", help="クリップディレクトリの接頭辞")
    parser.add_argument("--fps", type=float, help="抽出するフレームのFPS（指定なしの場合は動画のFPSを使用）")
    parser.add_argument("--quality", type=int, default=95, help="JPEGの品質（0-100）")
    parser.add_argument("--json_output", type=str, required=True, help="COCO形式のJSONメタデータの出力パス")
    parser.add_argument("--game_id", type=str, help="ゲームID（指定なしの場合は出力ディレクトリの最後の部分を使用）")
    parser.add_argument("--append", action="store_true", help="既存のJSONファイルに追記する")
    parser.add_argument("--resolution", type=str, default="640,360", 
                        help="出力画像の解像度（幅,高さ）（例: '640,360'）")

    args = parser.parse_args()

    # 時間範囲の文字列をリストに変換
    time_ranges = parse_time_ranges(args.time_ranges)
    
    # 解像度の文字列をタプルに変換
    try:
        width, height = map(int, args.resolution.split(','))
        target_resolution = (width, height)
    except ValueError:
        logger.warning(f"解像度の形式が不正です: {args.resolution}、デフォルト値(640,360)を使用します")
        target_resolution = (640, 360)

    # 既存のJSONファイルを読み込む（追記モードの場合）
    coco_data = None
    if args.append and os.path.exists(args.json_output):
        try:
            with open(args.json_output, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
            logger.info(f"既存のJSONファイルを読み込みました: {args.json_output}")
        except json.JSONDecodeError:
            logger.warning(f"JSONファイルの読み込みに失敗しました: {args.json_output}")
            coco_data = None

    # フレームを抽出してCOCO形式のメタデータを生成
    coco_data = extract_frames_to_dir(
        input_video=args.input_video,
        output_dir=args.output_dir,
        time_ranges=time_ranges,
        clip_prefix=args.clip_prefix,
        fps=args.fps,
        quality=args.quality,
        coco_data=coco_data,
        game_id=args.game_id,
        target_resolution=target_resolution,
    )

    # JSONファイルを保存
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"COCO形式のメタデータを保存しました: {args.json_output}")
    logger.info(f"画像数: {len(coco_data['images'])}")

if __name__ == "__main__":
    main() 