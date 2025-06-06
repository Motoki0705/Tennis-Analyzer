#!/usr/bin/env python
"""
COCO形式のアノテーションファイルを可視化するビューワー
Ball/Court/Poseの各アノテーションを画像上に描画して表示する
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from tqdm import tqdm


# 色の定義
BALL_COLOR = (0, 255, 0)  # 緑
COURT_COLOR = (0, 255, 255)  # 黄色
PLAYER_COLORS = [
    (255, 0, 0),  # 青
    (0, 0, 255),  # 赤
]
# 骨格ライン
PLAYER_SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]


def load_coco_annotations(file_path: str) -> Dict:
    """COCO形式のアノテーションファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"アノテーションファイルの読み込みに失敗しました: {e}")
        return None


def draw_ball(image: np.ndarray, keypoints: List[float], visibility_threshold: float = 0.5) -> np.ndarray:
    """ボールアノテーションを画像上に描画"""
    if len(keypoints) < 3:
        return image
    
    x, y, v = keypoints
    if v < 1:  # visibility check
        return image
    
    # ボールの位置に円を描画
    center = (int(x), int(y))
    cv2.circle(image, center, 5, BALL_COLOR, -1)  # 塗りつぶし円
    cv2.circle(image, center, 8, BALL_COLOR, 2)  # 外枠円
    
    return image


def draw_court(image: np.ndarray, keypoints: List[float], visibility_threshold: float = 0.5) -> np.ndarray:
    """コートアノテーションを画像上に描画"""
    if len(keypoints) < 3:
        return image
    
    # キーポイントを(x, y, v)の形式に変換
    points = []
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v >= 1:  # visibility check
                points.append((int(x), int(y)))
    
    # コートの各点を描画
    for point in points:
        cv2.circle(image, point, 3, COURT_COLOR, -1)
    
    return image


def draw_player(image: np.ndarray, keypoints: List[float], bbox: List[float], player_id: int = 0) -> np.ndarray:
    """選手アノテーションを画像上に描画"""
    if len(keypoints) < 3:
        return image
    
    # 色を選択（複数の選手を区別するため）
    color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
    
    # バウンディングボックスの描画
    if bbox and len(bbox) == 4:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    # キーポイントを(x, y, v)の形式に変換
    points = []
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            points.append((int(x), int(y), int(v)))
    
    # キーポイントを描画
    for point in points:
        x, y, v = point
        if v >= 1:  # visibility check
            radius = 3 if v == 1 else 5  # 可視性によってサイズを変える
            cv2.circle(image, (x, y), radius, color, -1)
    
    # スケルトン（骨格）を描画
    for connection in PLAYER_SKELETON:
        idx1, idx2 = connection
        if idx1 < len(points) and idx2 < len(points):
            pt1 = points[idx1]
            pt2 = points[idx2]
            if pt1[2] > 0 and pt2[2] > 0:  # 両方のポイントが可視である場合のみ線を描画
                cv2.line(image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, 2)
    
    return image


def process_image(
    image_path: str, 
    annotations: List[Dict], 
    draw_balls: bool = True, 
    draw_courts: bool = True, 
    draw_players: bool = True
) -> np.ndarray:
    """画像と関連するアノテーションを処理して可視化"""
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像の読み込みに失敗しました: {image_path}")
        return None
    
    # アノテーションを画像に描画
    player_count = 0
    for ann in annotations:
        category_id = ann.get('category_id')
        
        # ボールの描画
        if category_id == 1 and draw_balls:
            keypoints = ann.get('keypoints', [])
            image = draw_ball(image, keypoints)
        
        # コートの描画
        elif category_id == 3 and draw_courts:
            keypoints = ann.get('keypoints', [])
            image = draw_court(image, keypoints)
        
        # 選手の描画
        elif category_id == 2 and draw_players:
            keypoints = ann.get('keypoints', [])
            bbox = ann.get('bbox', [])
            image = draw_player(image, keypoints, bbox, player_count)
            player_count += 1
    
    return image


def main():
    parser = argparse.ArgumentParser(description='COCO形式アノテーションの可視化ツール')
    parser.add_argument(
        '--annotation', '-a', 
        required=False, 
        default='datasets/event/coco_annotations_ball_pose_court_event_status.json',
        help='COCO形式のアノテーションファイルパス'
    )
    parser.add_argument(
        '--image-dir', '-i', 
        default='datasets/ball/images',
        help='画像ディレクトリのパス（指定がない場合はアノテーションファイルから推測）'
    )
    parser.add_argument(
        '--output-dir', '-o', 
        help='可視化結果の出力ディレクトリ（指定がない場合は表示のみ）'
    )
    parser.add_argument(
        '--no-ball', 
        action='store_true', 
        help='ボールのアノテーションを描画しない'
    )
    parser.add_argument(
        '--no-court', 
        action='store_true', 
        help='コートのアノテーションを描画しない'
    )
    parser.add_argument(
        '--no-player', 
        action='store_true', 
        help='選手のアノテーションを描画しない'
    )
    parser.add_argument(
        '--limit', '-l', 
        type=int, 
        default=0, 
        help='処理する画像の最大数（0=すべて）'
    )
    parser.add_argument(
        '--game', '-g',
        type=int,
        help='特定のゲームIDのみを処理'
    )
    parser.add_argument(
        '--clip', '-c',
        type=int,
        help='特定のクリップIDのみを処理'
    )
    
    args = parser.parse_args()
    
    # アノテーションファイルの読み込み
    coco_data = load_coco_annotations(args.annotation)
    if not coco_data:
        return
    
    # 画像とアノテーションのインデックス作成
    image_dict = {img['id']: img for img in coco_data['images']}
    annotation_dict = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotation_dict:
            annotation_dict[image_id] = []
        annotation_dict[image_id].append(ann)
    
    # 画像ディレクトリの特定
    base_dir = args.image_dir
    if not base_dir:
        # 最初の画像パスからベースディレクトリを推測
        for img in coco_data['images']:
            if 'original_path' in img:
                original_path = img['original_path']
                # ここで元のパスからベースディレクトリを推測
                # datasets/ball/images/ などを想定
                base_parts = original_path.split('/')
                if len(base_parts) > 1:
                    base_dir = '/'.join(base_parts[:-1])
                    break
    
    if not base_dir:
        base_dir = 'datasets/ball/images'  # デフォルト値
    
    base_dir = Path(base_dir)
    
    # 出力ディレクトリの設定
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像のフィルタリング
    filtered_images = coco_data['images']
    
    # ゲームIDでフィルタリング
    if args.game is not None:
        filtered_images = [img for img in filtered_images if img.get('game_id') == args.game]
    
    # クリップIDでフィルタリング
    if args.clip is not None:
        filtered_images = [img for img in filtered_images if img.get('clip_id') == args.clip]
    
    # 処理する画像数の制限
    if args.limit > 0:
        filtered_images = filtered_images[:args.limit]
    
    print(f"アノテーション: {args.annotation}")
    print(f"画像ディレクトリ: {base_dir}")
    print(f"処理対象画像数: {len(filtered_images)}")
    
    # 各画像を処理
    for img_data in tqdm(filtered_images, desc="画像処理中"):
        # 画像ファイルパスの取得
        if 'original_path' in img_data:
            # オリジナルパスがある場合はそれを使用
            img_path = Path(img_data['original_path'])
            if not img_path.is_absolute():
                img_path = base_dir / img_path
        else:
            # ファイル名のみの場合
            img_path = base_dir / img_data['file_name']
        
        # アノテーションの取得
        img_id = img_data['id']
        annotations = annotation_dict.get(img_id, [])
        
        # 画像とアノテーションの処理
        if os.path.exists(str(img_path)):
            vis_image = process_image(
                str(img_path),
                annotations,
                not args.no_ball,
                not args.no_court,
                not args.no_player
            )
            
            if vis_image is not None:
                if output_dir:
                    # 出力ファイル名の生成
                    out_file = output_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                    cv2.imwrite(str(out_file), vis_image)
                else:
                    # 画像の表示
                    cv2.imshow('Annotation Viewer', vis_image)
                    key = cv2.waitKey(0)
                    if key == 27 or key == ord('q'):  # ESCまたはqキーで終了
                        break
        else:
            print(f"警告: 画像ファイルが見つかりません: {img_path}")
    
    # OpenCVのウィンドウを閉じる
    cv2.destroyAllWindows()
    
    if output_dir:
        print(f"\n✅ 完了！可視化結果を保存しました: {output_dir}")


if __name__ == "__main__":
    main() 