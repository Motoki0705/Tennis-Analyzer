#!/usr/bin/env python
"""
COCO形式のアノテーションファイルからボールカテゴリのevent_statusフィールドを抽出し、
新しく生成されたアノテーションファイルのボールアノテーションに追加するスクリプト
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# ボールカテゴリのID
BALL_CATEGORY_ID = 1


def load_json(file_path: str) -> Dict:
    """JSONファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path: str, data: Dict) -> None:
    """JSONファイルを保存する"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_ball_annotations(coco_data: Dict) -> List[Dict]:
    """ボールカテゴリのアノテーションを抽出する"""
    return [ann for ann in coco_data['annotations'] if ann['category_id'] == BALL_CATEGORY_ID]


def create_frame_key(img_data: Dict) -> Tuple[Any, Any, Any]:
    """画像データからフレームキーを作成する"""
    game_id = img_data.get('game_id', -1)
    clip_id = img_data.get('clip_id', -1)
    file_name = img_data.get('file_name', '')
    
    # ファイル名から数字部分を抽出（フレーム番号）
    frame_num = 0
    if file_name:
        frame_num_str = ''.join(filter(str.isdigit, file_name))
        if frame_num_str:
            frame_num = int(frame_num_str)
    
    return (game_id, clip_id, frame_num)


def build_event_status_map(source_coco: Dict) -> Dict[Tuple[Any, Any, Any], Optional[str]]:
    """ソースアノテーションからフレームキー→event_statusのマッピングを構築する"""
    ball_annotations = extract_ball_annotations(source_coco)
    image_dict = {img['id']: img for img in source_coco['images']}
    
    # フレームキー→event_statusのマッピング
    event_status_map = {}
    
    for ann in ball_annotations:
        image_id = ann['image_id']
        if image_id in image_dict:
            img_data = image_dict[image_id]
            frame_key = create_frame_key(img_data)
            
            # event_statusフィールドがある場合のみマッピングに追加
            if 'event_status' in ann:
                event_status_map[frame_key] = ann['event_status']
    
    return event_status_map


def apply_event_status(target_coco: Dict, event_status_map: Dict[Tuple[Any, Any, Any], Optional[str]]) -> Dict:
    """ターゲットアノテーションにevent_statusを適用する"""
    ball_annotations = extract_ball_annotations(target_coco)
    image_dict = {img['id']: img for img in target_coco['images']}
    
    # ボールアノテーションを更新
    updated_count = 0
    
    for ann in ball_annotations:
        image_id = ann['image_id']
        if image_id in image_dict:
            img_data = image_dict[image_id]
            frame_key = create_frame_key(img_data)
            
            # フレームキーに対応するevent_statusがある場合、アノテーションに追加
            if frame_key in event_status_map:
                ann['event_status'] = event_status_map[frame_key]
                updated_count += 1
    
    return target_coco, updated_count


def main():
    parser = argparse.ArgumentParser(
        description='既存のボールアノテーションからevent_statusを抽出し、新しいファイルに追加するツール',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--source', '-s', 
        required=False, 
        default='datasets/ball/coco_annotations_ball_pose_court.json',
        help='event_statusを取得する既存のCOCOファイル（datasets/ball/coco_annotations_ball_pose_court.json）'
    )
    parser.add_argument(
        '--target', '-t', 
        required=False, 
        default='datasets/ball/image_annotations_2025-06-05_08-55-52.json',
        help='event_statusを追加する新しく生成されたCOCOファイル（ImageAnnotatorで生成されたファイル）'
    )
    parser.add_argument(
        '--output', '-o', 
        required=False,
        default='datasets/ball/coco_annotations_ball_pose_court_event_status.json',
        help='出力ファイルパス'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='実際に上書きせずに統計情報だけ表示'
    )
    
    args = parser.parse_args()
    
    # ソースとターゲットのCOCOデータを読み込む
    source_coco = load_json(args.source)
    target_coco = load_json(args.target)
    
    # 統計情報の表示
    source_ball_anns = extract_ball_annotations(source_coco)
    target_ball_anns = extract_ball_annotations(target_coco)
    
    # ソースからevent_statusを持つアノテーション数を計算
    source_with_event_status = sum(1 for ann in source_ball_anns if 'event_status' in ann)
    
    print(f"ソースCOCO（既存のアノテーション）: {args.source}")
    print(f"  - 総画像数: {len(source_coco['images'])}")
    print(f"  - ボールアノテーション数: {len(source_ball_anns)}")
    print(f"  - event_status付きボールアノテーション数: {source_with_event_status}")
    
    print(f"\nターゲットCOCO（新しく生成されたアノテーション）: {args.target}")
    print(f"  - 総画像数: {len(target_coco['images'])}")
    print(f"  - ボールアノテーション数: {len(target_ball_anns)}")
    
    # event_statusのマッピングを構築
    event_status_map = build_event_status_map(source_coco)
    print(f"\nevent_statusマッピング数: {len(event_status_map)}")
    
    # dry-runモードの場合はここで終了
    if args.dry_run:
        # 更新予定のアノテーション数を見積もる
        _, estimated_updates = apply_event_status(target_coco.copy(), event_status_map)
        print(f"更新予定のボールアノテーション数: {estimated_updates}")
        print("dry-runモードのため、実際の上書きは行いません。")
        return
    
    # event_statusを適用
    updated_coco, updated_count = apply_event_status(target_coco, event_status_map)
    
    # 結果を保存
    save_json(args.output, updated_coco)
    
    # 結果の統計情報
    print(f"\n出力COCO: {args.output}")
    print(f"  - 総画像数: {len(updated_coco['images'])}")
    print(f"  - 総アノテーション数: {len(updated_coco['annotations'])}")
    print(f"  - ボールアノテーション数: {len(extract_ball_annotations(updated_coco))}")
    print(f"  - event_status追加数: {updated_count}")
    print("\n✅ 完了！ボールアノテーションにevent_statusを追加しました。")


if __name__ == "__main__":
    main() 