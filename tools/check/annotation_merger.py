#!/usr/bin/env python
"""
COCO形式のアノテーションファイルからボールカテゴリのアノテーションを抽出し、
新しく生成されたアノテーションファイルのボールカテゴリを上書きするスクリプト
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


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


def get_image_id_map(source_coco: Dict, target_coco: Dict) -> Dict[int, int]:
    """ソースCOCOからターゲットCOCOへの画像IDのマッピングを作成する"""
    # 画像パスをキーとしたマッピングを作成
    source_image_map = {}
    for img in source_coco['images']:
        key = (img.get('original_path', ''), img.get('file_name', ''))
        source_image_map[key] = img['id']
    
    # ターゲットの画像IDをマッピング
    image_id_map = {}  # source_id -> target_id
    for img in target_coco['images']:
        key = (img.get('original_path', ''), img.get('file_name', ''))
        if key in source_image_map:
            image_id_map[source_image_map[key]] = img['id']
    
    return image_id_map


def merge_ball_annotations(source_coco: Dict, target_coco: Dict) -> Dict:
    """ソースからターゲットにボールアノテーションをマージする"""
    # 画像IDのマッピングを取得
    image_id_map = get_image_id_map(source_coco, target_coco)
    
    # ターゲットからボールアノテーションを削除
    target_coco['annotations'] = [
        ann for ann in target_coco['annotations'] 
        if ann['category_id'] != BALL_CATEGORY_ID
    ]
    
    # 最大アノテーションIDを取得
    max_ann_id = max([ann['id'] for ann in target_coco['annotations']], default=0)
    
    # ソースからボールアノテーションを抽出
    ball_annotations = extract_ball_annotations(source_coco)
    
    # 新しいアノテーションIDを割り当てて追加
    for i, ann in enumerate(ball_annotations):
        # 対応する画像IDがある場合のみ追加
        if ann['image_id'] in image_id_map:
            new_ann = ann.copy()
            new_ann['id'] = max_ann_id + i + 1
            new_ann['image_id'] = image_id_map[ann['image_id']]
            target_coco['annotations'].append(new_ann)
    
    return target_coco


def main():
    parser = argparse.ArgumentParser(
        description='既存のボールアノテーションを新しく生成されたCOCOファイルに上書きするツール',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--source', '-s', 
        required=True, 
        help='ボールアノテーションを取得する既存のCOCOファイル（datasets/ball/coco_annotations_ball_pose_court.json）'
    )
    parser.add_argument(
        '--target', '-t', 
        required=True, 
        help='上書き対象の新しく生成されたCOCOファイル（ImageAnnotatorで生成されたファイル）'
    )
    parser.add_argument(
        '--output', '-o', 
        required=True,
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
    
    print(f"ソースCOCO（既存のアノテーション）: {args.source}")
    print(f"  - 総画像数: {len(source_coco['images'])}")
    print(f"  - 総アノテーション数: {len(source_coco['annotations'])}")
    print(f"  - ボールアノテーション数: {len(source_ball_anns)}")
    
    print(f"\nターゲットCOCO（新しく生成されたアノテーション）: {args.target}")
    print(f"  - 総画像数: {len(target_coco['images'])}")
    print(f"  - 総アノテーション数: {len(target_coco['annotations'])}")
    print(f"  - ボールアノテーション数: {len(target_ball_anns)}")
    
    # dry-runモードの場合はここで終了
    if args.dry_run:
        image_id_map = get_image_id_map(source_coco, target_coco)
        print(f"\n対応する画像数: {len(image_id_map)}")
        
        # 追加される予定のボールアノテーション数を計算
        valid_ball_anns = [ann for ann in source_ball_anns if ann['image_id'] in image_id_map]
        print(f"追加予定のボールアノテーション数: {len(valid_ball_anns)}")
        print("dry-runモードのため、実際の上書きは行いません。")
        return
    
    # ボールアノテーションをマージ
    merged_coco = merge_ball_annotations(source_coco, target_coco)
    
    # 結果を保存
    save_json(args.output, merged_coco)
    
    # 結果の統計情報
    merged_ball_anns = extract_ball_annotations(merged_coco)
    print(f"\n出力COCO: {args.output}")
    print(f"  - 総画像数: {len(merged_coco['images'])}")
    print(f"  - 総アノテーション数: {len(merged_coco['annotations'])}")
    print(f"  - ボールアノテーション数: {len(merged_ball_anns)}")
    print(f"  - 削除されたターゲットのボールアノテーション数: {len(target_ball_anns)}")
    print(f"  - 追加されたソースのボールアノテーション数: {len(merged_ball_anns)}")
    print("\n✅ 完了！ボールアノテーションの上書きが正常に完了しました。")


if __name__ == "__main__":
    main() 