import json
import os
import random
from pathlib import Path

def extract_samples(json_file_path, output_dir, num_samples=5):
    """
    JSONファイルからサンプルをランダムに抽出し、別ファイルとして保存する

    Args:
        json_file_path (str): 入力JSONファイルのパス
        output_dir (str): 出力ディレクトリのパス
        num_samples (int, optional): 抽出するサンプル数。デフォルトは5
    
    Returns:
        dict: 抽出されたサンプルデータ
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"JSONファイル '{json_file_path}' からサンプルを抽出中...")
    
    # JSONファイルを読み込む
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データの構造を分析
    print("\nデータ構造:")
    for key, value in data.items():
        if isinstance(value, list):
            print(f"- {key}: リスト ({len(value)}件)")
        else:
            print(f"- {key}: {type(value).__name__}")
    
    # サンプルを抽出
    sample_data = {}
    
    # 基本構造をコピー
    for key in data.keys():
        if key != "images" and key != "annotations":
            sample_data[key] = data[key]
    
    # 画像をランダムに選択
    if "images" in data and len(data["images"]) > 0:
        sample_images = random.sample(data["images"], min(num_samples, len(data["images"])))
        sample_data["images"] = sample_images
        
        # 選択した画像のIDを取得
        image_ids = [img["id"] for img in sample_images]
        
        # 選択した画像に関連するアノテーションを抽出
        if "annotations" in data:
            sample_annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]
            sample_data["annotations"] = sample_annotations
    
    # サンプルデータを保存
    output_file = os.path.join(output_dir, "sample_annotations.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{len(sample_data.get('images', []))}件の画像と{len(sample_data.get('annotations', []))}件のアノテーションを抽出しました")
    print(f"サンプルデータを '{output_file}' に保存しました")
    
    return sample_data

def analyze_sample(sample_data):
    """
    サンプルデータの詳細構造を分析する
    
    Args:
        sample_data (dict): 分析するサンプルデータ
    """
    print("\n===== サンプルデータの詳細分析 =====")
    
    # カテゴリ情報
    if "categories" in sample_data:
        print("\n【カテゴリ情報】")
        for category in sample_data["categories"]:
            print(f"ID: {category.get('id')}, 名前: {category.get('name')}")
    
    # 画像情報
    if "images" in sample_data and len(sample_data["images"]) > 0:
        print("\n【画像情報のサンプル】")
        sample_image = sample_data["images"][0]
        print(f"画像ID: {sample_image.get('id')}")
        print(f"ファイル名: {sample_image.get('file_name')}")
        print(f"幅: {sample_image.get('width')}, 高さ: {sample_image.get('height')}")
        
        # その他のキーを表示
        other_keys = [k for k in sample_image.keys() if k not in ['id', 'file_name', 'width', 'height']]
        if other_keys:
            print("その他の属性:")
            for key in other_keys:
                print(f"  - {key}: {sample_image.get(key)}")
    
    # アノテーション情報
    if "annotations" in sample_data and len(sample_data["annotations"]) > 0:
        print("\n【アノテーション情報のサンプル】")
        sample_ann = sample_data["annotations"][0]
        print(f"アノテーションID: {sample_ann.get('id')}")
        print(f"画像ID: {sample_ann.get('image_id')}")
        print(f"カテゴリID: {sample_ann.get('category_id')}")
        
        if "bbox" in sample_ann:
            print(f"バウンディングボックス: {sample_ann['bbox']}")
        
        if "segmentation" in sample_ann:
            seg_type = "ポリゴン" if isinstance(sample_ann["segmentation"], list) else "RLE"
            print(f"セグメンテーション: {seg_type}形式")
        
        if "keypoints" in sample_ann:
            print(f"キーポイント数: {len(sample_ann['keypoints']) // 3}")
        
        # その他のキーを表示
        other_keys = [k for k in sample_ann.keys() if k not in ['id', 'image_id', 'category_id', 'bbox', 'segmentation', 'keypoints']]
        if other_keys:
            print("その他の属性:")
            for key in other_keys:
                print(f"  - {key}: {sample_ann.get(key)}")

if __name__ == "__main__":
    # 入力ファイルと出力ディレクトリを指定
    input_file = "datasets/ball/coco_annotations_ball_pose_court.json"
    output_dir = "datasets/ball/samples"
    
    # サンプルを抽出
    sample_data = extract_samples(input_file, output_dir, num_samples=5)
    
    # サンプルを分析
    analyze_sample(sample_data) 