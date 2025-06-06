#!/usr/bin/env python
"""
アノテーションのevent_statusの比率を分析し、データバランス調整の提案を行うスクリプト
"""
import json
import os
import sys
import random
import argparse
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import matplotlib as mpl

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 日本語フォント設定
mpl.rcParams['font.family'] = 'sans-serif'
if sys.platform.startswith('win'):
    mpl.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
elif sys.platform.startswith('darwin'):  # macOS
    mpl.rcParams['font.sans-serif'] = ['Hiragino Sans', 'AppleGothic']
else:  # Linux等
    mpl.rcParams['font.sans-serif'] = ['IPAGothic', 'Noto Sans CJK JP']

# event_statusの名前マッピング
from src.utils.event_status_dict import event_status_dict

def analyze_event_status(annotation_file: str) -> Tuple[Dict, List]:
    """
    アノテーションファイルからevent_statusの割合を分析する
    
    Args:
        annotation_file: COCOフォーマットのアノテーションファイルパス
        
    Returns:
        Tuple[統計情報を含む辞書, アノテーションのリスト]
    """
    print(f"アノテーションファイルを読み込み中: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # カテゴリIDの取得
    category_ids = {cat["name"]: cat["id"] for cat in data["categories"]}
    ball_id = category_ids.get("ball", 1)
    
    # event_statusを持つアノテーションのみを抽出
    event_statuses = []
    ball_annotations = []
    
    for ann in data["annotations"]:
        if ann["category_id"] == ball_id and "event_status" in ann:
            event_statuses.append(ann["event_status"])
            ball_annotations.append(ann)
    
    # カウント
    status_counter = Counter(event_statuses)
    
    # 総数
    total_count = len(event_statuses)
    
    print(f"\n全体のevent_status数: {total_count}")
    
    # ステータス名の取得
    # event_status_dictの逆引き（値→キー）
    reversed_status_dict = {v: k for k, v in event_status_dict.items()}
    
    # 結果表示
    print("\n--- イベントステータスの分布 ---")
    
    results = {}
    for status, count in sorted(status_counter.items()):
        percentage = (count / total_count) * 100
        status_name = reversed_status_dict.get(status, f"Unknown({status})")
        print(f"ステータス {status} ({status_name}): {count}件 ({percentage:.2f}%)")
        results[status] = {
            "name": status_name,
            "count": count,
            "percentage": percentage
        }
    
    return {
        "total": total_count,
        "status_counts": results,
        "data": data
    }, ball_annotations

def plot_distribution(stats: Dict, output_path: str = None) -> None:
    """
    イベントステータスの分布を可視化する
    
    Args:
        stats: 統計情報を含む辞書
        output_path: 出力ファイルパス（省略可）
    """
    status_counts = stats["status_counts"]
    
    # プロット用のデータ準備
    statuses = []
    counts = []
    labels = []
    
    for status, info in sorted(status_counts.items()):
        statuses.append(status)
        counts.append(info["count"])
        labels.append(f"{info['name']} ({status}): {info['percentage']:.1f}%")
    
    # プロット
    plt.figure(figsize=(10, 6))
    bars = plt.bar(statuses, counts, color='skyblue')
    
    # ラベルと数値を表示
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            str(count),
            ha='center',
            va='bottom'
        )
    
    plt.title('イベントステータスの分布')
    plt.xlabel('ステータスID')
    plt.ylabel('アノテーション数')
    plt.xticks(statuses)
    
    # 凡例
    plt.legend(bars, labels, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"プロットを保存しました: {output_path}")
    
    plt.show()

def analyze_imbalance(stats: Dict) -> Dict:
    """
    データの不均衡を分析し、バランス調整の提案を行う
    
    Args:
        stats: 統計情報を含む辞書
    
    Returns:
        提案を含む辞書
    """
    status_counts = stats["status_counts"]
    
    # イベントステータス0と1に焦点を当てる
    if 0 not in status_counts or 1 not in status_counts:
        return {"error": "ステータス0または1が見つかりません"}
    
    count_0 = status_counts[0]["count"]
    count_1 = status_counts[1]["count"]
    ratio = count_1 / count_0
    
    print(f"\n--- データ不均衡の分析 ---")
    print(f"ステータス0 (no_hit): {count_0}件")
    print(f"ステータス1 (bounce): {count_1}件")
    print(f"比率 (1:0): {ratio:.4f}")
    
    # バランス調整の提案
    proposals = []
    
    # 1. アンダーサンプリング
    under_sample_count = count_1 * 2  # bounce数の2倍のno_hitを使用
    print(f"\n1. アンダーサンプリング（マジョリティクラスの削減）")
    print(f"   no_hitを{under_sample_count}件に削減（{count_0 - under_sample_count}件削減）")
    print(f"   結果の比率: {count_1/(under_sample_count):.4f}")
    
    proposals.append({
        "method": "under_sampling",
        "majority_count": under_sample_count,
        "minority_count": count_1,
        "ratio": count_1/(under_sample_count)
    })
    
    # 2. オーバーサンプリング
    over_sample_multiplier = 5  # bounceを5倍に増やす
    over_sample_count = count_1 * over_sample_multiplier
    print(f"\n2. オーバーサンプリング（マイノリティクラスの複製）")
    print(f"   bounceを{over_sample_count}件に増加（{over_sample_multiplier}倍に複製）")
    print(f"   結果の比率: {over_sample_count/count_0:.4f}")
    
    proposals.append({
        "method": "over_sampling",
        "majority_count": count_0,
        "minority_count": over_sample_count,
        "ratio": over_sample_count/count_0
    })
    
    # 3. ハイブリッドアプローチ
    hybrid_over = 3  # bounceを3倍に
    hybrid_under_ratio = 0.2  # no_hitを20%に削減
    
    hybrid_minority = count_1 * hybrid_over
    hybrid_majority = int(count_0 * hybrid_under_ratio)
    
    print(f"\n3. ハイブリッドアプローチ（両方の組み合わせ）")
    print(f"   bounceを{hybrid_minority}件に増加（{hybrid_over}倍）")
    print(f"   no_hitを{hybrid_majority}件に削減（元の{hybrid_under_ratio*100:.0f}%）")
    print(f"   結果の比率: {hybrid_minority/hybrid_majority:.4f}")
    
    proposals.append({
        "method": "hybrid",
        "majority_count": hybrid_majority,
        "minority_count": hybrid_minority,
        "ratio": hybrid_minority/hybrid_majority
    })
    
    # 4. 重み付け
    weight_0 = 1.0
    weight_1 = count_0 / count_1  # クラス0とクラス1の比率を重みに
    
    print(f"\n4. クラス重み付け（サンプル数はそのままに重要度を調整）")
    print(f"   no_hitの重み: {weight_0:.2f}")
    print(f"   bounceの重み: {weight_1:.2f}")
    
    proposals.append({
        "method": "class_weight",
        "weight_0": weight_0,
        "weight_1": weight_1
    })
    
    return {
        "current_ratio": ratio,
        "proposals": proposals
    }

def create_balanced_dataset(annotations: List, stats: Dict, method: str = "hybrid") -> Dict:
    """
    バランスの取れたデータセットを作成する
    
    Args:
        annotations: アノテーションのリスト
        stats: 統計情報を含む辞書
        method: バランス調整手法 ("under_sampling", "over_sampling", "hybrid")
        
    Returns:
        バランス調整されたアノテーションデータ
    """
    # 元のデータ
    data = stats["data"]
    status_counts = stats["status_counts"]
    
    # イベントステータス0と1のアノテーションを分離
    status_0_anns = [ann for ann in annotations if ann["event_status"] == 0]
    status_1_anns = [ann for ann in annotations if ann["event_status"] == 1]
    other_anns = [ann for ann in annotations if ann["event_status"] not in [0, 1]]
    
    count_0 = len(status_0_anns)
    count_1 = len(status_1_anns)
    
    # バランス調整手法に基づいて新しいアノテーションリストを作成
    balanced_anns = []
    
    if method == "under_sampling":
        # アンダーサンプリング: マジョリティクラスをランダムに削減
        under_sample_count = count_1 * 2  # マイノリティの2倍を使用
        sampled_0 = random.sample(status_0_anns, under_sample_count)
        balanced_anns = sampled_0 + status_1_anns + other_anns
        
    elif method == "over_sampling":
        # オーバーサンプリング: マイノリティクラスを複製
        over_sample_multiplier = 5
        # 複製
        oversampled_1 = []
        for _ in range(over_sample_multiplier):
            oversampled_1.extend(status_1_anns)
        
        balanced_anns = status_0_anns + oversampled_1 + other_anns
        
    elif method == "hybrid":
        # ハイブリッド: オーバーサンプリングとアンダーサンプリングの組み合わせ
        hybrid_over = 3
        hybrid_under_ratio = 0.2
        
        # アンダーサンプリング
        hybrid_majority_count = int(count_0 * hybrid_under_ratio)
        sampled_0 = random.sample(status_0_anns, hybrid_majority_count)
        
        # オーバーサンプリング
        oversampled_1 = []
        for _ in range(hybrid_over):
            oversampled_1.extend(status_1_anns)
        
        balanced_anns = sampled_0 + oversampled_1 + other_anns
    
    else:
        print(f"未知のバランス調整手法: {method}")
        return None
    
    # 新しいアノテーションカウント
    new_status_0 = len([ann for ann in balanced_anns if ann["event_status"] == 0])
    new_status_1 = len([ann for ann in balanced_anns if ann["event_status"] == 1])
    
    print(f"\n--- バランス調整後のデータ ({method}) ---")
    print(f"元のno_hit: {count_0}件 → 調整後: {new_status_0}件")
    print(f"元のbounce: {count_1}件 → 調整後: {new_status_1}件")
    print(f"元の比率: {count_1/count_0:.4f} → 調整後の比率: {new_status_1/new_status_0:.4f}")
    
    # 新しいCOCOフォーマットデータを作成
    balanced_data = data.copy()
    balanced_data["annotations"] = balanced_anns
    
    return balanced_data

def main():
    """メイン関数"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='イベントステータスの分布を分析し、バランス調整を行います。')
    parser.add_argument('--input', '-i', default="datasets/event/coco_annotations_ball_pose_court_event_status.json",
                        help='入力アノテーションファイルのパス')
    parser.add_argument('--create-balanced', '-c', action='store_true',
                        help='バランス調整されたデータセットを作成する')
    parser.add_argument('--method', '-m', choices=['under_sampling', 'over_sampling', 'hybrid'],
                        default='hybrid', help='バランス調整手法')
    parser.add_argument('--output-dir', '-o', default='outputs/analysis',
                        help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析実行
    stats, ball_annotations = analyze_event_status(args.input)
    
    # 可視化
    plot_path = output_dir / "event_status_distribution.png"
    plot_distribution(stats, str(plot_path))
    
    # イベントステータス0と1の比率を特に表示
    status_counts = stats["status_counts"]
    if 0 in status_counts and 1 in status_counts:
        count_0 = status_counts[0]["count"]
        count_1 = status_counts[1]["count"]
        total = count_0 + count_1
        
        print(f"\n--- イベントステータス0と1の比率 ---")
        print(f"ステータス0 ({status_counts[0]['name']}): {count_0}件 ({count_0/total*100:.2f}%)")
        print(f"ステータス1 ({status_counts[1]['name']}): {count_1}件 ({count_1/total*100:.2f}%)")
        print(f"比率 (1:0): {count_1/count_0:.4f}")
    
    # データバランスの分析と提案
    balance_proposals = analyze_imbalance(stats)
    
    # バランス調整したデータセットの作成
    if args.create_balanced:
        balanced_data = create_balanced_dataset(ball_annotations, stats, args.method)
        
        if balanced_data:
            # バランス調整されたデータセットを保存
            balanced_file = output_dir / f"balanced_dataset_{args.method}.json"
            with open(balanced_file, 'w', encoding='utf-8') as f:
                json.dump(balanced_data, f, ensure_ascii=False, indent=2)
            print(f"\nバランス調整されたデータセットを保存しました: {balanced_file}")
    
if __name__ == "__main__":
    main() 