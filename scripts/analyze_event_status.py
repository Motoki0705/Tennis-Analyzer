#!/usr/bin/env python
"""
アノテーションのevent_statusの割合を分析するスクリプト
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
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

def analyze_event_status(annotation_file: str) -> Dict:
    """
    アノテーションファイルからevent_statusの割合を分析する
    
    Args:
        annotation_file: COCOフォーマットのアノテーションファイルパス
        
    Returns:
        統計情報を含む辞書
    """
    print(f"アノテーションファイルを読み込み中: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # カテゴリIDの取得
    category_ids = {cat["name"]: cat["id"] for cat in data["categories"]}
    ball_id = category_ids.get("ball", 1)
    
    # event_statusを持つアノテーションのみを抽出
    event_statuses = []
    for ann in data["annotations"]:
        if ann["category_id"] == ball_id and "event_status" in ann:
            event_statuses.append(ann["event_status"])
    
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
        "status_counts": results
    }

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

def main():
    """メイン関数"""
    # デフォルトのアノテーションファイルパス
    default_annotation_file = "datasets/event/coco_annotations_ball_pose_court_event_status.json"
    
    # コマンドライン引数からファイルパスを取得（指定がなければデフォルト）
    annotation_file = sys.argv[1] if len(sys.argv) > 1 else default_annotation_file
    
    # 出力ディレクトリ
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析実行
    stats = analyze_event_status(annotation_file)
    
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
    
if __name__ == "__main__":
    main() 