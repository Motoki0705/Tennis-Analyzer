# plot_histogram.py

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def plot_score_histogram(csv_path, output_image_path, bins=30):
    """
    トラッキング結果のCSVからスコアのヒストグラムを生成し、保存する。

    Args:
        csv_path (str): 入力となるCSVファイルのパス。
        output_image_path (str): 出力するグラフ画像のパス。
        bins (int): ヒストグラムのビンの数（棒の数）。
    """
    try:
        log.info(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        log.error(f"Error: The file was not found at {csv_path}")
        return
    except Exception as e:
        log.error(f"Error reading CSV file: {e}")
        return

    # 'visible=1' のデータ、つまり実際に検出されたスコアのみを抽出
    # 'visible=0' のスコア (-inf) は分布の分析には不要なため除外する
    visible_scores_df = df[df['visible'] == 1].copy()

    if visible_scores_df.empty:
        log.error("No valid scores (where visible=1) found in the CSV. Cannot create a histogram.")
        return

    scores = visible_scores_df['score']
    
    # スコアの基本統計量を計算
    mean_score = scores.mean()
    median_score = scores.median()
    std_dev = scores.std()
    
    log.info(f"Score statistics: Mean={mean_score:.2f}, Median={median_score:.2f}, Std Dev={std_dev:.2f}")

    # グラフのスタイル設定
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # ヒストグラムをプロット
    # kde=True にすると、分布を滑らかにした曲線（カーネル密度推定）も同時に描画される
    sns.histplot(scores, bins=bins, kde=True, color='skyblue', alpha=0.7, label=f'Scores (bins={bins})')

    # 平均値と中央値の線を引く
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='green', linestyle=':', linewidth=2, label=f'Median: {median_score:.2f}')

    # グラフの装飾
    plt.title(f'Distribution of Detection Scores (Std Dev: {std_dev:.2f})', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # グラフをファイルに保存
    try:
        plt.savefig(output_image_path, dpi=300)
        log.info(f"Histogram successfully saved to {output_image_path}")
    except Exception as e:
        log.error(f"Failed to save histogram: {e}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot a histogram of tracking scores from a CSV file.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file (e.g., tracking_results.csv).")
    parser.add_argument("--output", default="score_histogram.png", help="Path to save the output histogram image.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for the histogram.")
    args = parser.parse_args()

    plot_score_histogram(args.csv, args.output, args.bins)

if __name__ == "__main__":
    main()