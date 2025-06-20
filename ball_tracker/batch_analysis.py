"""
Batch Analysis Tool for Ball Tracker
複数動画の一括分析と結果比較ツール
"""

import argparse
import os
import json
import glob
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

from .analysis_tool import BallTrackerAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchAnalyzer:
    """複数動画の一括分析クラス"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        
    def analyze_directory(self, video_dir: str, output_dir: str = None, 
                         confidence_thresholds: List[float] = [0.3, 0.5, 0.7, 0.8, 0.9],
                         max_workers: int = 1, video_extensions: List[str] = None) -> Dict:
        """
        ディレクトリ内の全動画を分析
        
        Args:
            video_dir (str): 動画ディレクトリ
            output_dir (str): 結果出力ディレクトリ
            confidence_thresholds (List[float]): 分析閾値
            max_workers (int): 並列処理数
            video_extensions (List[str]): 対象動画拡張子
            
        Returns:
            Dict: 統合分析結果
        """
        if output_dir is None:
            output_dir = f"batch_analysis_results/{Path(video_dir).name}"
        os.makedirs(output_dir, exist_ok=True)
        
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            
        # 動画ファイル検索
        video_files = []
        for ext in video_extensions:
            pattern = os.path.join(video_dir, f"**/*{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))
            
        if not video_files:
            raise ValueError(f"動画ファイルが見つかりません: {video_dir}")
            
        logger.info(f"発見された動画: {len(video_files)}本")
        
        # 分析実行
        if max_workers > 1:
            results = self._analyze_parallel(video_files, output_dir, confidence_thresholds, max_workers)
        else:
            results = self._analyze_sequential(video_files, output_dir, confidence_thresholds)
            
        # 統合分析
        integrated_results = self._integrate_results(results, output_dir, confidence_thresholds)
        
        # 比較可視化
        self._generate_comparison_charts(integrated_results, output_dir)
        
        # レポート生成
        self.generate_report(integrated_results, output_dir)
        
        logger.info(f"バッチ分析完了。結果保存: {output_dir}")
        return integrated_results
        
    def _analyze_sequential(self, video_files: List[str], output_dir: str, 
                           confidence_thresholds: List[float]) -> List[Dict]:
        """逐次処理での分析"""
        results = []
        
        for video_file in tqdm(video_files, desc="動画分析中"):
            try:
                analyzer = BallTrackerAnalyzer(self.model_path, self.device)
                video_output_dir = os.path.join(output_dir, Path(video_file).stem)
                result = analyzer.analyze_video(video_file, video_output_dir, confidence_thresholds)
                results.append(result)
                
            except Exception as e:
                logger.error(f"動画分析エラー {video_file}: {e}")
                
        return results
        
    def _analyze_parallel(self, video_files: List[str], output_dir: str, 
                         confidence_thresholds: List[float], max_workers: int) -> List[Dict]:
        """並列処理での分析"""
        results = []
        
        def analyze_single_video(video_file):
            try:
                analyzer = BallTrackerAnalyzer(self.model_path, self.device)
                video_output_dir = os.path.join(output_dir, Path(video_file).stem)
                return analyzer.analyze_video(video_file, video_output_dir, confidence_thresholds)
            except Exception as e:
                logger.error(f"動画分析エラー {video_file}: {e}")
                return None
                
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {executor.submit(analyze_single_video, video): video 
                              for video in video_files}
            
            for future in tqdm(as_completed(future_to_video), 
                              total=len(video_files), desc="動画分析中"):
                result = future.result()
                if result is not None:
                    results.append(result)
                    
        return results
        
    def _integrate_results(self, results: List[Dict], output_dir: str, 
                          confidence_thresholds: List[float]) -> Dict:
        """分析結果の統合"""
        
        # データフレーム用のリスト
        summary_data = []
        threshold_data = []
        
        for result in results:
            video_name = Path(result['video_info']['path']).stem
            stats = result['statistics']['basic_stats']
            
            # 基本統計
            summary_data.append({
                'video_name': video_name,
                'total_frames': stats['total_processed_frames'],
                'visible_frames': stats['visible_frames'],
                'visibility_ratio': stats['visibility_ratio'],
                'avg_detections_per_frame': stats['avg_detections_per_frame'],
                'total_detections': stats['total_detections'],
                'overall_avg_score': stats['overall_avg_score'],
                'overall_std_score': stats['overall_std_score'],
                'trajectory_smoothness': stats['trajectory_smoothness'],
                'tracking_length': stats['tracking_length'],
                'video_duration': result['video_info']['duration_sec'],
                'fps': result['video_info']['fps']
            })
            
            # 閾値別統計
            for threshold, thresh_stats in result['statistics']['threshold_analysis'].items():
                threshold_data.append({
                    'video_name': video_name,
                    'threshold': threshold,
                    'valid_frame_ratio': thresh_stats['valid_frame_ratio'],
                    'avg_score': thresh_stats['avg_score'],
                    'total_detections': thresh_stats['total_detections']
                })
                
        # データフレーム作成
        summary_df = pd.DataFrame(summary_data)
        threshold_df = pd.DataFrame(threshold_data)
        
        # 統計サマリー計算
        integrated_stats = {
            'total_videos': len(results),
            'total_frames': summary_df['total_frames'].sum(),
            'avg_visibility_ratio': summary_df['visibility_ratio'].mean(),
            'avg_score_overall': summary_df['overall_avg_score'].mean(),
            'avg_smoothness': summary_df['trajectory_smoothness'].mean(),
            'best_video': summary_df.loc[summary_df['visibility_ratio'].idxmax()]['video_name'],
            'worst_video': summary_df.loc[summary_df['visibility_ratio'].idxmin()]['video_name']
        }
        
        # 閾値別推奨値計算
        threshold_recommendations = {}
        for threshold in confidence_thresholds:
            thresh_subset = threshold_df[threshold_df['threshold'] == threshold]
            threshold_recommendations[threshold] = {
                'avg_valid_ratio': thresh_subset['valid_frame_ratio'].mean(),
                'avg_score': thresh_subset['avg_score'].mean(),
                'videos_with_good_ratio': len(thresh_subset[thresh_subset['valid_frame_ratio'] > 0.5])
            }
            
        integrated_results = {
            'summary_dataframe': summary_df,
            'threshold_dataframe': threshold_df,
            'integrated_statistics': integrated_stats,
            'threshold_recommendations': threshold_recommendations,
            'individual_results': results
        }
        
        # 結果保存
        summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
        threshold_df.to_csv(os.path.join(output_dir, 'threshold_statistics.csv'), index=False)
        
        with open(os.path.join(output_dir, 'integrated_results.json'), 'w', encoding='utf-8') as f:
            # DataFrameは除外してJSON保存
            save_data = {k: v for k, v in integrated_results.items() 
                        if not isinstance(v, pd.DataFrame)}
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        return integrated_results
        
    def _generate_comparison_charts(self, integrated_results: Dict, output_dir: str):
        """比較可視化の生成"""
        
        summary_df = integrated_results['summary_dataframe']
        threshold_df = integrated_results['threshold_dataframe']
        
        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 動画別基本統計比較
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 可視率比較
        axes[0, 0].bar(range(len(summary_df)), summary_df['visibility_ratio'])
        axes[0, 0].set_title('動画別可視率')
        axes[0, 0].set_ylabel('可視率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 平均確信度比較
        axes[0, 1].bar(range(len(summary_df)), summary_df['overall_avg_score'])
        axes[0, 1].set_title('動画別平均確信度')
        axes[0, 1].set_ylabel('平均確信度')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 軌跡滑らかさ比較
        axes[0, 2].bar(range(len(summary_df)), summary_df['trajectory_smoothness'])
        axes[0, 2].set_title('動画別軌跡滑らかさ')
        axes[0, 2].set_ylabel('滑らかさ指標')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 散布図: 可視率 vs 確信度
        axes[1, 0].scatter(summary_df['visibility_ratio'], summary_df['overall_avg_score'], alpha=0.7)
        axes[1, 0].set_xlabel('可視率')
        axes[1, 0].set_ylabel('平均確信度')
        axes[1, 0].set_title('可視率 vs 平均確信度')
        
        # 検出数分布
        axes[1, 1].hist(summary_df['avg_detections_per_frame'], bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('平均検出数/フレーム')
        axes[1, 1].set_ylabel('動画数')
        axes[1, 1].set_title('フレーム当たり検出数分布')
        
        # 動画時間 vs 可視率
        axes[1, 2].scatter(summary_df['video_duration'], summary_df['visibility_ratio'], alpha=0.7)
        axes[1, 2].set_xlabel('動画時間(秒)')
        axes[1, 2].set_ylabel('可視率')
        axes[1, 2].set_title('動画時間 vs 可視率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'video_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 閾値別統計ヒートマップ
        plt.figure(figsize=(14, 8))
        
        # 閾値別有効フレーム率のヒートマップ用データ準備
        pivot_data = threshold_df.pivot(index='video_name', columns='threshold', values='valid_frame_ratio')
        
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': '有効フレーム率'})
        plt.title('動画×閾値別 有効フレーム率')
        plt.ylabel('動画')
        plt.xlabel('確信度閾値')
        
        # 閾値別平均確信度
        pivot_score = threshold_df.pivot(index='video_name', columns='threshold', values='avg_score')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_score, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': '平均確信度'})
        plt.title('動画×閾値別 平均確信度')
        plt.ylabel('動画')
        plt.xlabel('確信度閾値')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 推奨閾値分析
        plt.figure(figsize=(12, 8))
        
        thresh_rec = integrated_results['threshold_recommendations']
        thresholds = list(thresh_rec.keys())
        avg_ratios = [thresh_rec[th]['avg_valid_ratio'] for th in thresholds]
        avg_scores = [thresh_rec[th]['avg_score'] for th in thresholds]
        good_videos = [thresh_rec[th]['videos_with_good_ratio'] for th in thresholds]
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, avg_ratios, 'o-', linewidth=2, markersize=8)
        plt.xlabel('確信度閾値')
        plt.ylabel('平均有効フレーム率')
        plt.title('閾値別 平均有効フレーム率')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, avg_scores, 'o-', linewidth=2, markersize=8, color='orange')
        plt.xlabel('確信度閾値')
        plt.ylabel('平均確信度')
        plt.title('閾値別 平均確信度')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.bar(range(len(thresholds)), good_videos, alpha=0.7)
        plt.xlabel('確信度閾値')
        plt.ylabel('良好な動画数(有効率>0.5)')
        plt.title('閾値別 良好動画数')
        plt.xticks(range(len(thresholds)), [f'{th:.1f}' for th in thresholds])
        plt.grid(True, alpha=0.3)
        
        # 統計サマリー表示
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        stats = integrated_results['integrated_statistics']
        summary_text = f"""
バッチ分析サマリー:

• 総動画数: {stats['total_videos']}
• 総フレーム数: {stats['total_frames']:,}
• 平均可視率: {stats['avg_visibility_ratio']:.3f}
• 平均確信度: {stats['avg_score_overall']:.3f}
• 平均軌跡滑らかさ: {stats['avg_smoothness']:.2f}

最高性能動画: {stats['best_video']}
最低性能動画: {stats['worst_video']}
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"比較チャート保存: {output_dir}/")
        
    def generate_report(self, integrated_results: Dict, output_dir: str):
        """分析レポート生成"""
        
        report_file = os.path.join(output_dir, 'analysis_report.md')
        
        stats = integrated_results['integrated_statistics']
        thresh_rec = integrated_results['threshold_recommendations']
        summary_df = integrated_results['summary_dataframe']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Ball Tracker バッチ分析レポート\n\n")
            
            f.write("## 📊 全体サマリー\n")
            f.write(f"- **総動画数**: {stats['total_videos']}\n")
            f.write(f"- **総フレーム数**: {stats['total_frames']:,}\n")
            f.write(f"- **平均可視率**: {stats['avg_visibility_ratio']:.1%}\n")
            f.write(f"- **平均確信度**: {stats['avg_score_overall']:.3f}\n")
            f.write(f"- **最高性能動画**: {stats['best_video']}\n")
            f.write(f"- **最低性能動画**: {stats['worst_video']}\n\n")
            
            f.write("## 🎯 推奨閾値設定\n\n")
            
            # 最適な閾値を決定
            best_threshold = None
            best_score = 0
            for threshold, rec in thresh_rec.items():
                # 有効フレーム率と良好動画数のバランス
                score = 0.7 * rec['avg_valid_ratio'] + 0.3 * (rec['videos_with_good_ratio'] / stats['total_videos'])
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            f.write("### 用途別推奨設定\n\n")
            f.write("#### 🏆 高品質データ生成用\n")
            f.write("- **閾値**: 0.8以上\n")
            f.write(f"- **期待有効率**: {thresh_rec[0.8]['avg_valid_ratio']:.1%}\n")
            f.write(f"- **良好動画数**: {thresh_rec[0.8]['videos_with_good_ratio']}/{stats['total_videos']}\n\n")
            
            f.write("#### ⚖️ バランス型\n")
            f.write(f"- **閾値**: {best_threshold}\n")
            f.write(f"- **期待有効率**: {thresh_rec[best_threshold]['avg_valid_ratio']:.1%}\n")
            f.write(f"- **良好動画数**: {thresh_rec[best_threshold]['videos_with_good_ratio']}/{stats['total_videos']}\n\n")
            
            f.write("#### 📈 大規模データ用\n")
            f.write("- **閾値**: 0.5以上\n")
            f.write(f"- **期待有効率**: {thresh_rec[0.5]['avg_valid_ratio']:.1%}\n")
            f.write(f"- **良好動画数**: {thresh_rec[0.5]['videos_with_good_ratio']}/{stats['total_videos']}\n\n")
            
            f.write("## 📋 動画別詳細統計\n\n")
            f.write("| 動画名 | 可視率 | 平均確信度 | 軌跡滑らかさ | フレーム数 |\n")
            f.write("|--------|--------|------------|--------------|------------|\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"| {row['video_name']} | {row['visibility_ratio']:.1%} | "
                       f"{row['overall_avg_score']:.3f} | {row['trajectory_smoothness']:.2f} | "
                       f"{row['total_frames']} |\n")
                       
        logger.info(f"分析レポート生成: {report_file}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Ball Tracker Batch Analysis Tool")
    parser.add_argument("--video_dir", required=True, help="動画ディレクトリのパス")
    parser.add_argument("--model_path", required=True, help="ball_tracker学習済みモデルのパス")
    parser.add_argument("--output_dir", help="結果出力ディレクトリ")
    parser.add_argument("--device", default="cuda", help="推論デバイス")
    parser.add_argument("--max_workers", type=int, default=1, help="並列処理数")
    parser.add_argument("--thresholds", nargs='+', type=float, 
                       default=[0.3, 0.5, 0.7, 0.8, 0.9], help="分析する確信度閾値リスト")
    parser.add_argument("--extensions", nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.wmv'], help="対象動画拡張子")
    
    args = parser.parse_args()
    
    # バッチ分析実行
    batch_analyzer = BatchAnalyzer(args.model_path, args.device)
    integrated_results = batch_analyzer.analyze_directory(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        confidence_thresholds=args.thresholds,
        max_workers=args.max_workers,
        video_extensions=args.extensions
    )
    
    # 結果サマリー表示
    stats = integrated_results['integrated_statistics']
    print("\n" + "="*80)
    print("📊 バッチ分析結果サマリー")
    print("="*80)
    print(f"分析動画数: {stats['total_videos']}")
    print(f"総フレーム数: {stats['total_frames']:,}")
    print(f"平均可視率: {stats['avg_visibility_ratio']:.1%}")
    print(f"平均確信度: {stats['avg_score_overall']:.3f}")
    print(f"最高性能: {stats['best_video']}")
    print(f"最低性能: {stats['worst_video']}")


if __name__ == "__main__":
    main() 