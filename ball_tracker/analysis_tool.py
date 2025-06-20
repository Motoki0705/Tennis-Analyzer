"""
Ball Tracker Analysis Tool
動画を入力としてball_trackerの結果を分析・可視化するツール
"""

import argparse
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

# ball_tracker modules
from .video_demo import SimpleDetector, load_simple_config
from .online import OnlineTracker
from omegaconf import OmegaConf

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallTrackerAnalyzer:
    """Ball Tracker結果の分析・可視化クラス"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path (str): ball_trackerの学習済みモデルパス
            device (str): 推論デバイス
        """
        self.model_path = model_path
        self.device = device
        
        # ball_tracker components の初期化
        cfg = load_simple_config()
        cfg.detector.model_path = model_path
        
        self.detector = SimpleDetector(cfg, device)
        self.tracker = OnlineTracker(cfg)
        
        # 分析結果保存用
        self.results = {
            'detections': [],      # フレーム毎の検出結果
            'tracking': [],        # トラッキング結果
            'statistics': {},      # 統計情報
            'video_info': {}       # 動画情報
        }
        
    def analyze_video(self, video_path: str, output_dir: str = None, 
                     confidence_thresholds: List[float] = [0.3, 0.5, 0.7, 0.8, 0.9]) -> Dict:
        """
        動画を分析してball_trackerの結果を評価
        
        Args:
            video_path (str): 分析対象の動画パス
            output_dir (str): 結果保存ディレクトリ
            confidence_thresholds (List[float]): 分析する確信度閾値リスト
            
        Returns:
            Dict: 分析結果
        """
        if output_dir is None:
            output_dir = f"analysis_results/{Path(video_path).stem}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"動画分析開始: {video_path}")
        
        # 動画情報取得
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.results['video_info'] = {
            'path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration_sec': total_frames / fps
        }
        
        logger.info(f"動画情報: {width}x{height}, {fps:.2f}fps, {total_frames}フレーム")
        
        # フレーム毎の処理
        buffer = []
        frame_idx = 0
        self.tracker.refresh()
        
        pbar = tqdm(total=total_frames, desc="フレーム処理中")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            buffer.append(frame.copy())
            
            # 十分なフレームが溜まったら検出実行
            if len(buffer) >= self.detector.frames_in:
                if len(buffer) > self.detector.frames_in:
                    buffer.pop(0)
                    
                # 検出実行
                detections = self.detector.process_frames(buffer)
                
                # トラッキング更新
                tracking_result = self.tracker.update(detections)
                
                # 結果保存
                frame_result = {
                    'frame_idx': frame_idx,
                    'detections': detections,
                    'tracking': tracking_result,
                    'num_detections': len(detections)
                }
                
                self.results['detections'].append(frame_result)
                
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        # 統計分析実行
        self._compute_statistics(confidence_thresholds)
        
        # 結果保存
        self._save_results(output_dir)
        
        # 可視化生成
        self._generate_visualizations(output_dir, confidence_thresholds)
        
        logger.info(f"分析完了。結果を保存: {output_dir}")
        return self.results
        
    def _compute_statistics(self, confidence_thresholds: List[float]):
        """統計情報の計算"""
        
        # 基本統計
        all_scores = []
        all_detections_per_frame = []
        visible_frames = 0
        total_processed_frames = len(self.results['detections'])
        
        for frame_result in self.results['detections']:
            detections = frame_result['detections']
            tracking = frame_result['tracking']
            
            # 検出数統計
            all_detections_per_frame.append(len(detections))
            
            # スコア統計
            for det in detections:
                all_scores.append(det['score'])
                
            # 可視性統計
            if tracking.get('visi', False):
                visible_frames += 1
                
        # 確信度別統計
        threshold_stats = {}
        for threshold in confidence_thresholds:
            high_conf_detections = [s for s in all_scores if s >= threshold]
            valid_frames = 0
            
            for frame_result in self.results['detections']:
                frame_max_score = max([det['score'] for det in frame_result['detections']], default=0)
                if frame_max_score >= threshold:
                    valid_frames += 1
                    
            threshold_stats[threshold] = {
                'total_detections': len(high_conf_detections),
                'valid_frames': valid_frames,
                'valid_frame_ratio': valid_frames / total_processed_frames if total_processed_frames > 0 else 0,
                'avg_score': np.mean(high_conf_detections) if high_conf_detections else 0,
                'std_score': np.std(high_conf_detections) if high_conf_detections else 0
            }
            
        # 軌跡一貫性分析
        tracking_positions = []
        tracking_scores = []
        for frame_result in self.results['detections']:
            tracking = frame_result['tracking']
            if tracking.get('visi', False):
                tracking_positions.append([tracking['x'], tracking['y']])
                tracking_scores.append(tracking['score'])
                
        # 軌跡の滑らかさ（速度変化）
        trajectory_smoothness = 0
        if len(tracking_positions) > 2:
            velocities = []
            for i in range(1, len(tracking_positions)):
                pos_curr = np.array(tracking_positions[i])
                pos_prev = np.array(tracking_positions[i-1])
                velocity = np.linalg.norm(pos_curr - pos_prev)
                velocities.append(velocity)
                
            if len(velocities) > 1:
                # 速度変化の標準偏差（小さいほど滑らか）
                trajectory_smoothness = np.std(velocities)
                
        self.results['statistics'] = {
            'basic_stats': {
                'total_processed_frames': total_processed_frames,
                'visible_frames': visible_frames,
                'visibility_ratio': visible_frames / total_processed_frames if total_processed_frames > 0 else 0,
                'avg_detections_per_frame': np.mean(all_detections_per_frame),
                'total_detections': len(all_scores),
                'overall_avg_score': np.mean(all_scores) if all_scores else 0,
                'overall_std_score': np.std(all_scores) if all_scores else 0,
                'trajectory_smoothness': trajectory_smoothness,
                'tracking_length': len(tracking_positions)
            },
            'threshold_analysis': threshold_stats,
            'score_distribution': {
                'min': float(np.min(all_scores)) if all_scores else 0,
                'max': float(np.max(all_scores)) if all_scores else 0,
                'median': float(np.median(all_scores)) if all_scores else 0,
                'q25': float(np.percentile(all_scores, 25)) if all_scores else 0,
                'q75': float(np.percentile(all_scores, 75)) if all_scores else 0
            }
        }
        
    def _save_results(self, output_dir: str):
        """結果をJSONファイルに保存"""
        
        # メイン結果ファイル
        results_file = os.path.join(output_dir, 'analysis_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        # 統計サマリー
        summary_file = os.path.join(output_dir, 'statistics_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results['statistics'], f, indent=2, ensure_ascii=False)
            
        logger.info(f"結果ファイル保存: {results_file}")
        logger.info(f"統計サマリー保存: {summary_file}")
        
    def _generate_visualizations(self, output_dir: str, confidence_thresholds: List[float]):
        """可視化グラフの生成"""
        
        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 確信度分布ヒストグラム
        all_scores = []
        for frame_result in self.results['detections']:
            for det in frame_result['detections']:
                all_scores.append(det['score'])
                
        if all_scores:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 3, 1)
            plt.hist(all_scores, bins=50, alpha=0.7, edgecolor='black')
            for threshold in confidence_thresholds:
                plt.axvline(threshold, color='red', linestyle='--', alpha=0.7, label=f'閾値 {threshold}')
            plt.xlabel('確信度')
            plt.ylabel('頻度')
            plt.title('確信度分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. フレーム毎の検出数
            frame_indices = [fr['frame_idx'] for fr in self.results['detections']]
            detection_counts = [fr['num_detections'] for fr in self.results['detections']]
            
            plt.subplot(2, 3, 2)
            plt.plot(frame_indices, detection_counts, alpha=0.7)
            plt.xlabel('フレーム番号')
            plt.ylabel('検出数')
            plt.title('フレーム毎の検出数')
            plt.grid(True, alpha=0.3)
            
            # 3. 軌跡プロット
            tracking_positions = []
            for frame_result in self.results['detections']:
                tracking = frame_result['tracking']
                if tracking.get('visi', False):
                    tracking_positions.append([tracking['x'], tracking['y']])
                    
            if tracking_positions:
                positions = np.array(tracking_positions)
                plt.subplot(2, 3, 3)
                plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
                plt.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)), 
                           cmap='viridis', s=20, alpha=0.8)
                plt.xlabel('X座標')
                plt.ylabel('Y座標')
                plt.title('ボール軌跡')
                plt.colorbar(label='時間')
                plt.grid(True, alpha=0.3)
                
            # 4. 確信度閾値別統計
            thresholds = list(confidence_thresholds)
            valid_ratios = [self.results['statistics']['threshold_analysis'][th]['valid_frame_ratio'] 
                           for th in thresholds]
            
            plt.subplot(2, 3, 4)
            plt.bar(range(len(thresholds)), valid_ratios, alpha=0.7)
            plt.xlabel('確信度閾値')
            plt.ylabel('有効フレーム率')
            plt.title('閾値別有効フレーム率')
            plt.xticks(range(len(thresholds)), [f'{th:.1f}' for th in thresholds])
            plt.grid(True, alpha=0.3)
            
            # 5. 時系列での確信度変化
            frame_indices = []
            max_scores = []
            for frame_result in self.results['detections']:
                frame_indices.append(frame_result['frame_idx'])
                if frame_result['detections']:
                    max_score = max([det['score'] for det in frame_result['detections']])
                    max_scores.append(max_score)
                else:
                    max_scores.append(0)
                    
            plt.subplot(2, 3, 5)
            plt.plot(frame_indices, max_scores, alpha=0.7)
            for threshold in confidence_thresholds:
                plt.axhline(threshold, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('フレーム番号')
            plt.ylabel('最大確信度')
            plt.title('フレーム毎の最大確信度')
            plt.grid(True, alpha=0.3)
            
            # 6. 統計サマリーテーブル
            plt.subplot(2, 3, 6)
            plt.axis('off')
            
            stats = self.results['statistics']['basic_stats']
            summary_text = f"""
統計サマリー:
• 総フレーム数: {stats['total_processed_frames']}
• 可視フレーム数: {stats['visible_frames']}
• 可視率: {stats['visibility_ratio']:.3f}
• 平均検出数/フレーム: {stats['avg_detections_per_frame']:.2f}
• 総検出数: {stats['total_detections']}
• 平均確信度: {stats['overall_avg_score']:.3f}
• 軌跡滑らかさ: {stats['trajectory_smoothness']:.2f}
            """
            plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'analysis_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # 確信度閾値別詳細分析
        self._generate_threshold_analysis_chart(output_dir, confidence_thresholds)
        
        logger.info(f"可視化ファイル保存: {output_dir}/analysis_overview.png")
        
    def _generate_threshold_analysis_chart(self, output_dir: str, confidence_thresholds: List[float]):
        """確信度閾値別の詳細分析チャート"""
        
        plt.figure(figsize=(15, 10))
        
        threshold_stats = self.results['statistics']['threshold_analysis']
        
        metrics = ['valid_frame_ratio', 'avg_score', 'total_detections']
        metric_names = ['有効フレーム率', '平均確信度', '総検出数']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            plt.subplot(2, 3, i+1)
            
            thresholds = list(confidence_thresholds)
            values = [threshold_stats[th][metric] for th in thresholds]
            
            if metric == 'total_detections':
                # 検出数は正規化
                max_val = max(values) if values else 1
                values = [v / max_val for v in values]
                name += ' (正規化)'
                
            plt.plot(thresholds, values, 'o-', linewidth=2, markersize=6)
            plt.xlabel('確信度閾値')
            plt.ylabel(name)
            plt.title(f'閾値別{name}')
            plt.grid(True, alpha=0.3)
            
        # 推奨閾値の提案
        plt.subplot(2, 3, 4)
        plt.axis('off')
        
        # 最適閾値の計算（有効フレーム率と確信度のバランス）
        best_threshold = None
        best_score = 0
        
        for threshold in confidence_thresholds:
            stats = threshold_stats[threshold]
            # 有効フレーム率と平均確信度の加重平均
            score = 0.6 * stats['valid_frame_ratio'] + 0.4 * (stats['avg_score'] if stats['avg_score'] > 0 else 0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        recommendation_text = f"""
推奨設定:

高品質データ用:
• 閾値: 0.8以上
• フレーム率: {threshold_stats[0.8]['valid_frame_ratio']:.3f}

バランス型:
• 閾値: {best_threshold}
• フレーム率: {threshold_stats[best_threshold]['valid_frame_ratio']:.3f}
• 平均確信度: {threshold_stats[best_threshold]['avg_score']:.3f}

大規模データ用:
• 閾値: 0.5以上  
• フレーム率: {threshold_stats[0.5]['valid_frame_ratio']:.3f}
        """
        
        plt.text(0.1, 0.5, recommendation_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"閾値分析チャート保存: {output_dir}/threshold_analysis.png")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Ball Tracker Analysis Tool")
    parser.add_argument("--video", required=True, help="分析対象動画のパス")
    parser.add_argument("--model_path", required=True, help="ball_tracker学習済みモデルのパス")
    parser.add_argument("--output_dir", help="結果出力ディレクトリ")
    parser.add_argument("--device", default="cuda", help="推論デバイス")
    parser.add_argument("--thresholds", nargs='+', type=float, 
                       default=[0.3, 0.5, 0.7, 0.8, 0.9], help="分析する確信度閾値リスト")
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = BallTrackerAnalyzer(args.model_path, args.device)
    results = analyzer.analyze_video(
        video_path=args.video,
        output_dir=args.output_dir,
        confidence_thresholds=args.thresholds
    )
    
    # 結果サマリー表示
    stats = results['statistics']['basic_stats']
    print("\n" + "="*60)
    print("📊 分析結果サマリー")
    print("="*60)
    print(f"動画: {Path(args.video).name}")
    print(f"総フレーム数: {stats['total_processed_frames']}")
    print(f"可視フレーム数: {stats['visible_frames']} ({stats['visibility_ratio']:.1%})")
    print(f"平均検出数/フレーム: {stats['avg_detections_per_frame']:.2f}")
    print(f"平均確信度: {stats['overall_avg_score']:.3f}")
    print(f"軌跡滑らかさ: {stats['trajectory_smoothness']:.2f}")
    
    print("\n確信度閾値別統計:")
    for threshold, stat in results['statistics']['threshold_analysis'].items():
        print(f"  閾値 {threshold}: 有効率 {stat['valid_frame_ratio']:.1%}, "
              f"平均確信度 {stat['avg_score']:.3f}")
    
    print(f"\n詳細結果: {args.output_dir or f'analysis_results/{Path(args.video).stem}'}")


if __name__ == "__main__":
    main() 