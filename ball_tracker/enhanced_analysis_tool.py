"""
Enhanced Ball Tracker Analysis Tool
3段階フィルタリングシステムによる強化分析ツール
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

from .analysis_tool import BallTrackerAnalyzer
from .local_classifier.inference import LocalClassifierInference, EnhancedTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAnalyzer(BallTrackerAnalyzer):
    """
    3段階フィルタリングシステムを統合した強化分析クラス
    """
    
    def __init__(self, 
                 ball_tracker_model_path: str,
                 local_classifier_model_path: str = None,
                 device: str = 'cuda'):
        """
        Args:
            ball_tracker_model_path (str): ball_trackerモデルパス
            local_classifier_model_path (str): ローカル分類器モデルパス
            device (str): デバイス
        """
        # Initialize base analyzer
        super().__init__(ball_tracker_model_path, device)
        
        # Initialize local classifier if provided
        self.local_classifier = None
        if local_classifier_model_path and os.path.exists(local_classifier_model_path):
            self.local_classifier = LocalClassifierInference(
                model_path=local_classifier_model_path,
                device=device,
                confidence_threshold=0.7
            )
            logger.info("Local classifier loaded successfully")
        else:
            logger.warning("Local classifier not loaded - using basic analysis only")
            
        # Enhanced tracking parameters
        self.enhanced_params = {
            'primary_threshold': 0.5,
            'local_threshold': 0.7,
            'max_jump_distance': 150.0
        }
        
    def analyze_video_enhanced(self, 
                              video_path: str, 
                              output_dir: str = None,
                              confidence_thresholds: List[float] = [0.3, 0.5, 0.7, 0.8, 0.9]) -> Dict:
        """
        3段階フィルタリングによる強化分析
        
        Args:
            video_path (str): 動画パス
            output_dir (str): 出力ディレクトリ
            confidence_thresholds (List[float]): 分析閾値
            
        Returns:
            Dict: 強化分析結果
        """
        if output_dir is None:
            output_dir = f"enhanced_analysis_results/{Path(video_path).stem}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"強化分析開始: {video_path}")
        
        # Basic analysis first
        basic_results = super().analyze_video(video_path, 
                                            os.path.join(output_dir, "basic"),
                                            confidence_thresholds)
        
        if self.local_classifier is None:
            logger.warning("Local classifier not available - returning basic results")
            return basic_results
            
        # Enhanced analysis with 3-stage filtering
        enhanced_results = self._perform_enhanced_analysis(video_path, output_dir, confidence_thresholds)
        
        # Combine results
        combined_results = {
            'basic_analysis': basic_results,
            'enhanced_analysis': enhanced_results,
            'comparison': self._compare_analyses(basic_results, enhanced_results)
        }
        
        # Generate enhanced visualizations
        self._generate_enhanced_visualizations(combined_results, output_dir)
        
        # Save combined results
        self._save_enhanced_results(combined_results, output_dir)
        
        logger.info(f"強化分析完了: {output_dir}")
        return combined_results
        
    def _perform_enhanced_analysis(self, video_path: str, output_dir: str, 
                                 confidence_thresholds: List[float]) -> Dict:
        """3段階フィルタリングによる分析実行"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize tracking components
        buffer = []
        frame_idx = 0
        self.tracker.refresh()
        
        # Results storage
        enhanced_results = {
            'detections': [],
            'filter_statistics': {
                'stage1_total': 0,
                'stage2_verified': 0,
                'stage3_final': 0,
                'filter_efficiency': {}
            },
            'tracking_history': [],
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height
            }
        }
        
        pbar = tqdm(total=total_frames, desc="強化分析中")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            buffer.append(frame.copy())
            
            if len(buffer) >= self.detector.frames_in:
                if len(buffer) > self.detector.frames_in:
                    buffer.pop(0)
                    
                # 3段階フィルタリング実行
                stage_results = self._apply_three_stage_filtering(buffer[-1], buffer)
                
                enhanced_results['detections'].append({
                    'frame_idx': frame_idx,
                    'stage_results': stage_results
                })
                
                # Statistics update
                stats = enhanced_results['filter_statistics']
                stats['stage1_total'] += stage_results['stage1_count']
                stats['stage2_verified'] += stage_results['stage2_count']
                stats['stage3_final'] += stage_results['stage3_count']
                
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        # Calculate filter efficiency
        stats = enhanced_results['filter_statistics']
        if stats['stage1_total'] > 0:
            stats['filter_efficiency'] = {
                'stage2_efficiency': stats['stage2_verified'] / stats['stage1_total'],
                'stage3_efficiency': stats['stage3_final'] / stats['stage1_total'] if stats['stage1_total'] > 0 else 0,
                'overall_efficiency': stats['stage3_final'] / stats['stage1_total'] if stats['stage1_total'] > 0 else 0
            }
            
        return enhanced_results
        
    def _apply_three_stage_filtering(self, current_frame: np.ndarray, frame_buffer: List[np.ndarray]) -> Dict:
        """3段階フィルタリングの適用"""
        
        # Stage 1: Primary ball_tracker detection
        detections = self.detector.process_frames(frame_buffer)
        stage1_filtered = [d for d in detections if d['score'] >= self.enhanced_params['primary_threshold']]
        
        stage_results = {
            'stage1_count': len(stage1_filtered),
            'stage2_count': 0,
            'stage3_count': 0,
            'stage1_detections': stage1_filtered,
            'stage2_verified': [],
            'stage3_final': None
        }
        
        if not stage1_filtered:
            return stage_results
            
        # Stage 2: Local classifier verification
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        verifications = self.local_classifier.verify_detections_batch(frame_rgb, stage1_filtered)
        
        stage2_verified = []
        for detection, verification in zip(stage1_filtered, verifications):
            if verification['verified']:
                detection['local_confidence'] = verification['local_confidence']
                stage2_verified.append(detection)
                
        stage_results['stage2_count'] = len(stage2_verified)
        stage_results['stage2_verified'] = stage2_verified
        
        if not stage2_verified:
            return stage_results
            
        # Stage 3: Trajectory consistency (simplified)
        # Select best detection based on combined scores
        best_detection = max(stage2_verified, 
                           key=lambda d: d['score'] * d.get('local_confidence', 1.0))
        
        stage_results['stage3_count'] = 1
        stage_results['stage3_final'] = best_detection
        
        return stage_results
        
    def _compare_analyses(self, basic_results: Dict, enhanced_results: Dict) -> Dict:
        """基本分析と強化分析の比較"""
        
        basic_stats = basic_results['statistics']['basic_stats']
        enhanced_stats = enhanced_results['filter_statistics']
        
        comparison = {
            'detection_improvement': {
                'basic_detections': basic_stats['total_detections'],
                'enhanced_final': enhanced_stats['stage3_final'],
                'filter_reduction_ratio': 1 - (enhanced_stats['stage3_final'] / enhanced_stats['stage1_total']) if enhanced_stats['stage1_total'] > 0 else 0
            },
            'quality_metrics': {
                'basic_avg_score': basic_stats['overall_avg_score'],
                'filter_efficiency': enhanced_stats.get('filter_efficiency', {}),
                'precision_improvement': 'TBD'  # 実際の評価データがあれば計算可能
            }
        }
        
        return comparison
        
    def _generate_enhanced_visualizations(self, combined_results: Dict, output_dir: str):
        """強化分析用の可視化生成"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        enhanced_stats = combined_results['enhanced_analysis']['filter_statistics']
        comparison = combined_results['comparison']
        
        # 1. フィルタリング段階別統計
        stages = ['Stage 1\n(Primary)', 'Stage 2\n(Local Classifier)', 'Stage 3\n(Final)']
        counts = [enhanced_stats['stage1_total'], enhanced_stats['stage2_verified'], enhanced_stats['stage3_final']]
        
        axes[0, 0].bar(stages, counts, color=['lightblue', 'orange', 'green'], alpha=0.7)
        axes[0, 0].set_title('3段階フィルタリング統計')
        axes[0, 0].set_ylabel('検出数')
        
        # Add percentages
        for i, (stage, count) in enumerate(zip(stages, counts)):
            if enhanced_stats['stage1_total'] > 0:
                percentage = count / enhanced_stats['stage1_total'] * 100
                axes[0, 0].text(i, count + max(counts)*0.01, f'{percentage:.1f}%', 
                               ha='center', va='bottom')
        
        # 2. フィルタ効率
        if 'filter_efficiency' in enhanced_stats:
            eff = enhanced_stats['filter_efficiency']
            efficiency_metrics = ['Stage 2\nEfficiency', 'Stage 3\nEfficiency', 'Overall\nEfficiency']
            efficiency_values = [eff.get('stage2_efficiency', 0), 
                               eff.get('stage3_efficiency', 0),
                               eff.get('overall_efficiency', 0)]
            
            axes[0, 1].bar(efficiency_metrics, efficiency_values, color=['skyblue', 'lightgreen', 'gold'], alpha=0.7)
            axes[0, 1].set_title('フィルタ効率')
            axes[0, 1].set_ylabel('効率率')
            axes[0, 1].set_ylim(0, 1)
            
            # Add percentage labels
            for i, val in enumerate(efficiency_values):
                axes[0, 1].text(i, val + 0.02, f'{val:.1%}', ha='center', va='bottom')
        
        # 3. 基本 vs 強化分析比較
        basic_detections = comparison['detection_improvement']['basic_detections']
        enhanced_final = comparison['detection_improvement']['enhanced_final']
        
        axes[0, 2].bar(['Basic\nAnalysis', 'Enhanced\nAnalysis'], 
                      [basic_detections, enhanced_final],
                      color=['lightcoral', 'lightgreen'], alpha=0.7)
        axes[0, 2].set_title('検出数比較')
        axes[0, 2].set_ylabel('総検出数')
        
        # 4-6. 追加の分析グラフ（フレーム時系列など）
        # この部分は実際のデータに基づいて実装
        for i in range(1, 2):
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'追加分析 {i+1}-{j+1}', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'追加分析 {i+1}-{j+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enhanced_analysis_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"強化分析可視化保存: {output_dir}/enhanced_analysis_overview.png")
        
    def _save_enhanced_results(self, combined_results: Dict, output_dir: str):
        """強化分析結果の保存"""
        
        # Main results
        results_file = os.path.join(output_dir, 'enhanced_analysis_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
        # Summary
        summary = {
            'basic_summary': combined_results['basic_analysis']['statistics']['basic_stats'],
            'enhanced_summary': combined_results['enhanced_analysis']['filter_statistics'],
            'comparison_summary': combined_results['comparison']
        }
        
        summary_file = os.path.join(output_dir, 'enhanced_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"強化分析結果保存: {results_file}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Enhanced Ball Tracker Analysis Tool")
    parser.add_argument("--video", required=True, help="分析対象動画のパス")
    parser.add_argument("--ball_tracker_model", required=True, help="ball_tracker学習済みモデルのパス")
    parser.add_argument("--local_classifier_model", help="ローカル分類器学習済みモデルのパス")
    parser.add_argument("--output_dir", help="結果出力ディレクトリ")
    parser.add_argument("--device", default="cuda", help="推論デバイス")
    parser.add_argument("--thresholds", nargs='+', type=float, 
                       default=[0.3, 0.5, 0.7, 0.8, 0.9], help="分析する確信度閾値リスト")
    
    args = parser.parse_args()
    
    # 強化分析実行
    analyzer = EnhancedAnalyzer(
        ball_tracker_model_path=args.ball_tracker_model,
        local_classifier_model_path=args.local_classifier_model,
        device=args.device
    )
    
    results = analyzer.analyze_video_enhanced(
        video_path=args.video,
        output_dir=args.output_dir,
        confidence_thresholds=args.thresholds
    )
    
    # 結果サマリー表示
    if 'enhanced_analysis' in results:
        enhanced_stats = results['enhanced_analysis']['filter_statistics']
        comparison = results['comparison']
        
        print("\n" + "="*70)
        print("🚀 強化分析結果サマリー")
        print("="*70)
        print(f"動画: {Path(args.video).name}")
        print(f"Stage 1 (Primary): {enhanced_stats['stage1_total']} 検出")
        print(f"Stage 2 (Local Classifier): {enhanced_stats['stage2_verified']} 検証済み")
        print(f"Stage 3 (Final): {enhanced_stats['stage3_final']} 最終検出")
        
        if 'filter_efficiency' in enhanced_stats:
            eff = enhanced_stats['filter_efficiency']
            print(f"\nフィルタ効率:")
            print(f"  Stage 2 効率: {eff.get('stage2_efficiency', 0):.1%}")
            print(f"  全体効率: {eff.get('overall_efficiency', 0):.1%}")
            
        detection_improvement = comparison['detection_improvement']
        reduction_ratio = detection_improvement['filter_reduction_ratio']
        print(f"\nノイズ除去率: {reduction_ratio:.1%}")
        
    print(f"\n詳細結果: {args.output_dir or f'enhanced_analysis_results/{Path(args.video).stem}'}")


if __name__ == "__main__":
    main() 