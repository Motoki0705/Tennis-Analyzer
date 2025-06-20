#!/usr/bin/env python3
"""
Ball Tracker Analysis Runner
動画分析の簡易実行スクリプト
"""

import os
import sys
from pathlib import Path

# 必要なライブラリの確認とインポート
try:
    # Try relative import first (when run as module)
    try:
        from .analysis_tool import BallTrackerAnalyzer
    except ImportError:
        # Fallback to absolute import (when run as script)
        from analysis_tool import BallTrackerAnalyzer
    
    import argparse
    import logging
except ImportError as e:
    print(f"❌ 必要なライブラリが見つかりません: {e}")
    print("以下をインストールしてください:")
    print("pip install matplotlib seaborn tqdm pandas")
    sys.exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_model_path(model_path: str) -> bool:
    """モデルファイルの存在確認"""
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        print("\n推奨: ball_trackerの学習済みモデルを以下に配置してください:")
        print("- checkpoints/ball_tracker/model.pth.tar")
        print("- または適切なパスを --model_path で指定")
        return False
    return True


def check_video_path(video_path: str) -> bool:
    """動画ファイルの存在確認"""
    if not os.path.exists(video_path):
        print(f"❌ 動画ファイルが見つかりません: {video_path}")
        return False
    
    # 対応形式の確認
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    if not any(video_path.lower().endswith(fmt) for fmt in supported_formats):
        print(f"⚠️  未対応の動画形式の可能性があります: {Path(video_path).suffix}")
        print(f"対応形式: {', '.join(supported_formats)}")
        
    return True


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="Ball Tracker動画分析ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用
  python run_analysis.py --video sample.mp4 --model_path model.pth.tar
  
  # 詳細設定
  python run_analysis.py \\
    --video sample.mp4 \\
    --model_path model.pth.tar \\
    --output_dir ./results \\
    --thresholds 0.5 0.7 0.8 0.9 \\
    --device cuda
        """
    )
    
    parser.add_argument("--video", required=True, help="分析対象動画のパス")
    parser.add_argument("--model_path", required=True, help="ball_tracker学習済みモデルのパス (.pth.tar)")
    parser.add_argument("--output_dir", help="結果出力ディレクトリ (デフォルト: analysis_results/{動画名})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推論デバイス")
    parser.add_argument("--thresholds", nargs='+', type=float, 
                       default=[0.3, 0.5, 0.7, 0.8, 0.9], 
                       help="分析する確信度閾値リスト")
    
    args = parser.parse_args()
    
    print("🎾 Ball Tracker Analysis Tool")
    print("=" * 50)
    
    # 事前チェック
    if not check_model_path(args.model_path):
        sys.exit(1)
        
    if not check_video_path(args.video):
        sys.exit(1)
    
    # 出力ディレクトリ設定
    if args.output_dir is None:
        video_name = Path(args.video).stem
        args.output_dir = f"analysis_results/{video_name}"
    
    print(f"📹 動画: {args.video}")
    print(f"🤖 モデル: {args.model_path}")
    print(f"📊 出力先: {args.output_dir}")
    print(f"⚙️  デバイス: {args.device}")
    print(f"🎯 閾値: {args.thresholds}")
    print()
    
    try:
        # 分析実行
        logger.info("分析開始...")
        analyzer = BallTrackerAnalyzer(args.model_path, args.device)
        results = analyzer.analyze_video(
            video_path=args.video,
            output_dir=args.output_dir,
            confidence_thresholds=args.thresholds
        )
        
        # 結果サマリー表示
        stats = results['statistics']['basic_stats']
        thresh_stats = results['statistics']['threshold_analysis']
        
        print("\n" + "=" * 60)
        print("📊 分析結果サマリー")
        print("=" * 60)
        
        print(f"動画: {Path(args.video).name}")
        print(f"総フレーム数: {stats['total_processed_frames']:,}")
        print(f"可視フレーム数: {stats['visible_frames']:,} ({stats['visibility_ratio']:.1%})")
        print(f"平均検出数/フレーム: {stats['avg_detections_per_frame']:.2f}")
        print(f"平均確信度: {stats['overall_avg_score']:.3f}")
        print(f"軌跡滑らかさ: {stats['trajectory_smoothness']:.2f}")
        
        print(f"\n📈 確信度閾値別統計:")
        print("-" * 40)
        for threshold in sorted(args.thresholds):
            stat = thresh_stats[threshold]
            print(f"閾値 {threshold:4.1f}: 有効率 {stat['valid_frame_ratio']:6.1%}, "
                  f"平均確信度 {stat['avg_score']:5.3f}")
        
        # 推奨設定の提案
        print(f"\n🎯 推奨設定:")
        print("-" * 40)
        
        # 最適閾値の計算
        best_threshold = None
        best_score = 0
        for threshold, stat in thresh_stats.items():
            if stat['valid_frame_ratio'] > 0:
                # 有効フレーム率と確信度のバランス
                score = 0.6 * stat['valid_frame_ratio'] + 0.4 * stat['avg_score']
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        print(f"高品質用 (閾値0.8): 有効率 {thresh_stats[0.8]['valid_frame_ratio']:.1%}")
        if best_threshold:
            print(f"バランス型 (閾値{best_threshold}): 有効率 {thresh_stats[best_threshold]['valid_frame_ratio']:.1%}")
        print(f"大規模用 (閾値0.5): 有効率 {thresh_stats[0.5]['valid_frame_ratio']:.1%}")
        
        print(f"\n📁 詳細結果:")
        print(f"  - 統計ファイル: {args.output_dir}/analysis_results.json")
        print(f"  - 可視化グラフ: {args.output_dir}/analysis_overview.png")
        print(f"  - 閾値分析: {args.output_dir}/threshold_analysis.png")
        
        print(f"\n✅ 分析完了!")
        
    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {e}")
        print(f"\n❌ エラー: {e}")
        print("\nトラブルシューティング:")
        print("1. モデルファイルが正しいか確認")
        print("2. 動画ファイルが破損していないか確認") 
        print("3. CUDA環境の場合、GPU メモリが十分か確認")
        print("4. --device cpu オプションを試してみる")
        sys.exit(1)


if __name__ == "__main__":
    main() 