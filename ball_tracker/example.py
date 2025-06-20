#!/usr/bin/env python3
"""
ボールトラッカー使用例
"""

import argparse
from ball_tracker import BallTracker

def main():
    parser = argparse.ArgumentParser(description="Ball Tracker Example")
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--model", required=True, help="モデルファイルパス")
    parser.add_argument("--output", help="出力動画パス（オプション）")
    args = parser.parse_args()

    # トラッカー初期化
    print("🏀 Ball Tracker を初期化中...")
    tracker = BallTracker(model_path=args.model)

    # 動画処理
    print("🎬 動画を処理中...")
    results = tracker.track_video(
        video_path=args.video,
        output_path=args.output,
        visualize=True
    )

    # 統計表示
    total_frames = len(results)
    visible_frames = sum(1 for r in results if r['visible'])
    
    print(f"\n📊 処理結果:")
    print(f"  総フレーム数: {total_frames}")
    print(f"  ボール検出フレーム数: {visible_frames}")
    print(f"  検出率: {visible_frames/total_frames*100:.1f}%")
    
    if args.output:
        print(f"  出力動画: {args.output}")

if __name__ == "__main__":
    main()
