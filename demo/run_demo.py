#!/usr/bin/env python3
"""
Tennis Analysis Demo Launcher
============================

🎾 テニス解析デモ起動スクリプト

このスクリプトは、利用可能なデモアプリケーションを確認し、
適切なデモを起動するためのランチャーです。

Available Demos:
- tennis_analysis_app.py: 統合型フル機能アプリ
- simple_demo.py: シンプル軽量版
- Legacy demos: ball.py, court.py, player.py等

Usage:
    python demo/run_demo.py [demo_name] [options]

Examples:
    python demo/run_demo.py                    # インタラクティブ選択
    python demo/run_demo.py simple             # シンプルデモ
    python demo/run_demo.py full               # フル機能デモ
    python demo/run_demo.py ball               # ボール検出のみ
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 利用可能なデモ一覧
AVAILABLE_DEMOS = {
    "simple": {
        "file": "simple_demo.py",
        "name": "Simple Tennis Ball Detector",
        "description": "軽量で使いやすいボール検出デモ",
        "features": ["🚀 ワンクリック解析", "📱 モバイル対応", "⚡ 高速処理"],
        "port": 7861
    },
    "full": {
        "file": "tennis_analysis_app.py", 
        "name": "Tennis Analysis System",
        "description": "統合型フル機能テニス解析システム",
        "features": ["🎯 ボール検出", "📁 バッチ処理", "📊 統計分析", "⚙️ 詳細設定"],
        "port": 7860
    },
    "ball": {
        "file": "ball.py",
        "name": "Ball Detection Demo",
        "description": "レガシー版ボール検出デモ",
        "features": ["🎾 ボール検出", "📈 軌跡追跡", "🔧 異常値除去"],
        "port": 7862
    },
    "court": {
        "file": "court.py",
        "name": "Court Detection Demo", 
        "description": "コート検出・キーポイント解析",
        "features": ["🏟️ コート認識", "📍 キーポイント検出", "🎨 ヒートマップ"],
        "port": 7863
    },
    "player": {
        "file": "player.py",
        "name": "Player Detection Demo",
        "description": "プレーヤー検出・姿勢推定",
        "features": ["👥 プレーヤー検出", "🤸 姿勢推定", "📊 動作解析"],
        "port": 7864
    }
}

def check_dependencies() -> Dict[str, bool]:
    """依存関係チェック"""
    dependencies = {}
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        dependencies["torch"] = False
    
    try:
        import gradio
        dependencies["gradio"] = True
    except ImportError:
        dependencies["gradio"] = False
    
    try:
        import cv2
        dependencies["cv2"] = True
    except ImportError:
        dependencies["cv2"] = False
    
    try:
        from src.predictor import VideoPipeline
        dependencies["predictor"] = True
    except ImportError:
        dependencies["predictor"] = False
    
    return dependencies

def check_model_files() -> Dict[str, List[str]]:
    """モデルファイル存在確認"""
    model_info = {"ball": [], "court": [], "player": []}
    
    # checkpointsディレクトリ検索
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for category in model_info.keys():
            category_dir = checkpoints_dir / category
            if category_dir.exists():
                for ext in ["*.ckpt", "*.pth"]:
                    model_info[category].extend([str(p) for p in category_dir.glob(ext)])
    
    return model_info

def print_system_status():
    """システム状態表示"""
    print("🎾" + "="*50)
    print("   Tennis Analysis Demo System")
    print("="*52)
    print()
    
    # 依存関係チェック
    print("📦 Dependencies:")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    print()
    
    # GPU情報
    try:
        import torch
        print("💻 System Info:")
        print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
    except:
        pass
    print()
    
    # モデルファイル確認
    print("🤖 Available Models:")
    models = check_model_files()
    for category, files in models.items():
        count = len(files)
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {category.capitalize()}: {count} files")
    print()

def print_demo_list():
    """利用可能デモ一覧表示"""
    print("🎮 Available Demos:")
    print()
    
    for key, demo in AVAILABLE_DEMOS.items():
        file_exists = os.path.exists(Path(__file__).parent / demo["file"])
        status = "✅" if file_exists else "❌"
        
        print(f"{status} [{key}] {demo['name']}")
        print(f"    📝 {demo['description']}")
        print(f"    🌐 Port: {demo['port']}")
        print(f"    ✨ Features: {', '.join(demo['features'])}")
        print()

def interactive_selection() -> Optional[str]:
    """インタラクティブデモ選択"""
    print("🎯 Select a demo to launch:")
    print()
    
    valid_options = []
    for key, demo in AVAILABLE_DEMOS.items():
        if os.path.exists(Path(__file__).parent / demo["file"]):
            valid_options.append(key)
            print(f"  {len(valid_options)}. [{key}] {demo['name']}")
    
    if not valid_options:
        print("❌ No demo files found!")
        return None
    
    print(f"  {len(valid_options) + 1}. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-{}): ".format(len(valid_options) + 1))
            choice_num = int(choice)
            
            if choice_num == len(valid_options) + 1:
                return None
            elif 1 <= choice_num <= len(valid_options):
                return valid_options[choice_num - 1]
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(valid_options) + 1}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None

def launch_demo(demo_key: str, share: bool = False, debug: bool = False):
    """デモ起動"""
    if demo_key not in AVAILABLE_DEMOS:
        print(f"❌ Unknown demo: {demo_key}")
        return False
    
    demo = AVAILABLE_DEMOS[demo_key]
    demo_file = Path(__file__).parent / demo["file"]
    
    if not demo_file.exists():
        print(f"❌ Demo file not found: {demo_file}")
        return False
    
    print(f"🚀 Launching {demo['name']}...")
    print(f"📁 File: {demo['file']}")
    print(f"🌐 URL: http://localhost:{demo['port']}")
    print()
    print("💡 Press Ctrl+C to stop the demo")
    print("-" * 50)
    
    try:
        # 環境変数設定
        env = os.environ.copy()
        if share:
            env["GRADIO_SHARE"] = "1"
        if debug:
            env["GRADIO_DEBUG"] = "1"
        
        # デモ実行
        subprocess.run([
            sys.executable, str(demo_file)
        ], env=env, cwd=project_root)
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        return False

def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="Tennis Analysis Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "demo", nargs="?",
        choices=list(AVAILABLE_DEMOS.keys()),
        help="Demo to launch (interactive selection if not specified)"
    )
    
    parser.add_argument(
        "--list", action="store_true",
        help="List available demos and exit"
    )
    
    parser.add_argument(
        "--status", action="store_true", 
        help="Show system status and exit"
    )
    
    parser.add_argument(
        "--share", action="store_true",
        help="Create shareable Gradio link"
    )
    
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # システム状態表示
    if args.status or args.list or not args.demo:
        print_system_status()
    
    # デモ一覧表示
    if args.list:
        print_demo_list()
        return
    
    if args.status:
        return
    
    # デモ選択・起動
    demo_key = args.demo
    if not demo_key:
        demo_key = interactive_selection()
    
    if demo_key:
        success = launch_demo(demo_key, args.share, args.debug)
        if not success:
            sys.exit(1)
    else:
        print("👋 No demo selected. Goodbye!")

if __name__ == "__main__":
    main() 