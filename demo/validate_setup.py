#!/usr/bin/env python
"""
VideoPredictor Demo Setup Validator
===================================

このスクリプトは、VideoPredictor Demo を実行する前に
必要なファイルや設定が正しく用意されているかを検証します。

使用方法:
    python demo/validate_setup.py
    python demo/validate_setup.py --config_path demo/my_config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import OmegaConf, DictConfig

# カレントディレクトリをPythonパスに追加
sys.path.append('.')


def check_file_exists(file_path: str, description: str) -> bool:
    """ファイルの存在を確認"""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}が見つかりません: {file_path}")
        return False


def check_directory_exists(dir_path: str, description: str) -> bool:
    """ディレクトリの存在を確認"""
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description}が見つかりません: {dir_path}")
        return False


def check_python_packages() -> bool:
    """必要なPythonパッケージの存在を確認"""
    required_packages = [
        "torch",
        "torchvision", 
        "transformers",
        "hydra-core",
        "omegaconf",
        "cv2",
        "tqdm",
        "numpy",
        "PIL"
    ]
    
    print("\n📦 必要なPythonパッケージの確認:")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
                print(f"✅ opencv-python: {cv2.__version__}")
            elif package == "PIL":
                import PIL
                print(f"✅ Pillow: {PIL.__version__}")
            else:
                module = __import__(package.replace("-", "_"))
                if hasattr(module, '__version__'):
                    print(f"✅ {package}: {module.__version__}")
                else:
                    print(f"✅ {package}: インストール済み")
        except ImportError:
            print(f"❌ {package}: インストールされていません")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 以下のパッケージをインストールしてください:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    
    return True


def check_config_file(config_path: str) -> Tuple[bool, DictConfig]:
    """設定ファイルの検証"""
    print(f"\n📋 設定ファイルの検証: {config_path}")
    
    try:
        cfg = OmegaConf.load(config_path)
        print("✅ 設定ファイルの読み込み成功")
        
        # 必要なセクションの確認
        required_sections = ["common", "ball", "court", "player", "pose", "processors", "predictors"]
        missing_sections = []
        
        for section in required_sections:
            if section in cfg:
                print(f"✅ {section}セクション: 存在")
            else:
                print(f"❌ {section}セクション: 不足")
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ 設定ファイルに必要なセクションが不足しています: {missing_sections}")
            return False, cfg
        
        return True, cfg
        
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みエラー: {e}")
        return False, None


def check_model_checkpoints(cfg: DictConfig) -> bool:
    """モデルチェックポイントの存在確認"""
    print(f"\n💾 モデルチェックポイントの確認:")
    
    all_exist = True
    
    # Lightning moduleのチェックポイント確認
    lightning_tasks = ["ball", "court", "player"]
    for task in lightning_tasks:
        if task in cfg:
            ckpt_path = cfg[task].get("ckpt_path")
            if ckpt_path:
                exists = check_file_exists(ckpt_path, f"{task}モデルのチェックポイント")
                all_exist = all_exist and exists
            else:
                print(f"❌ {task}モデルのckpt_pathが設定されていません")
                all_exist = False
    
    # Transformersモデルの確認
    if "pose" in cfg:
        pose_model = cfg.pose.get("pretrained_model_name_or_path")
        if pose_model:
            print(f"✅ poseモデル(Transformers): {pose_model}")
        else:
            print(f"❌ poseモデルのpretrained_model_name_or_pathが設定されていません")
            all_exist = False
    
    return all_exist


def check_predictors_config(cfg: DictConfig) -> bool:
    """予測器設定の確認"""
    print(f"\n🔧 予測器設定の確認:")
    
    all_valid = True
    
    # 必要な予測器の確認
    required_predictors = ["ball", "court", "pose", "streaming_overlayer"]
    for predictor in required_predictors:
        if predictor in cfg.predictors:
            print(f"✅ {predictor}予測器: 設定済み")
        else:
            print(f"❌ {predictor}予測器: 設定不足")
            all_valid = False
    
    # streaming_overlayerの詳細確認
    if "streaming_overlayer" in cfg.predictors:
        overlayer_cfg = cfg.predictors.streaming_overlayer
        
        # 必要なパラメータの確認
        required_params = ["intervals", "batch_sizes"]
        for param in required_params:
            if param in overlayer_cfg:
                print(f"✅ streaming_overlayer.{param}: {overlayer_cfg[param]}")
            else:
                print(f"❌ streaming_overlayer.{param}: 設定不足")
                all_valid = False
    
    return all_valid


def check_device_availability(cfg: DictConfig) -> bool:
    """デバイス利用可能性の確認"""
    print(f"\n🖥️ デバイス利用可能性の確認:")
    
    device = cfg.common.get("device", "cpu")
    
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"✅ CUDA利用可能: {torch.cuda.get_device_name(0)}")
            print(f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print(f"❌ CUDAが利用できません。CPUモードで実行してください。")
            return False
    else:
        print(f"✅ CPUモードで実行")
        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="VideoPredictor Demo セットアップ検証ツール"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/infer/infer.yaml",
        help="設定ファイルのパス"
    )
    
    args = parser.parse_args()
    
    print("🔍 VideoPredictor Demo セットアップ検証を開始します...")
    print("=" * 60)
    
    # 検証結果を記録
    results = []
    
    # 1. Pythonパッケージの確認
    results.append(check_python_packages())
    
    # 2. 設定ファイルの確認
    config_valid, cfg = check_config_file(args.config_path)
    results.append(config_valid)
    
    if not config_valid:
        print("\n❌ 設定ファイルの検証に失敗したため、以降の検証をスキップします")
        sys.exit(1)
    
    # 3. モデルチェックポイントの確認
    results.append(check_model_checkpoints(cfg))
    
    # 4. 予測器設定の確認
    results.append(check_predictors_config(cfg))
    
    # 5. デバイス利用可能性の確認
    results.append(check_device_availability(cfg))
    
    # 6. 重要なディレクトリの確認
    print(f"\n📁 重要なディレクトリの確認:")
    dir_results = []
    dir_results.append(check_directory_exists("src/multi/streaming_overlayer", "StreamingOverlayerディレクトリ"))
    dir_results.append(check_directory_exists("src/predictors", "Predictorsディレクトリ"))
    
    results.extend(dir_results)
    
    # 結果の集計
    print("\n" + "=" * 60)
    print("📊 検証結果の集計:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ すべての検証に合格しました！ ({passed}/{total})")
        print("🚀 VideoPredictor Demo の実行準備が整いました")
        
        print("\n🔥 実行例:")
        print("python demo/video_predictor_demo.py \\")
        print("    --input_path datasets/test/input.mp4 \\")
        print("    --output_path outputs/demo_output.mp4")
        
        sys.exit(0)
    else:
        print(f"❌ 検証に失敗しました ({passed}/{total})")
        print("🔧 上記の問題を解決してから再実行してください")
        sys.exit(1)


if __name__ == "__main__":
    main() 