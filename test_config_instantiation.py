#!/usr/bin/env python3
"""
VideoSwinTransformer設定ファイルの動作確認スクリプト
"""
import os
import sys
from omegaconf import DictConfig
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def test_config_instantiation(config_name: str):
    """設定ファイルの読み込みとモジュール初期化テスト"""
    print(f"\n=== {config_name} 設定テスト ===")
    
    try:
        # Hydraの初期化
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path="configs/train/ball"):
            cfg = compose(config_name=config_name)
            
        print(f"✓ 設定ファイル読み込み成功: {config_name}")
        print(f"  - モデル名: {cfg.litmodule.meta.name}")
        print(f"  - フレーム数: {cfg.litdatamodule.T}")
        print(f"  - バッチサイズ: {cfg.litdatamodule.batch_size}")
        print(f"  - 画像サイズ: {cfg.litdatamodule.input_size}")
        
        # モジュールの初期化テスト
        try:
            # LightningModule の初期化
            lit_module_cfg = cfg.litmodule.module
            lit_module_class = hydra.utils.get_class(lit_module_cfg._target_)
            
            # 設定から _target_ を除いてパラメータを取得
            module_params = {k: v for k, v in lit_module_cfg.items() if k != '_target_'}
            
            # n_framesパラメータの解決
            if 'n_frames' in module_params:
                module_params['n_frames'] = cfg.litdatamodule.T
            if 'max_epochs' in module_params:
                module_params['max_epochs'] = 50  # デフォルト値
                
            print(f"  - モデルパラメータ: {module_params}")
            
            # モジュールの初期化（実際にはインスタンス化しない、クラスの存在確認のみ）
            print(f"  - LightningModule クラス: {lit_module_class}")
            print("✓ LightningModule 設定確認完了")
            
            # データモジュールの設定確認
            data_module_cfg = cfg.litdatamodule
            data_module_class = hydra.utils.get_class(data_module_cfg._target_)
            print(f"  - DataModule クラス: {data_module_class}")
            print("✓ DataModule 設定確認完了")
            
            return True
            
        except Exception as e:
            print(f"✗ モジュール初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ 設定ファイル読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("VideoSwinTransformer設定ファイルテスト開始")
    
    # テストする設定ファイル
    configs_to_test = [
        "seq_lite_transformer.yaml",
        "seq_lite_transformer_small.yaml"
    ]
    
    results = {}
    for config_name in configs_to_test:
        try:
            result = test_config_instantiation(config_name)
            results[config_name] = result
        except Exception as e:
            print(f"✗ {config_name} テスト中に予期しないエラー: {e}")
            results[config_name] = False
    
    # 結果の要約
    print("\n" + "="*50)
    print("テスト結果サマリー:")
    success_count = 0
    for config_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"  {config_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n全{len(configs_to_test)}設定中 {success_count}設定が正常に動作しています。")
    
    if success_count == len(configs_to_test):
        print("🎉 全ての設定ファイルが正常です！")
        return True
    else:
        print("⚠️  一部の設定に問題があります。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 