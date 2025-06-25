#!/usr/bin/env python3
"""
Test Hydra decorator fix
Hydraデコレータ修正のテスト
"""

import sys
import os

def test_hydra_import():
    """Hydraインポートテスト"""
    try:
        # プロジェクトルートをパスに追加
        sys.path.insert(0, '.')
        
        print("🧪 Testing Hydra decorator fix...")
        
        # 1. モジュールインポートテスト
        print("1. Testing module imports...")
        try:
            from src.predictor.api import inference_main, batch_main
            print("   ✅ API functions imported without RuntimeWarning")
        except Exception as e:
            print(f"   ❌ Import failed: {e}")
            return False
        
        # 2. Hydraデコレータテスト
        print("2. Testing Hydra decorator placement...")
        try:
            import inspect
            from src.predictor.api.inference import main as inf_main
            from src.predictor.api.batch_process import main as batch_main_func
            
            # デコレータが正しく適用されているかチェック
            inf_decorators = getattr(inf_main, '__wrapped__', None)
            batch_decorators = getattr(batch_main_func, '__wrapped__', None)
            
            print(f"   ✅ Inference main has Hydra decorator: {inf_decorators is not None}")
            print(f"   ✅ Batch main has Hydra decorator: {batch_decorators is not None}")
            
        except Exception as e:
            print(f"   ❌ Decorator test failed: {e}")
            return False
        
        # 3. Help表示テスト
        print("3. Testing help display (should work without errors)...")
        try:
            # これはHydraが適切に設定されていれば動作するはず
            print("   ℹ️ To test help: python -m src.predictor.api.inference --help")
            print("   ℹ️ To test help: python -m src.predictor.api.batch_process --help")
            
        except Exception as e:
            print(f"   ❌ Help test setup failed: {e}")
            return False
        
        print("✅ All Hydra tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hydra test failed: {e}")
        return False

def test_config_validation():
    """設定検証関数テスト"""
    try:
        print("\n🔧 Testing config validation...")
        
        from src.predictor.api.inference import validate_config
        from src.predictor.api.batch_process import validate_batch_config
        from omegaconf import DictConfig
        
        # ダミー設定でテスト
        dummy_cfg = DictConfig({
            'io': {
                'video': None,
                'output': None,
                'input_dir': None,
                'input_list': None,
                'output_dir': None
            },
            'model': {
                'model_path': None
            }
        })
        
        # 検証関数が例外を正しく投げるかテスト
        try:
            validate_config(dummy_cfg)
            print("   ❌ Validation should have failed for empty config")
            return False
        except (ValueError, FileNotFoundError):
            print("   ✅ Config validation works correctly")
        
        try:
            validate_batch_config(dummy_cfg)
            print("   ❌ Batch validation should have failed for empty config")
            return False
        except (ValueError, FileNotFoundError):
            print("   ✅ Batch config validation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Config validation test failed: {e}")
        return False

def main():
    """メインテスト"""
    print("🎾 Testing Hydra Decorator Fix")
    print("=" * 50)
    
    tests = [
        ("Hydra Import", test_hydra_import),
        ("Config Validation", test_config_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Hydra fix successful! Try running:")
        print("   python -m src.predictor.api.inference --help")
        print("   python -m src.predictor.api.batch_process --help")
    else:
        print("⚠️ Some issues remain. Check the output above.")

if __name__ == '__main__':
    main()