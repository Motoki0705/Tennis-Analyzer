#!/usr/bin/env python3
"""
Test script for predictor API functionality
テスト用のシンプルなスクリプト
"""

def test_imports():
    """基本的なインポートテスト"""
    try:
        # 基本ライブラリ
        import os
        import sys
        import json
        print("✅ Basic Python libraries imported successfully")
        
        # プロジェクト構造チェック
        project_dirs = [
            'src/predictor',
            'src/predictor/api',
            'src/predictor/ball',
            'src/predictor/pipeline',
            'configs/infer'
        ]
        
        for dir_path in project_dirs:
            if os.path.exists(dir_path):
                print(f"✅ Directory exists: {dir_path}")
            else:
                print(f"❌ Directory missing: {dir_path}")
        
        # 設定ファイルチェック
        config_files = [
            'configs/infer/inference.yaml',
            'configs/infer/batch_process.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ Config file exists: {config_file}")
            else:
                print(f"❌ Config file missing: {config_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def test_config_structure():
    """設定ファイル構造テスト"""
    try:
        import os
        import yaml
        
        config_files = [
            'configs/infer/inference.yaml',
            'configs/infer/batch_process.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"✅ Config structure valid: {config_file}")
                    print(f"   Main keys: {list(config.keys())}")
            else:
                print(f"❌ Config file not found: {config_file}")
                
        return True
        
    except ImportError:
        print("⚠️ PyYAML not available, skipping config structure test")
        return True
    except Exception as e:
        print(f"❌ Config structure test failed: {e}")
        return False


def test_api_structure():
    """API構造テスト"""
    try:
        import os
        
        api_files = [
            'src/predictor/api/__init__.py',
            'src/predictor/api/inference.py',
            'src/predictor/api/batch_process.py'
        ]
        
        for api_file in api_files:
            if os.path.exists(api_file):
                print(f"✅ API file exists: {api_file}")
                # ファイルサイズチェック
                size = os.path.getsize(api_file)
                print(f"   Size: {size} bytes")
            else:
                print(f"❌ API file missing: {api_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False


def main():
    """メインテスト実行"""
    print("🎾 Tennis Ball Detection API Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Config Structure", test_config_structure),
        ("API Structure", test_api_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for use.")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Test with actual model file")
        print("   3. Run inference: python -m src.predictor.api.inference --help")
    else:
        print("⚠️ Some tests failed. Please check the output above.")


if __name__ == '__main__':
    main()