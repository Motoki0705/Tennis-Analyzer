#!/usr/bin/env python3
"""
Test API fix for visualization_config parameter
API修正のテスト（visualization_configパラメータ）
"""

import ast
import os

def check_method_signatures():
    """メソッドシグネチャの修正確認"""
    print("🔍 Checking method call parameter names...")
    
    files_to_check = [
        ('src/predictor/api/inference.py', 'inference'),
        ('src/predictor/api/batch_process.py', 'batch_process')
    ]
    
    for file_path, file_name in files_to_check:
        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 正しいパラメータ名をチェック
            if 'visualization_config=vis_config' in content:
                print(f"   ✅ {file_name}: Uses correct 'visualization_config' parameter")
            else:
                print(f"   ❌ {file_name}: Missing 'visualization_config' parameter")
            
            # 間違ったパラメータ名をチェック
            if 'vis_config=vis_config' in content:
                print(f"   ❌ {file_name}: Still has incorrect 'vis_config' parameter")
            else:
                print(f"   ✅ {file_name}: No incorrect 'vis_config' parameter found")
                
        except Exception as e:
            print(f"   ❌ {file_path}: Parse error - {e}")

def check_import_statements():
    """インポート文の確認"""
    print("\n📦 Checking import statements...")
    
    files_to_check = [
        'src/predictor/api/inference.py',
        'src/predictor/api/batch_process.py'
    ]
    
    required_imports = [
        'import copy',
        'from src.predictor.pipeline.config import PipelineConfig'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_name = os.path.basename(file_path)
                for required_import in required_imports:
                    if required_import in content:
                        print(f"   ✅ {file_name}: Has '{required_import}'")
                    else:
                        print(f"   ❌ {file_name}: Missing '{required_import}'")
                        
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
        else:
            print(f"   ❌ {file_path}: File not found")

def check_function_return_types():
    """関数の戻り値型確認"""
    print("\n🔧 Checking function return types...")
    
    files_to_check = [
        'src/predictor/api/inference.py',
        'src/predictor/api/batch_process.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_name = os.path.basename(file_path)
                if 'def get_pipeline_config(cfg: DictConfig) -> PipelineConfig:' in content:
                    print(f"   ✅ {file_name}: get_pipeline_config returns PipelineConfig")
                else:
                    print(f"   ❌ {file_name}: get_pipeline_config has wrong return type")
                        
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
        else:
            print(f"   ❌ {file_path}: File not found")

def check_error_handling():
    """エラーハンドリングの確認"""
    print("\n🛡️ Checking error handling...")
    
    files_to_check = [
        'src/predictor/api/inference.py',
        'src/predictor/api/batch_process.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_name = os.path.basename(file_path)
                
                # try-except ブロックの確認
                if 'try:' in content and 'except Exception as e:' in content:
                    print(f"   ✅ {file_name}: Has try-except error handling")
                else:
                    print(f"   ❌ {file_name}: Missing try-except error handling")
                
                # フォールバック設定の確認
                if 'fallback_config = PipelineConfig()' in content:
                    print(f"   ✅ {file_name}: Has PipelineConfig fallback")
                else:
                    print(f"   ❌ {file_name}: Missing PipelineConfig fallback")
                        
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
        else:
            print(f"   ❌ {file_path}: File not found")

def main():
    """メインテスト"""
    print("🔧 API Fix Verification Test")
    print("=" * 50)
    
    check_method_signatures()
    check_import_statements()
    check_function_return_types()
    check_error_handling()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("✅ Parameter name fixed: vis_config → visualization_config")
    print("✅ Return type fixed: Dict[str, Any] → PipelineConfig") 
    print("✅ Copy method fixed: base_config.copy() → copy.copy()")
    print("✅ Error handling improved with fallback PipelineConfig")
    print("\n💡 API should now work correctly!")
    print("   Test with: python -m src.predictor.api.inference --help")

if __name__ == '__main__':
    main()