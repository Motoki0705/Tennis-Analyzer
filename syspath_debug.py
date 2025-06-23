import sys
import os
from pathlib import Path

def debug_sys_path():
    """sys.pathの詳細情報を表示"""
    print("=== sys.path Debug Information ===")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__ if '__file__' in globals() else 'Interactive mode'}")
    print()
    
    print("sys.path contents (in order):")
    for i, path in enumerate(sys.path):
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{i:2d}: {exists} {path}")
    print()

def find_module_paths(module_name):
    """指定したモジュール名のパスを検索"""
    print(f"=== Searching for '{module_name}' ===")
    found_paths = []
    
    for path in sys.path:
        if not os.path.exists(path):
            continue
            
        # ディレクトリとして存在するか
        module_dir = os.path.join(path, module_name)
        if os.path.isdir(module_dir):
            has_init = os.path.exists(os.path.join(module_dir, '__init__.py'))
            init_status = "with __init__.py" if has_init else "without __init__.py"
            found_paths.append(f"Directory: {module_dir} ({init_status})")
        
        # .pyファイルとして存在するか
        module_file = os.path.join(path, f"{module_name}.py")
        if os.path.isfile(module_file):
            found_paths.append(f"File: {module_file}")
    
    if found_paths:
        print("Found:")
        for path in found_paths:
            print(f"  - {path}")
    else:
        print(f"No '{module_name}' found in sys.path")
    print()

def check_import_conflicts():
    """インポートの競合をチェック"""
    print("=== Checking for potential import conflicts ===")
    
    # 同じ名前のモジュール/パッケージを探す
    module_names = set()
    conflicts = {}
    
    for path in sys.path:
        if not os.path.exists(path):
            continue
            
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                
                # Pythonパッケージ/モジュールかチェック
                is_package = (os.path.isdir(item_path) and 
                             os.path.exists(os.path.join(item_path, '__init__.py')))
                is_module = (os.path.isfile(item_path) and 
                           item.endswith('.py') and 
                           not item.startswith('__'))
                
                if is_package or is_module:
                    name = item.replace('.py', '') if is_module else item
                    
                    if name in module_names:
                        if name not in conflicts:
                            conflicts[name] = []
                        conflicts[name].append(item_path)
                    else:
                        module_names.add(name)
                        conflicts[name] = [item_path]
        except (PermissionError, OSError):
            continue
    
    # 競合を表示
    actual_conflicts = {name: paths for name, paths in conflicts.items() if len(paths) > 1}
    
    if actual_conflicts:
        print("Potential conflicts found:")
        for name, paths in actual_conflicts.items():
            print(f"  {name}:")
            for path in paths:
                print(f"    - {path}")
    else:
        print("No obvious conflicts detected")
    print()

def add_path_safely(new_path, position=None):
    """安全にsys.pathにパスを追加"""
    abs_path = os.path.abspath(new_path)
    
    if not os.path.exists(abs_path):
        print(f"Warning: Path does not exist: {abs_path}")
        return False
    
    if abs_path in sys.path:
        print(f"Path already in sys.path: {abs_path}")
        return False
    
    if position is None:
        sys.path.append(abs_path)
        print(f"Added to end of sys.path: {abs_path}")
    else:
        sys.path.insert(position, abs_path)
        print(f"Inserted at position {position}: {abs_path}")
    
    return True

def trace_import(module_name):
    """インポートをトレースして詳細を表示"""
    print(f"=== Tracing import of '{module_name}' ===")
    
    try:
        # importlibを使って詳細情報を取得
        import importlib.util
        import importlib
        
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"Module '{module_name}' not found")
            return
        
        print(f"Module name: {spec.name}")
        print(f"Module file: {spec.origin}")
        print(f"Is package: {spec.submodule_search_locations is not None}")
        
        if spec.submodule_search_locations:
            print(f"Package paths: {spec.submodule_search_locations}")
        
        # 実際にインポートしてみる
        try:
            module = importlib.import_module(module_name)
            print(f"Successfully imported: {module}")
            if hasattr(module, '__file__'):
                print(f"Module __file__: {module.__file__}")
            if hasattr(module, '__path__'):
                print(f"Module __path__: {module.__path__}")
        except Exception as e:
            print(f"Import failed: {e}")
            
    except Exception as e:
        print(f"Error during trace: {e}")
    print()

# 使用例
if __name__ == "__main__":
    # 基本的なsys.path情報
    debug_sys_path()
    
    # 特定のモジュールを検索（例：srcという名前）
    find_module_paths("src")
    
    # インポート競合をチェック
    check_import_conflicts()
    
    # 特定のモジュールのインポートをトレース
    trace_import("src")
    
    # パスを安全に追加する例
    # add_path_safely("proj", 0)  # 先頭に追加
    # add_path_safely("proj/third_party", 1)  # 2番目に追加