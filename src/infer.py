#!/usr/bin/env python
"""
このファイルは scripts/infer/infer.py へのリダイレクトです。
"""
import os
import sys
from pathlib import Path

# プロジェクトルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# scripts/infer ディレクトリをパスに追加
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

# infer モジュールをインポート
from infer.infer import *

if __name__ == "__main__":
    # main関数を実行
    main() 