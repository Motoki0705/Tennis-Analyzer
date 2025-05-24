#!/usr/bin/env python
"""
このファイルは scripts/infer/infer.py へのリダイレクトです。
"""
import sys
from pathlib import Path

# scripts/infer ディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# infer モジュールをインポート
from infer.infer import *

if __name__ == "__main__":
    # main関数を実行
    main() 