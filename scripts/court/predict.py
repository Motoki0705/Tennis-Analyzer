#!/usr/bin/env python
"""
コート検出モデルの予測用スクリプト

使用例:
    python scripts/court/predict.py predict.image_dir=datasets/test
"""

import os
import sys

# プロジェクトルートをPYTHONPATHに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.court.predict import main

if __name__ == "__main__":
    main() 