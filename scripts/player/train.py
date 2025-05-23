#!/usr/bin/env python
"""
プレイヤー検出モデルのトレーニング用スクリプト

使用例:
    python scripts/player/train.py
"""

import os
import sys

# プロジェクトルートをPYTHONPATHに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.player.train import main

if __name__ == "__main__":
    main() 