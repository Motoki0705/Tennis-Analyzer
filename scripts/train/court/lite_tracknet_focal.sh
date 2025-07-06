#!/bin/bash

# ===============================================
# 実験スクリプト
# ===============================================

# --- Focal Lossで学習を実行 ---
echo "Running experiment with Focal Loss..."
python -m src.court.api.train --config-name config

echo "All experiments finished."