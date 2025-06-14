#!/bin/bash

# ===============================================
# 実験スクリプト
# ===============================================

# --- MSE損失で学習を実行 ---
echo "Running experiment with MSE Loss..."
python -m src.ball.api.train --config-name seq_lite_transformer

echo "All experiments finished."
