#!/bin/bash

# ===============================================
# 実験スクリプト
# ===============================================

# --- Focal Lossで学習を実行 ---
echo "Running experiment with Focal Loss..."
python -m src.ball.api.train --config-name seq_lite_transformer_focal

echo "All experiments finished."
