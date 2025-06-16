#!/bin/bash

# ===============================================
# 実験スクリプト
# ===============================================

echo "Running experiment with RT-DETR..."
python -m src.player.api.train --config-name config

echo "All experiments finished."