#!/bin/bash

# ===============================================
# 実験スクリプト
# ===============================================

echo "Running experiment with Event Transformer..."
python -m src.event.api.train --config-name config

echo "All experiments finished."