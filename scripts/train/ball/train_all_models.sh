#!/bin/bash

# Ball Detection Models Training Script
# Uses the standardized training template

# Find script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Array of ball model configurations to train
MODELS_TO_TRAIN=(
    "conv3d_tsm_fpn"
    "video_swin_generic"
    "lite_tracknet_generic"
)

# Execute training using the unified template
exec "${SCRIPT_DIR}/../../common/train_template.sh" "ball" "${MODELS_TO_TRAIN[@]}"