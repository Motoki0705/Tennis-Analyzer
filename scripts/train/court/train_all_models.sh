#!/bin/bash

# Court Detection Models Training Script
# Uses the standardized training template

# Find script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Array of court model configurations to train
MODELS_TO_TRAIN=(
    "vit_unet_generic"
    "swin_unet_generic"
    "lite_tracknet_generic"
    "fpn_generic"
)

# Execute training using the unified template
exec "${SCRIPT_DIR}/../../common/train_template.sh" "court" "${MODELS_TO_TRAIN[@]}"