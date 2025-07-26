#!/bin/bash

# Event Detection Models Training Script
# Uses the standardized training template

# Find script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Array of event model configurations to train
MODELS_TO_TRAIN=(
    "config"
)

# Execute training using the unified template
exec "${SCRIPT_DIR}/../../common/train_template.sh" "event" "${MODELS_TO_TRAIN[@]}"