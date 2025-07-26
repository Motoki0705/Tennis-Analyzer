#!/usr/bin/env python
"""
Court Detection Model Training Script
Unified training interface using the common training system.
"""
from src.common.api.training import TrainingManager

# Create training function using the unified system
main = TrainingManager.create_training_script(
    module_name="court",
    config_name="lite_tracknet_generic"
)

if __name__ == "__main__":
    main() 