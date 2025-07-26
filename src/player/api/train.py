#!/usr/bin/env python
"""
Player Detection Model Training Script
Unified training interface using the common training system.
"""
from src.common.api.training import TrainingManager

# Create training function using the unified system
main = TrainingManager.create_training_script(
    module_name="player",
    config_name="rtdetr_generic"
)

if __name__ == "__main__":
    main() 