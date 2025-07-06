# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis Systems is a comprehensive AI-powered tennis analysis platform that integrates 5 main components:
- **Ball Tracking**: High-precision ball detection and trajectory tracking
- **Court Detection**: Tennis court keypoint detection (15 keypoints)
- **Player Detection**: RT-DETR based player detection
- **Pose Estimation**: 17 keypoint pose estimation using VitPose
- **Event Detection**: Multi-modal time-series analysis for hit/bounce detection

The system uses PyTorch Lightning for training and Gradio for demo interfaces.

## Development Commands

### Environment Setup
```bash
# Install dependencies (GPU PyTorch recommended from pytorch.org first)
pip install -r requirements.txt
```

### Running Demos
```bash
# Individual component demos
python demo/ball.py      # Ball tracking
python demo/court.py     # Court detection
python demo/player.py    # Player detection
python demo/pose.py      # Pose estimation
python demo/event.py     # Integrated event detection (recommended)
```

### Model Training
```bash
# Ball detection training
bash scripts/train/ball/lite_tracknet_focal.sh
python -m src.ball.api.train --config-name lite_tracknet_focal

# Court detection training
bash scripts/train/court/lite_tracknet_focal.sh
python -m src.court.api.train --config-name config

# Player detection training
bash scripts/train/player/rt_detr.sh
python -m src.player.api.train --config-name config

# Event detection training
bash scripts/train/event/event_transformer.sh
python -m src.event.api.train --config-name config
```

### Testing
```bash
# Run model instantiation tests
python -m pytest tests/infer_model_instantiate/
python -m pytest tests/train_model_instantiate/

# Run dataset tests
python -m pytest tests/data/

# Run compatibility tests
python -m pytest tests/valid_output_format/
```

### Annotation System
```bash
# Setup annotation environment
cd tools/annotation
./run_annotation_system.sh setup

# Start web annotation server
./run_annotation_system.sh start
# Access at http://localhost:8000

# Generate empty annotations from clips
python generate_empty_annotations.py --clips_dir ./data/clips --annotations_dir ./data/annotations

# Merge annotations to COCO format
python merge_to_coco.py --input_dir ./data/annotations --output_file ./data/dataset.json
```

## Architecture

### Code Structure
- `src/`: Main source code organized by component (ball, court, player, pose, event)
- `configs/`: Hydra configuration files for each component and task
- `demo/`: Gradio demo applications for each component
- `tools/`: Development tools including web-based annotation system
- `tests/`: Test suites for model instantiation and data compatibility
- `scripts/`: Training scripts for each component
- `checkpoints/`: Pre-trained model weights
- `third_party/`: External dependencies (e.g., WASB-SBDT ball tracker)

### Model Architecture Patterns
Each component follows PyTorch Lightning patterns:
- `lit_module/`: Lightning modules with training logic
- `lit_datamodule/`: Lightning data modules
- `models/`: Core model architectures
- `dataset/`: Dataset classes
- `api/train.py`: Training entry points

### Key Models
- **Ball**: LiteTrackNet (lightweight U-Net), Video Swin Transformer
- **Court**: LiteTrackNet with 15 keypoint detection
- **Player**: RT-DETR v2 fine-tuned for tennis players
- **Pose**: HuggingFace VitPose ("usyd-community/vitpose-base-simple")
- **Event**: EventTransformerV2 (multi-modal time-series transformer)

## Configuration Management

The project uses Hydra for configuration management:
- `configs/train/`: Training configurations
- `configs/infer/`: Inference configurations  
- `configs/test/`: Test configurations

Each component has separate config directories with modular YAML files for:
- `litmodule/`: Model configurations
- `litdatamodule/`: Data loading configurations
- `trainer/`: Training configurations
- `callbacks/`: Callback configurations

## Data Formats

### Input Formats
- **Ball**: 3-frame RGB sequences → heatmap
- **Court**: Single RGB image → single heatmap with 15 keypoints (detect 15 keypoints using peak detection for a single heatmap)
- **Player**: RGB image → bounding boxes + scores
- **Event**: Multi-modal time-series (ball, court, player, pose features)

### Output Formats
- Ball/Court: Heatmap format for keypoint detection
- Player: COCO-style bounding box format
- Event: Binary classification (hit/bounce probabilities)

## Important Implementation Notes

### Coding Standards
- Follow PEP 8 code style
- Use Google-style docstrings for all functions/methods
- Implement proper exception handling with logging
- Modularize code into functions/classes for reusability

### Environment Constraints
- Development and execution environments are separated
- Avoid running heavy processing or main functions during development
- Use appropriate imports and handle missing dependencies gracefully

### Model Loading Patterns
```python
# Standard Lightning checkpoint loading
from src.{component}.lit_module.{module} import {LitModule}
model = {LitModule}.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# HuggingFace model loading (pose estimation)
from transformers import VitPoseForPoseEstimation
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
```

## File System Structure

Critical directories:
- `checkpoints/{component}/`: Pre-trained model weights
- `datasets/`: Training/validation data
- `samples/`: Sample videos for testing
- `tools/annotation/web_app/`: React-based annotation interface
- `third_party/WASB-SBDT/`: External ball tracking integration

## Troubleshooting

### Common Issues
- CUDA memory issues: Adjust batch sizes in config files
- Missing model files: Check `checkpoints/` directory structure
- FFmpeg errors: Ensure FFmpeg is installed for video processing
- Import errors: Verify PYTHONPATH includes project root

### Logging
Enable detailed logging:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python demo/event.py --log-level DEBUG
```

### Dependencies
- PyTorch (install from pytorch.org for GPU support before pip install)
- OpenCV for video processing
- FFmpeg for annotation tools
- Node.js for annotation frontend