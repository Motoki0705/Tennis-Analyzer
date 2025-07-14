# Tennis Ball Tracking Module - Complete Documentation

## Overview

The Tennis Ball Tracking module is a comprehensive system for detecting and tracking tennis balls in video sequences. It supports multiple neural network architectures, various data formats, and both heatmap-based and coordinate regression approaches.

## Architecture Overview

### Core Components

1. **Models** (`src/ball/models/`)
   - `lite_tracknet.py`: Lightweight U-Net architecture with depthwise separable convolutions
   - `video_swin_transformer.py`: Advanced spatio-temporal transformer for video sequences
   - `conv3d_tsm_fpn.py`: 3D CNN with Temporal Shift Module and Feature Pyramid Network

2. **Lightning Modules** (`src/ball/lit_module/`)
   - `lit_generic_ball_model.py`: Generic PyTorch Lightning wrapper supporting any model architecture

3. **Data Handling** (`src/ball/dataset/`, `src/ball/lit_datamodule/`)
   - `seq_key_dataset.py`: Heatmap-based dataset for keypoint detection
   - `seq_coord_dataset.py`: Coordinate regression dataset
   - `ball_data_module.py`: Lightning DataModule supporting both dataset types

4. **Training Infrastructure** (`src/ball/api/`)
   - `train.py`: Hydra-based training script with checkpoint management

## Model Architectures

### 1. LiteTrackNet (`lite_tracknet.py`)

**Purpose**: Lightweight real-time ball detection with heatmap output

**Architecture**:
- **Input**: `(B, C*T, H, W)` where C*T = concatenated channels from T frames
- **Output**: `(B, H, W)` heatmap
- **Design**: U-Net with depthwise separable convolutions and squeeze-excitation blocks

**Key Features**:
- Depthwise Separable Convolutions (DSConv) for efficiency
- Squeeze-and-Excitation (SE) blocks for channel attention
- Pixel Shuffle upsampling for better reconstruction
- Hardswish activation for mobile optimization

**Use Case**: Real-time applications requiring fast inference

### 2. Video Swin Transformer (`video_swin_transformer.py`)

**Purpose**: High-accuracy spatio-temporal ball tracking

**Architecture**:
- **Input**: `(B, N, C, H, W)` video sequences
- **Output**: `(B, N, 1, H, W)` per-frame heatmaps
- **Design**: CNN encoder + Swin Transformer + CNN decoder

**Key Components**:
1. **CNN Encoder**: 5-stage progressive downsampling with DSConv blocks
2. **Spatio-Temporal Transformer**: 
   - Window-based spatial attention (Swin Transformer)
   - Cross-frame temporal attention
   - Positional embeddings for temporal modeling
3. **CNN Decoder**: Progressive upsampling with skip connections

**Use Case**: High-accuracy applications where computational cost is less critical

### 3. Conv3D TSM FPN (`conv3d_tsm_fpn.py`)

**Purpose**: Efficient 3D spatio-temporal processing

**Architecture**:
- **Input**: `(B, N, 3, H, W)` video clips
- **Output**: `(B, N, H, W)` heatmaps
- **Design**: 3D CNN stem + TSM backbone + BiDirectional FPN

**Key Components**:
1. **Conv3D Stem**: Early temporal fusion with 3D convolutions
2. **Temporal Shift Module (TSM)**: Zero-cost temporal modeling
3. **ResNet-lite Backbone**: 2D convolutions with TSM integration
4. **Bi-Directional FPN**: Multi-scale feature fusion

**Use Case**: Balanced accuracy and efficiency for video processing

## Data Flow and Formats

### Input Data Structure

**COCO Format Annotations**:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "frame_001.jpg",
      "original_path": "game1/clip1/frame_001.jpg",
      "game_id": 1,
      "clip_id": 1,
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [960, 540, 2],  // x, y, visibility
      "bbox": [955, 535, 10, 10]
    }
  ]
}
```

### Dataset Types

#### 1. Heatmap Dataset (`SequenceKeypointDataset`)
- **Purpose**: Traditional keypoint detection with Gaussian heatmaps
- **Output**: Heatmap tensors `(B, T, H, W)` or `(B, H, W)`
- **Loss**: Focal Loss on heatmap values

#### 2. Coordinate Dataset (`SequenceCoordDataset`)
- **Purpose**: Direct coordinate regression
- **Output**: Normalized coordinates `(B, T, 2)` or `(B, 2)`
- **Loss**: L1/L2 loss on coordinate values

### Data Loading Configurations

**Input Types**:
- `"cat"`: Concatenate T frames into `(B, C*T, H, W)` - for 2D CNNs
- `"stack"`: Stack T frames into `(B, T, C, H, W)` - for 3D CNNs

**Output Types**:
- `"all"`: Return outputs for all T frames
- `"last"`: Return output only for the last frame

## Configuration System

### Hydra Configuration Structure

```
configs/train/ball/
├── lite_tracknet_generic.yaml     # Main config for LiteTrackNet
├── video_swin_generic.yaml        # Main config for Video Swin
├── conv3d_tsm_fpn.yaml           # Main config for Conv3D TSM
├── model/
│   ├── lite_tracknet.yaml        # Model architecture config
│   ├── video_swin.yaml           # Video Swin model config
│   └── conv3d_tsm_fpn.yaml       # Conv3D model config
├── litmodule/
│   └── generic_focal_loss.yaml   # Lightning module config
├── litdatamodule/
│   ├── 3_frames_cat_last.yaml    # 3-frame concatenated data
│   └── N_frames_stack_all.yaml   # N-frame stacked data
├── trainer/
│   └── default.yaml              # Training configuration
└── callbacks/
    └── default.yaml              # Callback configuration
```

### Key Configuration Parameters

**Model Configuration**:
```yaml
# litmodule/generic_focal_loss.yaml
_target_: src.ball.lit_module.lit_generic_ball_model.LitGenericBallModel
model: ${model}  # Injected from main config
criterion:
  alpha: 0.25
  gamma: 2.0
  reduction: "mean"
optimizer_params:
  lr: 1e-4
  weight_decay: 1e-4
scheduler_params:
  warmup_epochs: 1
  max_epochs: ${trainer.max_epochs}
```

**Data Configuration**:
```yaml
# litdatamodule/3_frames_cat_last.yaml
_target_: src.ball.lit_datamodule.ball_data_module.TennisBallDataModule
T: 3
input_type: "cat"
output_type: "last"
dataset_type: "heatmap"
batch_size: 32
```

## Training Pipeline

### 1. Training Script (`src/ball/api/train.py`)

**Features**:
- Automatic checkpoint resumption
- Comprehensive logging and visualization
- Error handling and recovery
- Hyperparameter tracking

**Key Functions**:
- `find_latest_checkpoint()`: Auto-resume from latest checkpoint
- `setup_callbacks()`: Configure training callbacks
- `get_best_model_info()`: Extract best model performance

### 2. Batch Training (`scripts/train/ball/train_all_models.sh`)

**Features**:
- Sequential training of all model configurations
- Automatic checkpoint archiving
- Comprehensive logging
- Environment validation
- Error recovery

**Usage**:
```bash
cd /path/to/tennis_systems
bash scripts/train/ball/train_all_models.sh
```

### 3. Training Process Flow

1. **Environment Check**: Validate Python, modules, and config files
2. **Data Preparation**: Load and split datasets by clips
3. **Model Instantiation**: Create model using Hydra configuration
4. **Training Loop**: 
   - Learning rate warmup
   - Cosine annealing schedule
   - Validation every epoch
   - Best checkpoint saving
5. **Checkpoint Archiving**: Move best models to standard locations

## Inference Pipeline

### Model Loading Pattern

```python
from src.ball.lit_module.lit_generic_ball_model import LitGenericBallModel

# Load from checkpoint
model = LitGenericBallModel.load_from_checkpoint(
    "checkpoints/ball/lite_tracknet_generic/best_model.ckpt"
)
model.eval()

# Or load specific architecture
from src.ball.models.lite_tracknet import LiteTrackNet
model = LiteTrackNet(in_channels=9, out_channels=1)
```

### Inference Example

```python
import torch
import cv2
from torchvision.transforms import Compose, Normalize, ToTensor

# Load model
model = LitGenericBallModel.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Prepare input (3 frames concatenated)
frames = []  # List of 3 consecutive frames
input_tensor = torch.cat([ToTensor()(frame) for frame in frames], dim=0)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    heatmap = model(input_tensor)
    
# Find ball position
ball_y, ball_x = torch.unravel_index(heatmap.argmax(), heatmap.shape[-2:])
```

## Complete Reconstruction Guide

### Prerequisites

1. **Environment Setup**:
```bash
# Create conda environment
conda create -n tennis_ball python=3.9
conda activate tennis_ball

# Install PyTorch (GPU version recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

2. **Directory Structure**:
```
tennis_systems/
├── src/ball/                    # Source code
├── configs/train/ball/          # Training configurations
├── scripts/train/ball/          # Training scripts
├── datasets/                    # Training data
│   ├── images/                  # Frame images
│   └── annotations/             # COCO format annotations
├── checkpoints/ball/            # Model checkpoints
└── outputs/                     # Training outputs
```

### Data Preparation

1. **Organize Video Data**:
```bash
# Structure your data as:
datasets/images/game_X/clip_Y/frame_ZZZZZ.jpg
```

2. **Create COCO Annotations**:
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x, y, visibility],
      "bbox": [x, y, w, h]
    }
  ],
  "categories": [{"id": 1, "name": "ball"}]
}
```

3. **Update Configuration Paths**:
```yaml
# In configs/train/ball/litdatamodule/*.yaml
annotation_file: "datasets/annotations/ball_annotations.json"
image_root: "datasets/images"
```

### Training from Scratch

1. **Single Model Training**:
```bash
cd tennis_systems
python -m src.ball.api.train --config-name lite_tracknet_generic
```

2. **Batch Training (All Models)**:
```bash
bash scripts/train/ball/train_all_models.sh
```

3. **Monitor Training**:
```bash
# Check logs
tail -f outputs/train_ball/*/training.log

# Tensorboard (if configured)
tensorboard --logdir outputs/train_ball/
```

### Model Customization

1. **Add New Model Architecture**:
```python
# src/ball/models/my_model.py
import torch.nn as nn

class MyBallDetector(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        # Your architecture here
        
    def forward(self, x):
        # x: (B, C*T, H, W) for "cat" input_type
        # or (B, T, C, H, W) for "stack" input_type
        return output  # (B, H, W) for heatmap
```

2. **Create Model Config**:
```yaml
# configs/train/ball/model/my_model.yaml
_target_: src.ball.models.my_model.MyBallDetector
in_channels: 9
out_channels: 1
```

3. **Create Main Config**:
```yaml
# configs/train/ball/my_model_config.yaml
defaults:
  - model: my_model
  - litmodule: generic_focal_loss
  - litdatamodule: 3_frames_cat_last
  - trainer: default
  - callbacks: default
  - _self_
```

### Advanced Configuration

1. **Custom Loss Function**:
```python
# In lit_module
def _common_step(self, batch, stage: str):
    frames, targets, _, _ = batch
    logits = self(frames)
    
    # Custom loss
    loss = my_custom_loss(logits, targets)
    return loss
```

2. **Custom Data Augmentation**:
```python
# Custom transform in datamodule
train_transform = A.ReplayCompose([
    A.Resize(height=input_size[0], width=input_size[1]),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(),
    A.pytorch.ToTensorV2(),
], keypoint_params=A.KeypointParams(format="xy"))
```

3. **Multi-GPU Training**:
```yaml
# configs/train/ball/trainer/multi_gpu.yaml
accelerator: gpu
devices: [0, 1, 2, 3]
strategy: ddp
```

### Troubleshooting

**Common Issues**:

1. **CUDA Out of Memory**:
   - Reduce batch_size in datamodule config
   - Use gradient accumulation
   - Enable mixed precision training

2. **Checkpoint Loading Errors**:
   - Check model architecture compatibility
   - Verify checkpoint file integrity
   - Use `strict=False` for partial loading

3. **Training Instability**:
   - Reduce learning rate
   - Increase warmup epochs
   - Check data normalization

4. **Poor Performance**:
   - Verify data annotation quality
   - Check input/output format compatibility
   - Tune loss function parameters

### Performance Optimization

1. **DataLoader Optimization**:
```yaml
num_workers: 8      # Adjust based on CPU cores
pin_memory: true    # For GPU training
persistent_workers: true  # Reduce worker startup time
```

2. **Mixed Precision Training**:
```yaml
# In trainer config
precision: 16       # Use fp16
```

3. **Compilation (PyTorch 2.0+)**:
```python
# In model definition
model = torch.compile(model)
```

### Deployment

1. **Export to TorchScript**:
```python
model = LitGenericBallModel.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# Trace or script
traced_model = torch.jit.trace(model, example_input)
traced_model.save("ball_detector.pt")
```

2. **ONNX Export**:
```python
torch.onnx.export(
    model, example_input, "ball_detector.onnx",
    export_params=True, opset_version=11,
    input_names=['input'], output_names=['heatmap']
)
```

### File Locations Reference

**Key Files for Reconstruction**:
- `src/ball/api/train.py`: Training entry point
- `src/ball/lit_module/lit_generic_ball_model.py`: Lightning wrapper
- `src/ball/lit_datamodule/ball_data_module.py`: Data loading
- `src/ball/models/*.py`: Model architectures
- `configs/train/ball/`: All configuration files
- `scripts/train/ball/train_all_models.sh`: Batch training script

**Checkpoint Locations**:
- `checkpoints/ball/{model_name}/best_model.ckpt`: Best trained models
- `outputs/train_ball/{model_name}/`: Training logs and temporary checkpoints

This documentation provides a complete guide for understanding, reproducing, and extending the tennis ball tracking system. Each component is designed to be modular and configurable, allowing for easy experimentation with different architectures and training strategies.