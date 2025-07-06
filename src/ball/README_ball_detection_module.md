# Ball Detection Module

A modular ball detection system that can easily integrate third-party ball detection models from external sources. The module is designed for batch inference and provides a clean, extensible interface.

## Features

- **Three-stage pipeline**: Preprocess → Infer → Postprocess with consistent metadata tracking
- **Multi-model support**: LiteTrackNet and WASB-SBDT models
- **Batch processing**: Efficient inference on multiple frames
- **Flexible configuration**: Support for different devices and model types
- **Consistent output format**: Normalized coordinates [0,1] with confidence scores
- **Extensible architecture**: Easy to add new ball detection models

## Architecture

The module implements a three-stage pipeline:

1. **Preprocess**: Convert raw frames to model-compatible format while preserving metadata
2. **Infer**: Perform batch inference using the loaded model  
3. **Postprocess**: Convert raw outputs to standardized ball detection coordinates

```
Input: List[(frame, metadata)] → Preprocess → Infer → Postprocess → Output: Dict[frame_id, [[x, y, conf]]]
```

## Quick Start

### Basic Usage

```python
from src.ball.ball_detection_module import create_ball_detection_module
import cv2

# Create detector (auto-detects model type)
detector = create_ball_detection_module(
    model_path="checkpoints/ball/lite_tracknet/model.ckpt",
    device="auto"  # Uses GPU if available
)

# Prepare frame data
frame_data = []
cap = cv2.VideoCapture("video.mp4")
for i in range(10):  # Process 10 frames
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        metadata = {'frame_id': f'frame_{i:06d}', 'timestamp': i/30.0}
        frame_data.append((frame_rgb, metadata))

# Run detection
detections = detector.detect_balls(frame_data)

# Results format: {frame_id: [[x, y, confidence], ...]}
for frame_id, balls in detections.items():
    for ball in balls:
        x, y, conf = ball  # Normalized coordinates [0,1]
        print(f"Frame {frame_id}: Ball at ({x:.3f}, {y:.3f}) confidence {conf:.3f}")
```

### Step-by-step Processing

```python
# Manual pipeline control
model_inputs = detector.preprocess(frame_data)
inference_results = detector.infer(model_inputs)
detections = detector.postprocess(inference_results)
```

## Supported Models

### LiteTrackNet (Internal Model)

- **Input**: 3 consecutive frames concatenated channel-wise [B, 9, H, W]
- **Output**: Heatmap for ball position detection
- **Model files**: PyTorch Lightning checkpoints (`.ckpt`)
- **Configuration**: Built-in transforms and preprocessing

```python
detector = create_ball_detection_module(
    model_path="checkpoints/ball/lite_tracknet/model.ckpt",
    model_type="lite_tracknet"
)
```

### WASB-SBDT (Third-party Model)

- **Input**: Configurable number of frames (typically 3)
- **Output**: Ball detections with tracking
- **Model files**: PyTorch weights (`.pth`, `.pth.tar`)
- **Configuration**: Uses `load_simple_config()` or custom config

```python
detector = create_ball_detection_module(
    model_path="third_party/WASB-SBDT/models/model.pth",
    model_type="wasb_sbdt",
    config_path="config.yaml"  # Optional
)
```

## API Reference

### BallDetectionModule

Main class providing unified interface for ball detection.

#### Constructor

```python
BallDetectionModule(model_path, config_path=None, device="auto", model_type="auto")
```

**Parameters:**
- `model_path` (str): Path to trained model weights
- `config_path` (str, optional): Path to configuration file
- `device` (str): Device for inference ("cuda", "cpu", or "auto")
- `model_type` (str): Model type ("lite_tracknet", "wasb_sbdt", or "auto")

#### Methods

##### `detect_balls(frame_data)`

End-to-end ball detection pipeline.

**Parameters:**
- `frame_data` (List[Tuple[np.array, dict]]): List of (frame, metadata) tuples

**Returns:**
- `Dict[str, List[List[float]]]`: Dictionary with frame_id as keys and ball detections as values

##### `preprocess(frame_data)`

Convert frames to model input format while preserving metadata.

**Parameters:**
- `frame_data` (List[Tuple[np.array, dict]]): Raw frame data

**Returns:**
- `List[Tuple[Any, dict]]`: Model-ready inputs with metadata

##### `infer(model_inputs)`

Perform batch inference while maintaining metadata association.

**Parameters:**
- `model_inputs` (List[Tuple[Any, dict]]): Preprocessed inputs

**Returns:**
- `List[Tuple[Any, dict]]`: Raw model outputs with metadata

##### `postprocess(inference_results)`

Convert raw outputs to standardized ball coordinates.

**Parameters:**
- `inference_results` (List[Tuple[Any, dict]]): Raw inference results

**Returns:**
- `Dict[str, List[List[float]]]`: Standardized detection format

##### `get_model_info()`

Get information about the loaded model.

**Returns:**
- `Dict[str, Any]`: Model configuration and metadata

## Data Formats

### Input Format

Frame data should be provided as a list of tuples:

```python
frame_data = [
    (frame_array, metadata_dict),
    ...
]
```

**Frame Array:**
- Type: `numpy.ndarray`
- Shape: `[H, W, C]` (height, width, channels)
- Format: RGB color order
- Data type: `uint8` (0-255)

**Metadata Dictionary:**
- Required: `frame_id` (str): Unique identifier for the frame
- Optional: `timestamp` (float), `frame_number` (int), etc.

### Output Format

Detection results are returned as a dictionary:

```python
{
    "frame_000001": [[x1, y1, conf1], [x2, y2, conf2], ...],
    "frame_000002": [[x1, y1, conf1], ...],
    ...
}
```

**Coordinates:**
- `x, y`: Normalized coordinates in range [0.0, 1.0]
- `conf`: Confidence score in range [0.0, 1.0]
- Multiple balls per frame are supported

## Configuration

### Device Selection

```python
# Automatic device selection (recommended)
detector = create_ball_detection_module(model_path, device="auto")

# Force specific device
detector = create_ball_detection_module(model_path, device="cuda")
detector = create_ball_detection_module(model_path, device="cpu")
```

### Model Type Selection

```python
# Automatic model type detection (recommended)
detector = create_ball_detection_module(model_path, model_type="auto")

# Force specific model type
detector = create_ball_detection_module(model_path, model_type="lite_tracknet")
detector = create_ball_detection_module(model_path, model_type="wasb_sbdt")
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    detector = create_ball_detection_module("model.ckpt")
    detections = detector.detect_balls(frame_data)
except FileNotFoundError:
    print("Model file not found")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Model loading failed: {e}")
```

Common errors:
- `FileNotFoundError`: Model file doesn't exist
- `ValueError`: Insufficient frames or invalid input format
- `RuntimeError`: Model loading or inference failure
- `ImportError`: Required dependencies not available

## Performance Considerations

### Memory Usage

- **Batch size**: Process frames in smaller batches for large videos
- **Frame resolution**: Lower resolution reduces memory usage
- **Device memory**: Monitor GPU memory usage for large batches

### Processing Speed

- **GPU acceleration**: Use CUDA when available for significant speedup
- **Model type**: LiteTrackNet is generally faster than WASB-SBDT
- **Frame skip**: Process every N-th frame for real-time applications

### Example: Batch Processing Large Videos

```python
def process_large_video(video_path, batch_size=32):
    detector = create_ball_detection_module("model.ckpt")
    cap = cv2.VideoCapture(video_path)
    
    all_detections = {}
    frame_batch = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        metadata = {'frame_id': f'frame_{frame_idx:06d}'}
        frame_batch.append((frame_rgb, metadata))
        frame_idx += 1
        
        # Process batch when full
        if len(frame_batch) >= batch_size:
            batch_detections = detector.detect_balls(frame_batch)
            all_detections.update(batch_detections)
            frame_batch = []
    
    # Process remaining frames
    if frame_batch:
        batch_detections = detector.detect_balls(frame_batch)
        all_detections.update(batch_detections)
    
    return all_detections
```

## Extending the Module

### Adding New Model Types

1. **Create detector class** inheriting from `BaseBallDetector`:

```python
class CustomDetector(BaseBallDetector):
    def __init__(self, model_path, device):
        # Initialize your model
        pass
    
    def preprocess(self, frame_data):
        # Convert frames to your model's input format
        pass
    
    def infer(self, model_inputs):
        # Run inference with your model
        pass
    
    def postprocess(self, inference_results):
        # Convert outputs to standard format
        pass
```

2. **Register in BallDetectionModule**:

```python
def _create_detector(self, model_type):
    if model_type == "custom_model":
        return CustomDetector(self.model_path, self.device)
    # ... existing types
```

### Custom Preprocessing

```python
class CustomLiteTrackNetDetector(LiteTrackNetDetector):
    def preprocess(self, frame_data):
        # Custom preprocessing logic
        processed = super().preprocess(frame_data)
        # Additional processing...
        return processed
```

## Examples

See `src/ball/example_usage.py` for comprehensive examples including:

- Basic ball detection with different models
- Batch processing strategies
- Error handling and edge cases
- Visualization of detection results
- Performance optimization techniques

## Dependencies

Required packages:
- `torch` >= 1.8.0
- `torchvision`
- `numpy`
- `opencv-python` 
- `albumentations` (for LiteTrackNet)
- `pytorch-lightning` (for LiteTrackNet)

Optional packages:
- `omegaconf` (for WASB-SBDT configuration)

## Troubleshooting

### Import Errors

**Problem**: `ImportError: No module named 'src.ball.lit_module'`
**Solution**: Ensure PYTHONPATH includes the project root directory

**Problem**: `ImportError: WASB-SBDT modules not available`
**Solution**: Check if `third_party/WASB-SBDT` is properly installed

### Model Loading Issues

**Problem**: `RuntimeError: Failed to load model`
**Solution**: Verify model file exists and is compatible with the detector type

**Problem**: `CUDA out of memory`
**Solution**: Reduce batch size or use CPU inference

### Detection Quality Issues

**Problem**: Low detection confidence scores
**Solution**: Check input frame quality, model compatibility, and preprocessing

**Problem**: No detections returned
**Solution**: Verify sufficient frames provided (3+ for most models)

## License

This module is part of the Tennis Systems project. See the main project license for details.