
# API Reference

This document provides a reference for the main API of the WASB-SBDT package.

## Overview

The `__init__.py` file serves as the main entry point to the package, providing high-level functions to create and configure the ball detection and tracking system. It simplifies the process of setting up the model, postprocessor, and tracker.

## Key Functions

### `create_ball_detector(config_path, model_path=None)`

This is the primary function for creating a complete ball detection system.

-   **`config_path`**: Path to a YAML configuration file or an `OmegaConf` object. This defines the parameters for the model, detector, and tracker.
-   **`model_path`** (optional): Path to the pre-trained model weights (`.pth` or `.pth.tar` file). If provided, it overrides the model path specified in the configuration.

**Returns:** A tuple containing the `detector` object. The `detector` object is a high-level class that encapsulates the model, post-processing, and tracking logic.

**Example:**

```python
from third_party.WASB_SBDT.src import create_ball_detector, load_default_config

# Load the default configuration
cfg = load_default_config()

# Create the detector
detector = create_ball_detector(cfg, model_path='path/to/your/model.pth')

# Now the detector is ready to process video frames
```

### `load_default_config()`

This function loads a default configuration for the tennis ball detection system. The default configuration is suitable for general use and is based on the HRNet model.

**Returns:** An `OmegaConf` object containing the default settings.

## Core Components

The `create_ball_detector` function internally builds and connects the following components:

-   **Model (`build_model`)**: The neural network (e.g., HRNet) that takes a sequence of frames as input and produces heatmaps indicating the likelihood of a ball's presence.
-   **Detector (`build_detector`)**: A high-level class that orchestrates the detection process. It uses the model to get heatmaps and then uses the postprocessor to extract ball coordinates.
-   **Postprocessor (`TracknetV2Postprocessor`)**: Processes the raw heatmaps from the model to identify and locate ball candidates. It uses techniques like blob detection and connected components analysis.
-   **Tracker (`build_tracker`)**: An online tracker that takes the detected ball candidates from each frame and links them over time to create a smooth trajectory. This helps to handle occlusions and false positives.

## Workflow

The typical workflow for using this package is as follows:

1.  **Load Configuration**: Load the default configuration using `load_default_config()` or provide a path to a custom YAML file.
2.  **Create Detector**: Instantiate the main `detector` object using `create_ball_detector()`, optionally providing a path to your model weights.
3.  **Process Frames**: Feed video frames to the `detector.process_frames()` method.
4.  **Get Results**: The detector returns the tracking results, including the ball's coordinates and visibility status for each frame.
