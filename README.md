# Automated Tennis Analysis System

This project is an AI-powered toolkit designed for the comprehensive analysis of tennis match videos. It leverages deep learning models to automatically detect and track key elements such as the tennis ball, players, court lines, and player poses. The system can process video footage to extract valuable insights and generate annotated outputs.

## Features

This system offers a range of features for detailed sports video analysis:

*   **Ball Detection and Tracking:** Accurately identifies and follows the ball's trajectory throughout the video.
*   **Court Line Detection:** Segments and identifies the court lines to understand the play area.
*   **Player Detection and Tracking:** Detects players on the court and can track their movements.
*   **Player Pose Estimation:** Estimates the poses of players, providing insights into their posture and actions.
*   **Multi-Object Overlay:** Combines detections (ball, court, players, pose) into a single video output with overlays.
*   **Frame-by-Frame Annotation:** Can output detailed annotations for each frame in JSONL format for further analysis or integration with other tools.
*   **Configurable Analysis:** Utilizes Hydra for flexible configuration of models, inference parameters, and input/output settings.

## Technologies Used

This project is built using a combination of powerful tools and libraries:

*   **Programming Language:** Python (primarily Python 3.11)
*   **Core Deep Learning Frameworks:**
    *   PyTorch
    *   PyTorch Lightning (for structuring training code)
*   **Configuration Management:**
    *   Hydra (for flexible and organized configuration)
*   **Computer Vision:**
    *   OpenCV (for video processing and image manipulation)
    *   Albumentations (for data augmentation)
*   **Data Handling & Scientific Computing:**
    *   NumPy
    *   Pandas
*   **Key Model Architectures/Libraries:** (The system integrates various models, potentially including or similar to)
    *   TrackNet (for ball/object tracking)
    *   Swin Transformer
    *   Vision Transformer (ViT)
    *   DETR (Detection Transformer)
    *   UniFormer/SlowFast (for video understanding tasks)
    *   `timm` (PyTorch Image Models)
    *   `transformers` (Hugging Face)
*   **Experiment Tracking (implied by dependencies):**
    *   TensorBoard / TensorBoardX
*   **Video Processing Utilities:**
    *   `yt-dlp` (for downloading video clips if needed)

## Project Structure

The project is organized as follows:

```
.
├── configs/              # Hydra configuration files for training, inference, etc.
│   ├── infer/            # Inference specific configurations
│   └── train/            # Training specific configurations
├── docs/                 # Documentation and supplementary materials
│   └── images/           # Images used in documentation
├── hydra_outputs/        # Default output directory for Hydra (logs, models, etc.)
├── samples/              # Sample video files for testing and demos
├── src/                  # Main source code
│   ├── annotation/       # Scripts for data annotation
│   ├── ball/             # Ball detection and tracking module
│   ├── court/            # Court line detection module
│   ├── event/            # Event recognition module
│   ├── multi/            # Module for combined multi-object analysis
│   ├── player/           # Player detection and tracking module
│   ├── pose/             # Pose estimation module
│   ├── utils/            # Common utility functions and classes
│   ├── infer.py          # Main script for running inference
│   ├── train_ball.py     # Example training script for ball detection
│   ├── train_court.py    # Example training script for court detection
│   └── train_player.py   # Example training script for player detection
├── tests/                # Test scripts (if any)
├── tools/                # Utility scripts for various tasks (e.g., dataset conversion)
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── pyproject.toml        # Project metadata and build system configuration (includes linters)
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

*   **`src/`**: Contains the core logic, with subdirectories for each major component (ball, court, player, pose, event detection) and shared utilities. Each component typically includes submodules for datasets, models, predictors, and trainers.
*   **`configs/`**: Holds all Hydra configuration files, separating settings for different tasks (inference, training) and components.
*   **`samples/`**: Provides example video data for quick tests and demonstrations.
*   **`tools/`**: Includes various helper scripts, such as data converters.
*   **`docs/`**: Contains additional documentation, including the original detailed project structure.
*   **`hydra_outputs/`**: Where logs, model checkpoints, and other artifacts from runs are typically saved by Hydra.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Python environment:**
    It's recommended to use a virtual environment (e.g., venv, conda). This project uses Python 3.11 (as indicated in `pyproject.toml`).
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Or using conda:
    ```bash
    conda create -n sports_analysis python=3.11
    conda activate sports_analysis
    ```

3.  **Install dependencies:**
    The project dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file seems to have a UTF-16 encoding. If you encounter issues, you might need to convert it to UTF-8 first or handle the encoding during installation if your pip version supports it.*
    One of the dependencies is a specific commit of `UniFormer` from GitHub. Ensure git is installed for pip to clone it.

4.  **Pre-trained Models (Checkpoints):**
    The configuration files (`configs/infer/**/*.yaml`) refer to `ckpt_path` for various models. You will need to:
    *   Download pre-trained model checkpoints if available separately.
    *   Or, train your own models (see Training section) and update these paths accordingly.
    *   The paths in the configs might be placeholders or relative paths that assume checkpoints are placed in a specific directory (e.g., `checkpoints/` or within `hydra_outputs/`).

5.  **Environment Variables (Optional):**
    Check if any components require specific environment variables (e.g., for accessing datasets or APIs). This is not explicitly stated but is common in complex projects.

## Usage

This project uses Hydra for managing configurations. All configurations can be found in the `configs/` directory and can be overridden via the command line.

### 1. Configuration

*   **Main Configuration Files:**
    *   Training: `configs/train/` (e.g., `ball.yaml`, `court.yaml`)
    *   Inference: `configs/infer/` (e.g., `infer.yaml`, and specific task configs under `ball/`, `court/`, etc.)
*   **Hydra Overrides:** You can override any configuration parameter from the command line. For example, to change the batch size for training:
    ```bash
    python src/train_ball.py batch_size=16
    ```
*   **Output Directory:** Hydra automatically creates output directories for each run under `hydra_outputs/`. This includes logs, checkpoints (if saved via callbacks), and other artifacts.

### 2. Training Models

Training scripts are provided in the `src/` directory (e.g., `train_ball.py`, `train_court.py`, `train_player.py`). These scripts use PyTorch Lightning.

*   **Example: Training the Ball Detection Model**
    ```bash
    python src/train_ball.py \
        annotation_file=path/to/your/ball_annotations.json \
        image_root=path/to/your/images/ \
        trainer.max_epochs=100 \
        batch_size=8 \
        num_workers=4
    ```
    *   Adjust `annotation_file` and `image_root` to point to your dataset.
    *   Modify other parameters as needed by referring to the respective config file (e.g., `configs/train/ball.yaml`).
    *   Trained model checkpoints will typically be saved by callbacks like `ModelCheckpoint` (often configured in the training script or its Hydra config) into the Hydra output directory for that run.

### 3. Running Inference

The main script for inference is `src/infer.py`. It supports various modes for different analysis tasks.

*   **General Command Structure:**
    ```bash
    python src/infer.py \
        mode=<inference_mode> \
        input_path=<path_to_video_or_image_directory> \
        output_path=<path_to_output_video_or_directory> \
        common.device=cuda  # or cpu
        [other_config_overrides]
    ```

*   **Inference Modes (`mode=`):**
    *   `ball`: Ball detection and tracking.
        ```bash
        python src/infer.py mode=ball input_path=samples/sample_video.mp4 output_path=outputs/ball_detected.mp4
        ```
    *   `court`: Court line detection.
        ```bash
        python src/infer.py mode=court input_path=samples/sample_video.mp4 output_path=outputs/court_detected.mp4
        ```
    *   `player`: Player detection.
        ```bash
        python src/infer.py mode=player input_path=samples/sample_video.mp4 output_path=outputs/player_detected.mp4
        ```
    *   `pose`: Player pose estimation (combines player detection with pose estimation).
        ```bash
        python src/infer.py mode=pose input_path=samples/sample_video.mp4 output_path=outputs/pose_estimated.mp4
        ```
    *   `multi`: Combines ball, court, and pose detections with overlays on the output video.
        ```bash
        python src/infer.py mode=multi input_path=samples/sample_video.mp4 output_path=outputs/multi_analysis.mp4
        ```
    *   `frames`: Processes video and outputs frame-by-frame annotations to a JSONL file and/or saves individual frames.
        ```bash
        python src/infer.py mode=frames input_path=samples/sample_video.mp4 output_path=outputs/frame_data/ output_json_path=outputs/annotations.jsonl
        ```

*   **Important Considerations for Inference:**
    *   **Checkpoint Paths (`ckpt_path`):** Ensure the `ckpt_path` variables within the relevant configuration files (e.g., `configs/infer/ball/lite_tracknet.yaml`) point to your trained or downloaded model checkpoints. You can override these paths from the command line:
        ```bash
        python src/infer.py mode=ball ball.ckpt_path=path/to/your/ball_model.ckpt ...
        ```
    *   **Input/Output:**
        *   `input_path`: Path to the input video file or directory of images.
        *   `output_path`: Path for the output video or directory. For `frames` mode, this is often a directory.
        *   `output_json_path` (for `frames` mode): Path to the output JSONL file.
    *   **Device:** Use `common.device=cuda` for GPU or `common.device=cpu` for CPU.
    *   **Half Precision:** For faster inference on compatible GPUs, you can try enabling half precision: `common.use_half=True`.

### 4. Annotation Tools

Scripts for data annotation and preparation can be found in `src/annotation/`.
*   `auto_court_annotator.py`: May assist in automatically annotating court lines.
*   `generate_pose.py`: Could be for generating pose annotations from detected players.
    (Refer to the scripts themselves for detailed usage.)

## Examples/Demos

*   **Sample Videos:** The `samples/` directory contains some video files (e.g., `lite_tracknet.mp4`, `multi_overlay_up.mp4`, `overlay_fpn.mp4`) that can be used as input for testing the inference scripts.
    Example using a sample video:
    ```bash
    python src/infer.py mode=multi input_path=samples/multi_overlay_up.mp4 output_path=outputs/demo_multi_output.mp4
    ```

*   **Demo Script:** The file `chat_gpt/demo_cache_batch.py` appears to be a demonstration script, possibly showcasing batch processing or caching mechanisms. Refer to the script for its specific usage and purpose.

*   **Visualization of Keypoints:** The `docs/images/` directory contains sample images showing visualized keypoints (e.g., `visualized_keypoints_sample_0_PuXlxKdUIes_2450.png`), which likely demonstrate the output of the pose estimation or other keypoint detection modules.

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please consider the following:

*   **Bug Reports:** If you find a bug, please open an issue detailing the problem, steps to reproduce, and your environment.
*   **Feature Requests:** For new features or enhancements, feel free to open an issue to discuss the idea.
*   **Pull Requests:**
    1.  Fork the repository.
    2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name` or `bugfix/your-bug-fix`).
    3.  Make your changes.
    4.  Ensure your code adheres to the project's style (linters like Black, isort, Ruff are configured in `pyproject.toml`).
    5.  Write tests for your changes if applicable.
    6.  Commit your changes and push to your fork.
    7.  Open a pull request to the main repository.

Please note that this project appears to have some Japanese comments and naming conventions. When contributing, try to maintain consistency or discuss with the maintainers about language use in code and documentation.

## License

The license for this project has not yet been specified. It is recommended to add a LICENSE file to the repository to clarify how others can use and contribute to the project. Common open-source licenses include MIT, Apache 2.0, and GPLv3.
