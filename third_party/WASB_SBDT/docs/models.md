
# Models

This document describes the model architecture and how to build models within the WASB-SBDT framework.

## `build_model(cfg)`

This function is the entry point for creating a model instance. It reads the model configuration from the provided `cfg` object and instantiates the corresponding model class.

-   **`cfg`**: An `OmegaConf` object containing the model configuration. The `cfg.model.name` field is used to determine which model to build.

**Returns:** A `torch.nn.Module` object representing the constructed model.

## Supported Models

The `build_model` function supports the following models through a factory pattern:

-   **`hrnet`**: (Default) High-Resolution Net (HRNet), a state-of-the-art model for human pose estimation, adapted for ball detection. It maintains high-resolution representations through the entire process, leading to strong performance.
-   **`tracknetv2`**: A U-Net based architecture specifically designed for ball tracking.
-   **`monotrack`**: A model for monocular object tracking.
-   **`restracknetv2`**: A ResNet-based version of TrackNetV2.
-   **`deepball`**: A CNN-based model for ball detection.
-   **`ballseg`**: A segmentation-based model for ball detection.

## Configuration

The behavior of `build_model` is controlled by the `model` section of the configuration file. Key parameters include:

-   **`name`**: The name of the model to use (e.g., `'hrnet'`).
-   **`frames_in`**: The number of input frames for the model.
-   **`frames_out`**: The number of output frames (heatmaps) produced by the model.
-   **`inp_width`**, **`inp_height`**: The dimensions of the input frames.

Each model has its own specific set of parameters that can be configured. For example, the `hrnet` model has detailed parameters for its stages, branches, and blocks.

## How It Works

The `build_model` function acts as a factory. It uses a dictionary (`__factory`) to map model names to their corresponding classes. When called, it looks up the model name from the configuration and instantiates the class with the parameters also defined in the configuration.

This design allows for easy extension with new models by simply adding them to the `__factory` dictionary and providing the necessary configuration options.
