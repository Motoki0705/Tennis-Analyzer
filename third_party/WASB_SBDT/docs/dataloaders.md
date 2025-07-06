
# Dataloaders and Transforms

This document outlines the data loading and transformation pipeline in the WASB-SBDT package.

## Image Transforms (`img_transforms.py`)

Image transforms are crucial for data augmentation and for preparing images for the model. The `video_demo.py` uses a `Compose` object from `img_transforms` (which imports from `torchvision.transforms`) to chain together several transformations.

### `T.Compose([...])`

The `SimpleDetector` class in `video_demo.py` uses the following composition of transforms:

```python
self._transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

-   **`T.ToTensor()`**: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
-   **`T.Normalize(...)`**: Normalizes a tensor image with mean and standard deviation. This is a standard practice for many pre-trained models.

### Custom Transforms

The `img_transforms.py` file also defines several custom transformation classes for data augmentation:

-   **`ResizeWithEqualScale`**: Resizes an image while maintaining its original aspect ratio. The remaining area is padded to match the target dimensions.
-   **`RandomCroping`**: Randomly crops a section of the image. This helps the model to become more robust to variations in object position.
-   **`RandomErasing`**: Randomly erases a rectangular region in the image. This technique helps to prevent the model from overfitting and improves its ability to handle occlusions.

These custom transforms are not used in the `video_demo.py` but are available for use in training and data augmentation pipelines.

## Data Loading

While `video_demo.py` reads frames directly from a video file, the `dataloaders` directory contains modules for loading and preparing datasets for training. The `dataset_loader.py` file likely contains the main `Dataset` class for handling the specific data format used in this project.
