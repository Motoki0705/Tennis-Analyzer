import json
import os
from typing import Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.utils.heatmap import draw_gaussian
from src.utils.visualization import visualize_overlay


class CourtDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        image_root,
        input_size=(360, 640),
        heatmap_size=(360, 640),
        default_num_keypoints=15,
        transform=None,
        sigma=2,
        is_each_keypoint=True,
    ):
        self.input_size = input_size
        self.image_root = image_root
        self.heatmap_size = heatmap_size
        self.default_num_keypoints = default_num_keypoints
        self.sigma = sigma
        self.is_each_keypoint = is_each_keypoint
        # Provide default transforms if none are given (resize, normalize, to_tensor)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transform(input_size)

        # Load dataset from JSON annotation file
        with open(annotation_path, "r") as f:
            self.data = json.load(f)

    def default_transform(self, input_size):
        return A.Compose(
            [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Load image
        file_name = example["file_name"]
        image_path = os.path.join(self.image_root, file_name)
        image: Image = self.load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image. image_path: {image_path}")

        # Convert keypoints to shape [num_keypoints, 3] => (x, y, visibility)
        keypoints = torch.tensor(example["keypoints"], dtype=torch.float32).view(-1, 3)
        default_num_keypoints = example.get("num_keypoints", self.default_num_keypoints)

        if keypoints.size(0) != default_num_keypoints:
            raise ValueError(
                f"Keypoint count mismatch before augmentation.\n"
                f"Actual: {keypoints.size(0)}\n"
                f"Expected: {default_num_keypoints}\n"
            )

        # Apply data augmentation (resizing, normalizing, to_tensor)
        image = np.array(image)
        argumented: dict = self.transform(image=image, keypoints=keypoints)
        arged_image: torch.Tensor = argumented["image"]
        arged_keypoints: list = argumented["keypoints"]
        arged_keypoints: torch.Tensor = torch.tensor(arged_keypoints)
        if arged_keypoints.size(0) != default_num_keypoints:
            raise ValueError(
                f"Keypoint count mismatch after augmentation.\n"
                f"After: {arged_keypoints.size(0)}\n"
                f"Expected: {default_num_keypoints}\n"
            )

        # Set visibility = 0 for keypoints outside the image
        self.filtering_outside_screen_keypoints(arged_keypoints, self.input_size)

        # Scale keypoints from input size to heatmap size
        scaled_keypoints: torch.Tensor = self.scaling_to_heatmap(
            arged_keypoints, self.input_size, self.heatmap_size
        )

        # Generate heatmaps for each keypoint
        if self.is_each_keypoint:
            heatmaps = torch.zeros(
                (len(scaled_keypoints), *self.heatmap_size), dtype=torch.float32
            )
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0:
                    self.draw_each_gaussian(
                        heatmaps[i], (x.item(), y.item()), sigma=self.sigma
                    )
            return arged_image, heatmaps

        else:
            heatmap = torch.zeros((1, *self.heatmap_size), dtype=torch.float32)
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0:
                    self.draw_each_gaussian(
                        heatmap[0], (x.item(), y.item()), sigma=self.sigma
                    )
            return arged_image, heatmap

    def load_image(self, image_path: str) -> Image:
        """Load an image from the given path and convert to RGB format.

        Parameters:
            image_path (str): File path to the image

        Returns:
            PIL.Image: RGB image object

        Raises:
            FileNotFoundError: If the file does not exist
            UnidentifiedImageError: If the file cannot be opened by PIL (caught internally)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            return image

        except UnidentifiedImageError as e:
            print(f"Failed to open image: {e}")
            return None

    def filtering_outside_screen_keypoints(
        self, keypoints: torch.Tensor, input_size: Tuple[int, int]
    ) -> None:
        """Set visibility to 0 for keypoints located outside the image bounds.

        Parameters:
            keypoints (torch.Tensor): Shape = [num_keypoints, 3]
            input_size (Tuple[int, int]): (height, width)
        """
        screen_h, screen_w = input_size
        for idx, keypoint in enumerate(keypoints):
            if 0 <= keypoint[0] <= screen_w - 1 and 0 <= keypoint[1] <= screen_h - 1:
                continue
            else:
                keypoint[2] = 0

    def scaling_to_heatmap(
        self,
        keypoints: torch.Tensor,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Scale keypoints from input size to heatmap size.

        Parameters:
            keypoints (torch.Tensor): Shape = [num_keypoints, 3]
            input_size (Tuple[int, int]): (height, width)
            heatmap_size (Tuple[int, int]): (height, width)

        Returns:
            torch.Tensor: Scaled keypoints
        """
        scale_x = heatmap_size[1] / input_size[1]
        scale_y = heatmap_size[0] / input_size[0]

        scaled_keypoints = keypoints.clone()
        scaled_keypoints[:, 0] *= scale_x
        scaled_keypoints[:, 1] *= scale_y

        return scaled_keypoints

    def draw_each_gaussian(self, heatmap, center, sigma=2):
        """Draw a 2D Gaussian on the heatmap centered at the given point.

        Parameters:
            heatmap (torch.Tensor): The target heatmap (H x W)
            center (Tuple[float, float]): Center of the Gaussian
            sigma (float): Standard deviation of the Gaussian
        """
        return draw_gaussian(heatmap, center, sigma)


if __name__ == "__main__":
    json_path = r"C:\Users\kamim\code\Tennis-Analyzer\data\court\converted_train.json"
    image_root = r"C:\Users\kamim\code\Tennis-Analyzer\data\court\images"
    dataset = CourtDataset(json_path, image_root, is_each_keypoint=False)
    image, heatmap = dataset[0]
    visualize_overlay(image, heatmap, alpha=0.6, cmap="jet", save_path="overlay.png")
