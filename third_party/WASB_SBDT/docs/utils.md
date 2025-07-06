
# Utilities

This document covers utility functions used in the WASB-SBDT package, with a focus on image-related operations.

## Image Utilities (`utils/image.py`)

This module provides functions for geometric transformations of images and coordinates, which are essential for the data preparation and post-processing steps.

### `get_affine_transform(center, scale, rot, output_size, shift, inv)`

This function computes the affine transformation matrix required to transform an image or a set of coordinates.

-   **`center`**: The center of the transformation in the source image.
-   **`scale`**: The scale factor for the transformation.
-   **`rot`**: The rotation angle in degrees.
-   **`output_size`**: The size of the output image.
-   **`shift`**: A shift to be applied to the center.
-   **`inv`**: If set to `1`, the function computes the inverse transformation.

**Returns:** A 2x3 affine transformation matrix (a NumPy array).

In `video_demo.py`, this function is used to:

1.  Create a transformation (`trans_in`) to warp the input video frames to the model's expected input size (`inp_width`, `inp_height`).
2.  Create an inverse transformation (`trans_inv`) to map the detected ball coordinates from the model's output space back to the original video's coordinate space.

### `affine_transform(pt, t)`

This function applies an affine transformation to a single point.

-   **`pt`**: The point to be transformed (a NumPy array of shape `(2,)`).
-   **`t`**: The 2x3 affine transformation matrix.

**Returns:** The transformed point (a NumPy array of shape `(2,)`).

This function is used by the `TracknetV2Postprocessor` to convert the ball coordinates found in the heatmap to their final positions in the original video frame.

### Other Utilities

The `utils/image.py` module also contains other useful functions, such as:

-   **`gaussian2D`**: Generates a 2D Gaussian kernel.
-   **`draw_umich_gaussian`**: Draws a Gaussian heatmap at a specified center point. This is likely used for creating the ground truth heatmaps during training.
-   **Color Augmentation Functions**: A set of functions for performing color augmentation, such as `brightness_`, `contrast_`, and `saturation_`.
