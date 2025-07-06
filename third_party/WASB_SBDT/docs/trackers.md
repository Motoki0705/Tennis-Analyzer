
# Trackers

This document describes the tracking component of the WASB-SBDT package.

## `build_tracker(cfg)`

This function is the entry point for creating a tracker instance. It reads the tracker configuration from the provided `cfg` object and instantiates the corresponding tracker class.

-   **`cfg`**: An `OmegaConf` object containing the tracker configuration. The `cfg.tracker.name` field is used to determine which tracker to build.

**Returns:** A tracker instance ready for ball tracking.

## Supported Trackers

The `build_tracker` function supports the following trackers through a factory pattern:

-   **`online`**: (Default) An online tracker that uses a simple yet effective algorithm to track the ball across frames. It maintains the state of the ball (position and visibility) and updates it based on the detections in the current frame.
-   **`intra_frame_peak`**: A tracker that focuses on selecting the best detection within a single frame, without considering temporal information.

## `OnlineTracker`

The `OnlineTracker` is the default tracker used in `video_demo.py`. It implements a simple tracking logic:

1.  **Prediction**: Predict the ball's position in the current frame based on its position in the previous frame.
2.  **Association**: Find the detection in the current frame that is closest to the predicted position.
3.  **Update**: If a close detection is found, update the ball's state with the new position. Otherwise, mark the ball as not visible.

### Key Methods

-   **`update(detections)`**: Updates the tracker with the detections from the current frame. `detections` is a list of dictionaries, where each dictionary contains the `xy` coordinates and `score` of a detection.
-   **`refresh()`**: Resets the tracker's state. This should be called at the beginning of each new video sequence.

## Configuration

The behavior of the tracker is controlled by the `tracker` section of the configuration file. Key parameters for the `OnlineTracker` include:

-   **`name`**: The name of the tracker to use (`'online'`).
-   **`max_disp`**: The maximum displacement (in pixels) allowed for a ball between consecutive frames. This is used to constrain the search for the ball in the current frame.
