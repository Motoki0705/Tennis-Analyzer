# -*- coding: utf-8 -*-
import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from typing import Dict, List, Any, Optional, Tuple
import time
import traceback
from tqdm import tqdm
import os # For path joining

# Define COCO keypoint connections (indices based on standard COCO 17 keypoints)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 8], [7, 9],
    [8, 10], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 12],
    [7, 11], [11, 13], [12, 14], [13, 15], [14, 16], [5, 6], [11, 12] # Added missing connections for completeness
]
# Adjust indices for 0-based COCO format used by many models (subtract 1)
# Corrected based on common 0-16 index convention
COCO_SKELETON_ZERO_BASED = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (4, 6), (5, 6), (5, 11), (6, 12), (11, 13), (12, 14) # Shoulder-hip connections added
]

# --- NEW VideoAnnotator Class ---
class VideoAnnotator:
    """
    Loads pose estimation results and overlays them onto the original video.
    Can display the annotated video and save it to a file.
    """
    def __init__(
        self,
        pose_results: Dict[int, List[Dict[str, Any]]],
        bbox_color: Tuple[int, int, int] = (0, 255, 0),      # Green
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),  # Red
        skeleton_color: Tuple[int, int, int] = (255, 0, 0),  # Blue
        text_color: Tuple[int, int, int] = (255, 255, 255), # White
        line_thickness: int = 2,
        circle_radius: int = 4,
        font_scale: float = 0.5
        ):
        """
        Initializes the VideoAnnotator.

        Args:
            pose_results (Dict[int, List[Dict[str, Any]]]): The output dictionary from VideoPoseEstimator.
            bbox_color (Tuple[int, int, int]): BGR color for bounding boxes.
            keypoint_color (Tuple[int, int, int]): BGR color for keypoints.
            skeleton_color (Tuple[int, int, int]): BGR color for skeleton lines.
            text_color (Tuple[int, int, int]): BGR color for text labels (like confidence).
            line_thickness (int): Thickness for lines and bbox borders.
            circle_radius (int): Radius for keypoint circles.
            font_scale (float): Font scale for text.
        """
        self.pose_results = pose_results
        self.bbox_color = bbox_color
        self.keypoint_color = keypoint_color
        self.skeleton_color = skeleton_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.circle_radius = circle_radius
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale

    def _draw_annotations(self, frame: np.ndarray, frame_poses: List[Dict[str, Any]], kp_confidence_threshold: float):
        """Draws annotations for a single frame."""
        if not frame_poses:
            return frame # Return original frame if no poses for this frame number

        for person_data in frame_poses:
            bbox = person_data.get('bbox')
            keypoints = person_data.get('keypoints')

            # --- Draw Bounding Box ---
            if bbox and len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.bbox_color, self.line_thickness)
                # Optional: Add person ID label near bbox
                # person_id = person_data.get('person_id', '?')
                # cv2.putText(frame, f"ID:{person_id}", (x, y - 5), self.font, self.font_scale, self.text_color, 1)


            # --- Draw Keypoints and Skeleton ---
            if keypoints and len(keypoints) == 17: # Assuming 17 COCO keypoints
                valid_keypoints = {} # Store valid keypoints for drawing skeleton

                # Draw individual keypoints first
                for idx, kp in enumerate(keypoints):
                    x, y, conf = kp['x'], kp['y'], kp['confidence']
                    if conf >= kp_confidence_threshold:
                        pt = (int(x), int(y))
                        cv2.circle(frame, pt, self.circle_radius, self.keypoint_color, -1) # Filled circle
                        valid_keypoints[idx] = pt # Store valid point coordinates

                        # Optional: Draw confidence text near keypoint
                        # cv2.putText(frame, f"{conf:.2f}", (pt[0]+5, pt[1]), self.font, 0.4, self.text_color, 1)

                # Draw skeleton lines using valid keypoints
                for (idx1, idx2) in COCO_SKELETON_ZERO_BASED:
                    if idx1 in valid_keypoints and idx2 in valid_keypoints:
                        pt1 = valid_keypoints[idx1]
                        pt2 = valid_keypoints[idx2]
                        cv2.line(frame, pt1, pt2, self.skeleton_color, self.line_thickness)

        return frame

    def annotate_video(
        self,
        input_video_path: str,
        output_video_path: str,
        kp_confidence_threshold: float = 0.3,
        display_window: bool = True
        ):
        """
        Reads the input video, draws annotations based on pose results,
        saves the output video, and optionally displays it.

        Args:
            input_video_path (str): Path to the original input video file.
            output_video_path (str): Path where the annotated video will be saved.
            kp_confidence_threshold (float): Minimum confidence score to draw a keypoint/skeleton line.
            display_window (bool): If True, displays the annotated video frames during processing.
        """
        # --- Input Video ---
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open input video: {input_video_path}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Input video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames.")

        # --- Output Video ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not open output video writer: {output_video_path}")
            cap.release()
            return
        print(f"Outputting annotated video to: {output_video_path}")

        # --- Processing Loop ---
        frame_num = 0
        pbar = tqdm(total=total_frames, desc="Annotating Video", unit="frame")
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                # Get poses for the current frame (if they exist)
                poses_for_this_frame = self.pose_results.get(frame_num, [])

                # Draw annotations on the frame
                annotated_frame = self._draw_annotations(frame, poses_for_this_frame, kp_confidence_threshold)

                # Write frame to output video
                out.write(annotated_frame)

                # Display frame (optional)
                if display_window:
                    cv2.imshow('Annotated Video', annotated_frame)
                    # Press 'q' to quit display early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n'q' pressed, stopping display and continuing saving...")
                        display_window = False # Stop showing window but continue processing
                        cv2.destroyWindow('Annotated Video') # Close the specific window

                frame_num += 1
                pbar.update(1)

        except Exception as e:
            pbar.write(f"\nError during annotation: {e}\n{traceback.format_exc()}")
        finally:
            # --- Cleanup ---
            pbar.close()
            cap.release()
            out.release()
            if display_window: # Ensure window is destroyed if it was ever opened
                cv2.destroyAllWindows()
            end_time = time.time()
            print(f"Annotation finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    pose_results_path = "pose_results.json"
    input_vid_path = "test_video_1.mp4"
    output_vid_path = "output_video_1.mp4"

    with open(pose_results_path, "r") as f:
        raw_results = json.load(f)

    pose_results = {int(k): v for k, v in raw_results.items()}
    # --- 3. Run Annotation ---
    if pose_results:
        print("\n--- Running Video Annotation ---")
        try:
            annotator = VideoAnnotator(pose_results)
            annotator.annotate_video(
                input_video_path=input_vid_path,
                output_video_path=output_vid_path,
                kp_confidence_threshold=0.3, # Only draw keypoints/lines above this confidence
                display_window=True          # Set to False to only save without showing
            )
            print(f"\nAnnotation complete. Output saved to {output_vid_path}")
        except Exception as e:
            print(f"\n--- Annotation Failed ---")
            print(traceback.format_exc())
    elif not pose_results:
         print("\nSkipping annotation because pose estimation results are empty or failed.")
    else:
         print("\nSkipping annotation step as results were not generated/loaded.")