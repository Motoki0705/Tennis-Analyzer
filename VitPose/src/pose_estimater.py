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
from tqdm import tqdm # Import tqdm

class VideoPoseEstimator:
    """
    Estimates human poses in videos using Hugging Face transformer models.

    Detects humans in each frame (or skipped frames) using RT-DETR,
    then runs batched pose estimation using ViTPose on detected persons.
    Outputs keypoints and bounding boxes for each detected person.
    Includes a progress bar for video processing.
    """

    def __init__(
        self,
        person_detector_model_name: str = "PekingU/rtdetr_r50vd_coco_o365",
        pose_estimator_model_name: str = "usyd-community/vitpose-base-simple",
        device: Optional[str] = None,
    ):
        # ... (Initialization remains the same) ...
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        try:
            print(f"Loading person detector: {person_detector_model_name}...")
            self.person_processor = AutoProcessor.from_pretrained(person_detector_model_name)
            self.person_model = RTDetrForObjectDetection.from_pretrained(person_detector_model_name).to(self.device)
            self.person_model.eval()
            print("Person detector loaded.")

            print(f"Loading pose estimator: {pose_estimator_model_name}...")
            self.pose_processor = AutoProcessor.from_pretrained(pose_estimator_model_name)
            self.pose_model = VitPoseForPoseEstimation.from_pretrained(pose_estimator_model_name).to(self.device)
            self.pose_model.eval()
            print("Pose estimator loaded.")

        except Exception as e:
            print(f"Error loading models: {e}")
            raise

        self.person_label_id = 0
        if hasattr(self.person_model.config, 'label2id'):
            person_labels = [label for label, id_ in self.person_model.config.label2id.items() if 'person' in label.lower()]
            if person_labels:
                 self.person_label_id = self.person_model.config.label2id[person_labels[0]]
                 print(f"Auto-detected 'person' label ID: {self.person_label_id}")
            else:
                 print(f"Warning: Could not auto-detect 'person' label ID. Assuming {self.person_label_id}.")
        else:
             print(f"Warning: Model config has no label2id. Assuming 'person' label ID is {self.person_label_id}.")


    def _detect_persons(self, image: Image.Image, threshold: float) -> np.ndarray:
        # ... (This method remains the same) ...
        inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        with torch.no_grad(): outputs = self.person_model(**inputs)
        results = self.person_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
        result = results[0]
        person_indices = result["labels"] == self.person_label_id
        person_boxes_voc = result["boxes"][person_indices].cpu().numpy()
        if person_boxes_voc.shape[0] == 0: return np.empty((0, 4), dtype=np.float32)
        person_boxes_coco = person_boxes_voc.copy()
        person_boxes_coco[:, 2] -= person_boxes_coco[:, 0]; person_boxes_coco[:, 3] -= person_boxes_coco[:, 1]
        return person_boxes_coco

    def _parse_pose_results(
        self,
        pose_results_raw_batch: List[Dict[str, Any]],
        input_boxes_batch: List[np.ndarray]
        ) -> List[List[Dict[str, Any]]]:
        # ... (This method remains the same) ...
        batch_parsed_poses = []
        min_len = len(pose_results_raw_batch) # Assume correct length initially
        if len(pose_results_raw_batch) != len(input_boxes_batch):
             print(f"\nWarning: Mismatch between pose results count ({len(pose_results_raw_batch)}) and input boxes count ({len(input_boxes_batch)}). Parsing may be incomplete.")
             min_len = min(len(pose_results_raw_batch), len(input_boxes_batch))

        for img_idx in range(min_len):
            image_results = pose_results_raw_batch[img_idx]
            input_boxes_for_image = input_boxes_batch[img_idx]
            image_parsed_poses = []
            persons_data = []
            if isinstance(image_results, list): persons_data = image_results
            elif isinstance(image_results, dict) and 'keypoints' in image_results and 'scores' in image_results: persons_data = [image_results]

            num_persons_to_process = len(persons_data)
            if len(persons_data) != input_boxes_for_image.shape[0]:
                 print(f"\nWarning: Image {img_idx}: Parsed poses ({len(persons_data)}) != input boxes ({input_boxes_for_image.shape[0]}). Association might be incorrect.")
                 num_persons_to_process = min(len(persons_data), input_boxes_for_image.shape[0])

            for person_idx in range(num_persons_to_process):
                person_data = persons_data[person_idx]; input_bbox = input_boxes_for_image[person_idx]
                if "keypoints" in person_data and "scores" in person_data:
                    keypoints_tensor = person_data["keypoints"]; scores_tensor = person_data["scores"]
                    if isinstance(keypoints_tensor, torch.Tensor): keypoints_tensor = keypoints_tensor.cpu().numpy()
                    if isinstance(scores_tensor, torch.Tensor): scores_tensor = scores_tensor.cpu().numpy()
                    if keypoints_tensor.ndim == 3 and keypoints_tensor.shape[0] == 1: keypoints_tensor = keypoints_tensor.squeeze(0)
                    if scores_tensor.ndim == 2 and scores_tensor.shape[0] == 1: scores_tensor = scores_tensor.squeeze(0)

                    if keypoints_tensor.ndim == 2 and scores_tensor.ndim == 1:
                         keypoints = [{"x": float(kp[0]), "y": float(kp[1]), "confidence": float(score)} for kp, score in zip(keypoints_tensor, scores_tensor)]
                         image_parsed_poses.append({"person_id": person_idx, "keypoints": keypoints, "bbox": input_bbox.astype(float).tolist()})
                    else: print(f"\nWarning: Img {img_idx}, Person {person_idx}: Unexpected tensor shapes kp={keypoints_tensor.shape}, score={scores_tensor.shape}")
                else: print(f"\nWarning: Img {img_idx}, Person {person_idx}: Missing keys in {person_data.keys()}")
            batch_parsed_poses.append(image_parsed_poses)

        for _ in range(len(input_boxes_batch) - min_len): batch_parsed_poses.append([]) # Add placeholders if needed
        return batch_parsed_poses


    def _estimate_poses_batch(
        self,
        image_batch: List[Image.Image],
        person_boxes_batch: List[np.ndarray]
        ) -> List[List[Dict[str, Any]]]:
        # ... (This method remains the same) ...
        if not image_batch: return []
        boxes_for_processor = [boxes.tolist() for boxes in person_boxes_batch]
        inputs = self.pose_processor(images=image_batch, boxes=boxes_for_processor, return_tensors="pt").to(self.device)
        with torch.no_grad(): outputs = self.pose_model(**inputs)
        pose_results_raw_batch = self.pose_processor.post_process_pose_estimation(outputs, boxes=boxes_for_processor)
        parsed_poses_batch = self._parse_pose_results(pose_results_raw_batch, person_boxes_batch)
        return parsed_poses_batch


    def _process_frame_for_detection(
        self,
        frame: np.ndarray,
        threshold: float
    ) -> Tuple[Optional[Image.Image], np.ndarray]:
        # ... (This method remains the same) ...
        if frame is None: return None, np.empty((0, 4), dtype=np.float32)
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); pil_image = Image.fromarray(rgb_frame)
            person_boxes_coco = self._detect_persons(pil_image, threshold=threshold)
            return pil_image, person_boxes_coco
        except Exception as e:
            print(f"\nError processing frame for detection: {e}")
            return None, np.empty((0, 4), dtype=np.float32)


    def predict(
        self,
        video_path: str,
        frame_skip: int = 1,
        person_detection_threshold: float = 0.5,
        pose_batch_size: int = 8
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Processes a video file to detect persons and estimate their poses using batching.

        Args:
            video_path (str): Path to the input video file.
            frame_skip (int): Process every 'frame_skip' frames.
            person_detection_threshold (float): Confidence threshold for person detection.
            pose_batch_size (int): Number of frames (with detected persons) to batch for pose estimation.

        Returns:
            Dict[int, List[Dict[str, Any]]]: Dictionary mapping frame numbers to lists of
            detected persons, each with 'person_id', 'keypoints', and 'bbox' [x, y, w, h].
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return {}

        all_results: Dict[int, List[Dict[str, Any]]] = {}
        frame_number = 0
        batch_data_for_pose = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
             print("Warning: Could not determine total frames. Progress bar may be inaccurate.")
             total_processed_frames = None # Indeterminate progress bar
        else:
             # Calculate the number of frames that will actually be processed after skipping
             total_processed_frames = (total_frames + frame_skip - 1) // frame_skip
             print(f"Video contains approx. {total_frames} frames. Will process approx. {total_processed_frames} frames.")

        if frame_skip > 1: print(f"Processing every {frame_skip} frames.")
        print(f"Using pose estimation batch size: {pose_batch_size}")

        # Initialize tqdm progress bar
        pbar = tqdm(total=total_processed_frames, desc="Processing Video", unit="frame")
        processing_start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                current_frame_num = frame_number
                frame_number += 1

                # --- Frame Skipping ---
                if current_frame_num % frame_skip != 0:
                    continue

                # --- Update Progress Bar ---
                # Update *after* skipping, so it reflects processed frames
                if pbar is not None: pbar.update(1)

                # --- Person Detection ---
                pil_image, person_boxes = self._process_frame_for_detection(
                    frame, threshold=person_detection_threshold
                )

                if pil_image is None: continue # Skip if frame processing failed

                # --- Store results / Prepare Batch ---
                if person_boxes.shape[0] > 0:
                    batch_data_for_pose.append((current_frame_num, pil_image, person_boxes))
                else:
                    all_results[current_frame_num] = []

                # --- Process Pose Estimation Batch ---
                if len(batch_data_for_pose) >= pose_batch_size:
                    # Use pbar.write to print messages without messing up the bar
                    pbar.write(f"Processing pose estimation batch (frames up to {current_frame_num})...")
                    frame_nums_in_batch = [data[0] for data in batch_data_for_pose]
                    images_in_batch = [data[1] for data in batch_data_for_pose]
                    boxes_in_batch = [data[2] for data in batch_data_for_pose]

                    batch_start_time = time.time()
                    parsed_poses_batch = self._estimate_poses_batch(images_in_batch, boxes_in_batch)
                    batch_end_time = time.time()
                    pbar.write(f"  Batch processed {len(images_in_batch)} images in {batch_end_time - batch_start_time:.2f} seconds.")

                    if len(parsed_poses_batch) == len(frame_nums_in_batch):
                        for i, frame_num in enumerate(frame_nums_in_batch):
                            all_results[frame_num] = parsed_poses_batch[i]
                    else:
                         pbar.write(f"Warning: Mismatch frames ({len(frame_nums_in_batch)}) vs results ({len(parsed_poses_batch)}) in batch.")

                    batch_data_for_pose = [] # Clear batch

            # --- Process the last partial batch ---
            if batch_data_for_pose:
                pbar.write(f"Processing final pose estimation batch ({len(batch_data_for_pose)} frames)...")
                frame_nums_in_batch = [data[0] for data in batch_data_for_pose]
                images_in_batch = [data[1] for data in batch_data_for_pose]
                boxes_in_batch = [data[2] for data in batch_data_for_pose]

                batch_start_time = time.time()
                parsed_poses_batch = self._estimate_poses_batch(images_in_batch, boxes_in_batch)
                batch_end_time = time.time()
                pbar.write(f"  Final batch processed {len(images_in_batch)} images in {batch_end_time - batch_start_time:.2f} seconds.")

                if len(parsed_poses_batch) == len(frame_nums_in_batch):
                    for i, frame_num in enumerate(frame_nums_in_batch):
                        all_results[frame_num] = parsed_poses_batch[i]
                else:
                     pbar.write(f"Warning: Mismatch frames ({len(frame_nums_in_batch)}) vs results ({len(parsed_poses_batch)}) in final batch.")

        except Exception as e:
            # Try to print error using pbar.write if available
            error_message = f"\n--- An error occurred during video processing near frame {frame_number} ---"
            if pbar: pbar.write(error_message)
            else: print(error_message)
            detailed_error = traceback.format_exc()
            if pbar: pbar.write(detailed_error)
            else: print(detailed_error)
        finally:
            # Ensure the progress bar is closed upon exit or error
            if pbar: pbar.close()
            cap.release()
            processing_end_time = time.time()
            print(f"Video processing finished in {processing_end_time - processing_start_time:.2f} seconds.")

        return dict(sorted(all_results.items()))
    
if __name__ == '__main__':
    real_video_path = r"data/raw/test_video_1.mp4"
    estimator = VideoPoseEstimator()
    results = estimator.predict(real_video_path, frame_skip=2) # Process all frames
    with open(r"outputs/pose/pose_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(results)