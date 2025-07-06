import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import logging

# WASB-SBDT modules - 改善されたAPIを使用
from . import create_ball_detector, load_default_config
from .utils.image import get_affine_transform
from .dataloaders import img_transforms as T

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SimpleDetector:
    """Simplified detector with batch processing support"""
    
    def __init__(self, cfg, device):
        self._frames_in = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out'] 
        self._input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        self._device = device
        
        # Build model using improved API
        from .models import build_model
        self._model = build_model(cfg)
        
        # Load weights
        model_path = cfg['detector']['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        log.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self._model.load_state_dict(new_state_dict)
        self._model = self._model.to(device)
        self._model.eval()
        
        # Image transforms
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Postprocessor
        from .detectors.postprocessor import TracknetV2Postprocessor
        self._postprocessor = TracknetV2Postprocessor(cfg)
        
    @property
    def frames_in(self):
        return self._frames_in
        
    @property
    def input_wh(self):
        return self._input_wh
    
    def process_frames(self, frames):
        """Process single frame sequence and return ball detections"""
        assert len(frames) == self._frames_in
        
        # Prepare input tensor
        imgs_t = []
        input_w, input_h = self._input_wh
        
        # Use the last frame for transform reference
        ref_frame = frames[-1]
        trans_in = get_affine_transform(
            center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
            scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
            rot=0,
            output_size=[input_w, input_h],
        )
        
        # Inverse transform for output mapping
        trans_inv = get_affine_transform(
            center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
            scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
            rot=0,
            output_size=[input_w, input_h],
            inv=1,
        )
        
        for frm in frames:
            frm_warp = cv2.warpAffine(frm, trans_in, (input_w, input_h), flags=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(cv2.cvtColor(frm_warp, cv2.COLOR_BGR2RGB))
            imgs_t.append(self._transform(img_pil))
            
        imgs_tensor = torch.cat(imgs_t, dim=0).unsqueeze(0).to(self._device)
        
        # Forward pass
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = self._model(imgs_tensor)
            
        # Post-process
        affine_mats = {0: torch.from_numpy(trans_inv).unsqueeze(0).to(self._device)}
        pp_results = self._postprocessor.run(preds, affine_mats)
        
        # Extract detections for the last frame (current frame)
        detections = []
        if 0 in pp_results and (self._frames_out - 1) in pp_results[0]:
            last_frame_results = pp_results[0][self._frames_out - 1]
            if 0 in last_frame_results:
                scale_results = last_frame_results[0]
                for xy, score in zip(scale_results['xys'], scale_results['scores']):
                    detections.append({'xy': xy, 'score': score})
                    
        return detections
    
    def process_frames_batch(self, frame_sequences):
        """Process multiple frame sequences in batch and return ball detections"""
        batch_size = len(frame_sequences)
        if batch_size == 0:
            return []
        
        # Validate all sequences have correct length
        for i, frames in enumerate(frame_sequences):
            assert len(frames) == self._frames_in, f"Sequence {i} has {len(frames)} frames, expected {self._frames_in}"
        
        # Prepare batch input tensors
        all_imgs_t = []
        all_trans_inv = []
        input_w, input_h = self._input_wh
        
        for seq_idx, frames in enumerate(frame_sequences):
            # Use the last frame for transform reference
            ref_frame = frames[-1]
            trans_in = get_affine_transform(
                center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
                scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
                rot=0,
                output_size=[input_w, input_h],
            )
            
            # Inverse transform for output mapping
            trans_inv = get_affine_transform(
                center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
                scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
                rot=0,
                output_size=[input_w, input_h],
                inv=1,
            )
            all_trans_inv.append(trans_inv)
            
            # Process frames in this sequence
            seq_imgs_t = []
            for frm in frames:
                frm_warp = cv2.warpAffine(frm, trans_in, (input_w, input_h), flags=cv2.INTER_LINEAR)
                img_pil = Image.fromarray(cv2.cvtColor(frm_warp, cv2.COLOR_BGR2RGB))
                seq_imgs_t.append(self._transform(img_pil))
            
            all_imgs_t.append(torch.cat(seq_imgs_t, dim=0))
        
        # Stack all sequences into batch tensor
        batch_tensor = torch.stack(all_imgs_t, dim=0).to(self._device)
        
        # Forward pass
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                batch_preds = self._model(batch_tensor)
        
        # Post-process the entire batch at once
        # Prepare affine matrices for all sequences in batch format (scale-based structure)
        batch_affine_mats = {0: torch.stack([torch.from_numpy(trans) for trans in all_trans_inv], dim=0).to(self._device)}
        
        # Handle model output structure - check if it's a dict or tensor
        if isinstance(batch_preds, dict):
            # Model outputs multiple scales as dict
            batch_scale_preds = batch_preds
        else:
            # Model outputs single tensor, wrap in scale-based structure
            batch_scale_preds = {0: batch_preds}
        
        # Post-process entire batch
        pp_results = self._postprocessor.run(batch_scale_preds, batch_affine_mats)
        
        # Extract detections for each sequence in the batch
        batch_detections = []
        for seq_idx in range(batch_size):
            # Extract detections for the last frame (current frame) of this sequence
            detections = []
            if seq_idx in pp_results and (self._frames_out - 1) in pp_results[seq_idx]:
                last_frame_results = pp_results[seq_idx][self._frames_out - 1]
                if 0 in last_frame_results:
                    scale_results = last_frame_results[0]
                    for xy, score in zip(scale_results['xys'], scale_results['scores']):
                        detections.append({'xy': xy, 'score': score})
            
            batch_detections.append(detections)
                        
        return batch_detections


def main():
    parser = argparse.ArgumentParser(description="WASB-SBDT テニスボール追跡デモ（バッチ推論対応）")
    parser.add_argument("--video", required=True, help="入力動画へのパス")
    parser.add_argument("--output", default="demo_output.mp4", help="出力動画ファイル")
    parser.add_argument("--model_path", required=True, help="学習済みモデルの .pth.tar または .pth ファイル")
    parser.add_argument("--device", default="auto", help="デバイス (cuda/cpu/auto)")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ（同時処理するフレーム数）")
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    log.info(f"Using device: {device}")
    log.info(f"Batch size: {args.batch_size}")

    # Load config using improved API
    cfg = load_default_config()
    cfg.detector.model_path = args.model_path

    # Build detector and tracker using improved API
    detector = SimpleDetector(cfg, device)
    from .trackers import build_tracker
    tracker = build_tracker(cfg)
    
    frames_in = detector.frames_in
    log.info(f"Model expects {frames_in} input frames")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log.info(f"Video: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Processing with batch support
    all_frames = []
    frame_buffers = []
    frame_idx = 0
    tracker.refresh()
    
    log.info("Processing frames...")
    
    # Read all frames first
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    log.info(f"Loaded {len(all_frames)} frames")
    
    # Process frames with batch support
    processed_frames = []
    batch_sequences = []
    
    for frame_idx, frame in enumerate(all_frames):
        # Build frame buffer for current position
        start_idx = max(0, frame_idx - frames_in + 1)
        current_buffer = []
        
        # Pad with first frame if needed
        for i in range(frames_in):
            buffer_idx = start_idx + i
            if buffer_idx < 0:
                current_buffer.append(all_frames[0])
            elif buffer_idx < len(all_frames):
                current_buffer.append(all_frames[buffer_idx])
            else:
                current_buffer.append(all_frames[-1])
        
        batch_sequences.append(current_buffer)
        
        # Process batch when full or at end
        if len(batch_sequences) == args.batch_size or frame_idx == len(all_frames) - 1:
            # Run batch detection
            batch_detections = detector.process_frames_batch(batch_sequences)
            
            # Update tracker for each detection in batch
            for seq_idx, detections in enumerate(batch_detections):
                tracking_output = tracker.update(detections)
                
                # Get the current frame for this sequence
                current_frame_idx = frame_idx - len(batch_sequences) + seq_idx + 1
                current_frame = all_frames[current_frame_idx].copy()
                
                # Visualize result
                if tracking_output.get("visi", False) and tracking_output.get("score", 0) > 0.1:
                    px, py = int(tracking_output["x"]), int(tracking_output["y"])
                    # Draw red circle for ball
                    cv2.circle(current_frame, (px, py), 8, (0, 0, 255), -1)
                    # Draw smaller white center
                    cv2.circle(current_frame, (px, py), 3, (255, 255, 255), -1)
                    
                    # Add score text
                    score_text = f"Score: {tracking_output['score']:.2f}"
                    cv2.putText(current_frame, score_text, (px + 15, py - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                processed_frames.append(current_frame)
            
            # Clear batch
            batch_sequences = []
            
            if (frame_idx + 1) % 100 == 0:
                log.info(f"Processed {frame_idx + 1}/{len(all_frames)} frames")
    
    # Write all processed frames
    log.info("Writing output video...")
    for frame in processed_frames:
        writer.write(frame)
    
    writer.release()
    log.info(f"出力動画を保存しました: {args.output}")


if __name__ == "__main__":
    main() 