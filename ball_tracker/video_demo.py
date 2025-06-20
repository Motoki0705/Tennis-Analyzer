import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import logging

# ball_tracker modules (with fallback)
try:
    from .models import build_model
    from .online import OnlineTracker
    from .postprocessor import TracknetV2Postprocessor
except ImportError:
    # Fallback for direct execution
    from models import build_model
    from online import OnlineTracker
    from postprocessor import TracknetV2Postprocessor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """Simple affine transform implementation"""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    """Get direction"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    """Get 3rd point"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


class SimpleDetector:
    """Simplified detector without complex dependencies"""
    
    def __init__(self, cfg, device):
        self._frames_in = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out'] 
        self._input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        self._device = device
        
        # Build model
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
        
        # Postprocessor
        self._postprocessor = TracknetV2Postprocessor(cfg)
        
    @property
    def frames_in(self):
        return self._frames_in
        
    @property
    def input_wh(self):
        return self._input_wh
    
    def process_frames(self, frames):
        """Process frames and return ball detections"""
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
            # Manual transform: BGR to RGB, normalize, to tensor
            frm_rgb = cv2.cvtColor(frm_warp, cv2.COLOR_BGR2RGB)
            frm_norm = frm_rgb.astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            frm_norm = (frm_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            # Convert to tensor [C, H, W]
            frm_tensor = torch.from_numpy(frm_norm.transpose(2, 0, 1))
            imgs_t.append(frm_tensor)
            
        imgs_tensor = torch.cat(imgs_t, dim=0).unsqueeze(0).to(self._device)
        
        # Forward pass
        with torch.no_grad():
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


def load_simple_config():
    """Load a simple config without Hydra"""
    cfg = {
        'model': {
            'name': 'hrnet',
            'frames_in': 3,
            'frames_out': 3,
            'inp_height': 288,
            'inp_width': 512,
            'out_height': 288, 
            'out_width': 512,
            'rgb_diff': False,
            'out_scales': [0],
            'MODEL': {
                'EXTRA': {
                    'FINAL_CONV_KERNEL': 1,
                    'PRETRAINED_LAYERS': ['*'],
                    'STEM': {'INPLANES': 64, 'STRIDES': [1,1]},
                    'STAGE1': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [1], 'NUM_CHANNELS': [32], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE2': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2,2], 'NUM_CHANNELS': [16,32], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 
                        'NUM_BLOCKS': [2,2,2], 'NUM_CHANNELS': [16,32,64], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2,2,2,2], 'NUM_CHANNELS': [16,32,64,128], 'FUSE_METHOD': 'SUM'
                    },
                    'DECONV': {'NUM_DECONVS': 0, 'KERNEL_SIZE': [], 'NUM_BASIC_BLOCKS': 2}
                },
                'INIT_WEIGHTS': True
            }
        },
        'detector': {
            'model_path': None,
            'postprocessor': {
                'name': 'tracknetv2',
                'score_threshold': 0.5,
                'scales': [0],
                'blob_det_method': 'concomp',
                'use_hm_weight': True
            }
        },
        'tracker': {
            'name': 'online',
            'max_disp': 100
        },
        'dataloader': {
            'heatmap': {
                'sigmas': {0: 2.0}
            }
        }
    }
    return OmegaConf.create(cfg)


def main():
    parser = argparse.ArgumentParser(description="WASB-SBDT テニスボール追跡デモ")
    parser.add_argument("--video", required=True, help="入力動画へのパス")
    parser.add_argument("--output", default="demo_output.mp4", help="出力動画ファイル")
    parser.add_argument("--model_path", required=True, help="学習済みモデルの .pth.tar または .pth ファイル")
    parser.add_argument("--device", default="auto", help="デバイス (cuda/cpu/auto)")
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    log.info(f"Using device: {device}")

    # Load config
    cfg = load_simple_config()
    cfg.detector.model_path = args.model_path

    # Build detector and tracker
    detector = SimpleDetector(cfg, device)
    tracker = OnlineTracker(cfg)
    
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

    # Processing
    buffer = []
    frame_idx = 0
    tracker.refresh()
    
    log.info("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        buffer.append(frame.copy())
        
        # Need enough frames for detection
        if len(buffer) < frames_in:
            # Write frame without detection
            writer.write(frame)
            frame_idx += 1
            continue
            
        # Keep only the required number of frames
        if len(buffer) > frames_in:
            buffer.pop(0)
            
        # Run detection on current frame window
        detections = detector.process_frames(buffer)
        
        # Update tracker
        tracking_output = tracker.update(detections)
        
        # Visualize result on the current frame (last in buffer)
        current_frame = buffer[-1].copy()
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
        
        writer.write(current_frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            log.info(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    log.info(f"出力動画を保存しました: {args.output}")


if __name__ == "__main__":
    main() 