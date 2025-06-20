"""
WASB-SBDT ボールトラッキングシステム

Usage:
    from ball_tracker import BallTracker
    
    tracker = BallTracker(model_path="weights.pth.tar")
    results = tracker.track_video("input.mp4", "output.mp4")
"""

from .video_demo import SimpleDetector
from .online import OnlineTracker

__version__ = "1.0.0"
__all__ = ["SimpleDetector", "OnlineTracker", "BallTracker"]

class BallTracker:
    """統合ボールトラッキングクラス"""
    
    def __init__(self, model_path, device='auto'):
        """
        Args:
            model_path (str): 学習済みモデルファイルパス
            device (str): 'cuda', 'cpu', or 'auto'
        """
        import torch
        from omegaconf import OmegaConf
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 簡易設定作成
        cfg = self._create_config()
        cfg.detector.model_path = model_path
        
        self.detector = SimpleDetector(cfg, torch.device(device))
        self.tracker = OnlineTracker(cfg)
        self.tracker.refresh()
        
    def _create_config(self):
        """簡易設定を作成"""
        cfg = {
            'model': {
                'name': 'hrnet', 'frames_in': 3, 'frames_out': 3,
                'inp_height': 288, 'inp_width': 512, 'out_height': 288, 'out_width': 512,
                'rgb_diff': False, 'out_scales': [0],
                'MODEL': {
                    'EXTRA': {
                        'FINAL_CONV_KERNEL': 1, 'PRETRAINED_LAYERS': ['*'],
                        'STEM': {'INPLANES': 64, 'STRIDES': [1,1]},
                        'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK',
                                  'NUM_BLOCKS': [1], 'NUM_CHANNELS': [32], 'FUSE_METHOD': 'SUM'},
                        'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',
                                  'NUM_BLOCKS': [2,2], 'NUM_CHANNELS': [16,32], 'FUSE_METHOD': 'SUM'},
                        'STAGE3': {'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',
                                  'NUM_BLOCKS': [2,2,2], 'NUM_CHANNELS': [16,32,64], 'FUSE_METHOD': 'SUM'},
                        'STAGE4': {'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
                                  'NUM_BLOCKS': [2,2,2,2], 'NUM_CHANNELS': [16,32,64,128], 'FUSE_METHOD': 'SUM'},
                        'DECONV': {'NUM_DECONVS': 0, 'KERNEL_SIZE': [], 'NUM_BASIC_BLOCKS': 2}
                    },
                    'INIT_WEIGHTS': True
                }
            },
            'detector': {
                'model_path': None,
                'postprocessor': {
                    'name': 'tracknetv2', 'score_threshold': 0.5, 'scales': [0],
                    'blob_det_method': 'concomp', 'use_hm_weight': True
                }
            },
            'tracker': {'name': 'online', 'max_disp': 100},
            'dataloader': {'heatmap': {'sigmas': {0: 2.0}}}
        }
        return OmegaConf.create(cfg)
    
    def track_video(self, video_path, output_path=None, visualize=True):
        """
        動画ファイルを処理してボールを追跡
        
        Args:
            video_path (str): 入力動画パス
            output_path (str): 出力動画パス（None の場合は結果のみ返す）
            visualize (bool): 可視化するかどうか
            
        Returns:
            list: フレームごとのボール位置 [{'frame': int, 'x': float, 'y': float, 'visible': bool}, ...]
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        
        results = []
        buffer = []
        frame_idx = 0
        
        # 出力動画設定
        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            buffer.append(frame.copy())
            
            if len(buffer) < self.detector.frames_in:
                if writer:
                    writer.write(frame)
                results.append({'frame': frame_idx, 'x': -1, 'y': -1, 'visible': False})
                frame_idx += 1
                continue
                
            if len(buffer) > self.detector.frames_in:
                buffer.pop(0)
                
            # ボール検出・追跡
            detections = self.detector.process_frames(buffer)
            tracking_output = self.tracker.update(detections)
            
            # 結果記録
            result = {
                'frame': frame_idx,
                'x': tracking_output.get('x', -1),
                'y': tracking_output.get('y', -1),
                'visible': tracking_output.get('visi', False),
                'score': tracking_output.get('score', 0.0)
            }
            results.append(result)
            
            # 可視化
            current_frame = buffer[-1].copy()
            if visualize and result['visible'] and result['score'] > 0.1:
                px, py = int(result['x']), int(result['y'])
                cv2.circle(current_frame, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(current_frame, (px, py), 3, (255, 255, 255), -1)
                
            if writer:
                writer.write(current_frame)
            frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
            
        return results
    
    def track_frames(self, frames):
        """
        フレーム配列を処理してボール位置を検出
        
        Args:
            frames (list): BGRフレームのリスト
            
        Returns:
            dict: {'x': float, 'y': float, 'visible': bool, 'score': float}
        """
        if len(frames) < self.detector.frames_in:
            return {'x': -1, 'y': -1, 'visible': False, 'score': 0.0}
            
        detections = self.detector.process_frames(frames[-self.detector.frames_in:])
        return self.tracker.update(detections)
