# pipeline_modules.py

import os
import cv2
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import logging

# WASB-SBDT modules
from .utils.image import get_affine_transform
from .dataloaders import img_transforms as T
from .models import build_model
from .detectors.postprocessor import TracknetV2Postprocessor

log = logging.getLogger(__name__)

class FramePreprocessor:
    """フレームシーケンスを前処理し、モデル入力用のテンソルを生成するクラス"""

    def __init__(self, cfg):
        self._frames_in = cfg['model']['frames_in']
        self._input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def frames_in(self):
        return self._frames_in

    def process_batch(self, frame_sequences):
        """
        複数のフレームシーケンスをバッチで前処理する。

        Args:
            frame_sequences (list of list of np.ndarray): 
                フレームシーケンスのリスト。例: [[frame1, frame2, ...], ...]

        Returns:
            tuple: (batch_tensor, batch_meta)
                - batch_tensor (torch.Tensor): モデル入力用のバッチテンソル
                - batch_meta (dict): 後処理に必要なメタデータ（アフィン逆変換行列など）
        """
        all_imgs_t = []
        all_trans_inv = []
        input_w, input_h = self._input_wh

        for frames in frame_sequences:
            assert len(frames) == self._frames_in, f"入力フレーム数が正しくありません。期待値: {self._frames_in}, 実際: {len(frames)}"
            
            ref_frame = frames[-1]
            trans_in = get_affine_transform(
                center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
                scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
                rot=0,
                output_size=[input_w, input_h],
            )
            
            trans_inv = get_affine_transform(
                center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
                scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
                rot=0,
                output_size=[input_w, input_h],
                inv=1,
            ),
            all_trans_inv.append(trans_inv)

            seq_imgs_t = []
            for frm in frames:
                frm_warp = cv2.warpAffine(frm, trans_in, (input_w, input_h), flags=cv2.INTER_LINEAR)
                img_pil = Image.fromarray(cv2.cvtColor(frm_warp, cv2.COLOR_BGR2RGB))
                seq_imgs_t.append(self._transform(img_pil))
            
            all_imgs_t.append(torch.cat(seq_imgs_t, dim=0))

        batch_tensor = torch.stack(all_imgs_t, dim=0)
        batch_meta = {'affine_mats_inv': all_trans_inv}
        
        return batch_tensor, batch_meta


class BallDetector:
    """モデルをロードし、ボール検出の推論を実行するクラス"""

    def __init__(self, cfg, device):
        self._device = device
        
        log.info("Building model...")
        self._model = build_model(cfg)
        
        model_path = cfg['detector']['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
        log.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
        self._model.load_state_dict(new_state_dict)
        self._model = self._model.to(device)
        self._model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_tensor):
        """
        バッチテンソルに対して推論を実行する。

        Args:
            batch_tensor (torch.Tensor): 前処理済みの入力テンソル

        Returns:
            torch.Tensor or dict: モデルの生の出力
        """
        batch_tensor = batch_tensor.to(self._device)
        device_type = 'cuda' if self._device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type=='cuda'):
            preds = self._model(batch_tensor)
        return preds


class DetectionPostprocessor:
    """モデルの出力を後処理し、検出結果を抽出するクラス"""

    def __init__(self, cfg):
        self._postprocessor = TracknetV2Postprocessor(cfg)
        self._frames_out = cfg['model']['frames_out']

    def process_batch(self, batch_preds, batch_meta, device):
        """
        モデルのバッチ出力とメタデータを後処理する。

        Args:
            batch_preds (torch.Tensor or dict): モデルの生のバッチ出力
            batch_meta (dict): 前処理時に生成されたメタデータ
            device (torch.device): 処理に使用するデバイス

        Returns:
            list of list of dict: 各シーケンスの検出結果のリスト
        """
        affine_mats_inv_list = batch_meta['affine_mats_inv']
        batch_size = len(affine_mats_inv_list)

        batch_affine_mats = {0: torch.from_numpy(np.stack(affine_mats_inv_list)).to(device)}
        
        if not isinstance(batch_preds, dict):
            batch_preds = {0: batch_preds}

        pp_results = self._postprocessor.run(batch_preds, batch_affine_mats)
        
        batch_detections = []
        for seq_idx in range(batch_size):
            detections = []
            if seq_idx in pp_results and (self._frames_out - 1) in pp_results[seq_idx]:
                last_frame_results = pp_results[seq_idx][self._frames_out - 1]
                if 0 in last_frame_results:
                    scale_results = last_frame_results[0]
                    for xy, score in zip(scale_results['xys'], scale_results['scores']):
                        detections.append({'xy': xy, 'score': score})
            batch_detections.append(detections)
                        
        return batch_detections