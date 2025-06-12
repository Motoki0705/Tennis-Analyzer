import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import logging

# from ..event.lit_module.lit_transformer_v2 import LitTransformerV2

logger = logging.getLogger(__name__)


class EventPredictor:
    """
    イベント検知（バウンド・ショット）のための予測器クラス。
    
    提供されたモデルインスタンスを使用してテニスの試合におけるバウンドとショットイベントを検知します。
    """

    def __init__(
        self,
        litmodule,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        smoothing_window: int = 5,
        debug: bool = False,
        use_half: bool = False
    ):
        """
        Args:
            litmodule: 初期化済みのモデルインスタンス（例: LitTransformerV2）
            device: 推論デバイス ("cpu" or "cuda")
            confidence_threshold: イベント検知の閾値
            smoothing_window: 信号平滑化のウィンドウサイズ
            debug: デバッグモード
            use_half: 推論に半精度(FP16)を使用するかどうか
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.debug = debug
        self.use_half = use_half
        
        # モデルの読み込み
        self.model = litmodule.to(self.device)
        self.model.eval()
        
        # 信号履歴を保持するバッファ
        self.signal_history: List[Tuple[float, float]] = []  # (hit_prob, bounce_prob)
        self.max_history_length = 60  # 約2秒分の履歴（30fps想定）

    def preprocess(self, combined_features: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Any]:
        """
        統合された特徴量を前処理します。
        
        Args:
            combined_features: 統合された特徴量辞書
            
        Returns:
            Tuple[Dict[str, torch.Tensor], Any]: 前処理されたテンソルとメタデータ
        """
        try:
            # 既にEventWorkerで前処理済みの特徴量を受け取る想定
            processed_data = {}
            
            for key, tensor in combined_features.items():
                if isinstance(tensor, torch.Tensor):
                    processed_data[key] = tensor.to(self.device)
                else:
                    # numpyやリストの場合はtensorに変換
                    processed_data[key] = torch.tensor(tensor, dtype=torch.float32).to(self.device)
            
            return processed_data, None
            
        except Exception as e:
            logger.error(f"前処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise

    def inference(self, tensor_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        推論を実行します。
        
        Args:
            tensor_data: 前処理済みのテンソルデータ
            
        Returns:
            torch.Tensor: 推論結果のlogits [B, T, 2]
        """
        try:
            with torch.no_grad():
                if self.use_half and self.device.type == 'cuda':
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        ball_features = tensor_data['ball_features']
                        player_bbox_features = tensor_data['player_bbox_features']
                        player_pose_features = tensor_data['player_pose_features']
                        court_features = tensor_data['court_features']
                        
                        # モデルの forward メソッドを呼び出し
                        logits = self.model(
                            ball_features=ball_features,
                            player_bbox_features=player_bbox_features,
                            player_pose_features=player_pose_features,
                            court_features=court_features
                        )
                else:
                    ball_features = tensor_data['ball_features']
                    player_bbox_features = tensor_data['player_bbox_features']
                    player_pose_features = tensor_data['player_pose_features']
                    court_features = tensor_data['court_features']
                    
                    # モデルの forward メソッドを呼び出し
                    logits = self.model(
                        ball_features=ball_features,
                        player_bbox_features=player_bbox_features,
                        player_pose_features=player_pose_features,
                        court_features=court_features
                    )
                
                if self.debug:
                    logger.debug(f"推論結果 logits shape: {logits.shape}")
                
                return logits
                
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise

    def postprocess(self, logits: torch.Tensor, meta_data: Any = None) -> Dict[str, Any]:
        """
        推論結果を後処理します。
        
        Args:
            logits: 推論結果 [B, T, 2]
            meta_data: メタデータ
            
        Returns:
            Dict[str, Any]: 後処理結果
        """
        try:
            # Sigmoid適用でprobabilityに変換
            probs = torch.sigmoid(logits)  # [B, T, 2]
            
            # 最新フレーム（時系列の最後）の結果を取得
            latest_probs = probs[0, -1, :].cpu().numpy()  # [2]
            hit_prob, bounce_prob = float(latest_probs[0]), float(latest_probs[1])
            
            # 信号履歴に追加
            self.signal_history.append((hit_prob, bounce_prob))
            if len(self.signal_history) > self.max_history_length:
                self.signal_history = self.signal_history[-self.max_history_length:]
            
            # 平滑化された信号を計算
            smoothed_signals = self._smooth_signals()
            
            # イベント検知
            hit_detected = hit_prob > self.confidence_threshold
            bounce_detected = bounce_prob > self.confidence_threshold
            
            result = {
                'hit_probability': hit_prob,
                'bounce_probability': bounce_prob,
                'hit_detected': hit_detected,
                'bounce_detected': bounce_detected,
                'smoothed_hit_signal': smoothed_signals[0],
                'smoothed_bounce_signal': smoothed_signals[1],
                'signal_history': self.signal_history[-min(30, len(self.signal_history)):],  # 直近30フレーム
                'timestamp': len(self.signal_history)
            }
            
            if self.debug:
                logger.debug(f"イベント検知結果: hit={hit_prob:.3f}, bounce={bounce_prob:.3f}")
                
            return result
            
        except Exception as e:
            logger.error(f"後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise

    def _smooth_signals(self) -> Tuple[float, float]:
        """
        移動平均による信号平滑化を行います。
        
        Returns:
            Tuple[float, float]: 平滑化された (hit_signal, bounce_signal)
        """
        if len(self.signal_history) == 0:
            return 0.0, 0.0
        
        # 直近のウィンドウサイズ分のサンプルを取得
        window_size = min(self.smoothing_window, len(self.signal_history))
        recent_signals = self.signal_history[-window_size:]
        
        # 移動平均を計算
        hit_signals = [signal[0] for signal in recent_signals]
        bounce_signals = [signal[1] for signal in recent_signals]
        
        smoothed_hit = np.mean(hit_signals)
        smoothed_bounce = np.mean(bounce_signals)
        
        return float(smoothed_hit), float(smoothed_bounce)

    def overlay(self, frame: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        """
        フレームにイベント検知結果をオーバーレイします。
        
        Args:
            frame: 入力フレーム [H, W, 3]
            prediction: 予測結果
            
        Returns:
            np.ndarray: オーバーレイされたフレーム
        """
        try:
            overlay_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # 信号波を描画する領域のサイズ
            signal_width = min(300, w // 3)
            signal_height = min(100, h // 6)
            signal_x = 20
            signal_y = 20
            
            # 背景の半透明矩形を描画
            overlay = overlay_frame.copy()
            cv2.rectangle(overlay, 
                         (signal_x - 10, signal_y - 10), 
                         (signal_x + signal_width + 10, signal_y + signal_height * 2 + 30), 
                         (0, 0, 0), -1)
            overlay_frame = cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0)
            
            # 信号履歴を描画
            signal_history = prediction.get('signal_history', [])
            if len(signal_history) > 1:
                self._draw_signal_wave(overlay_frame, signal_history, 
                                     signal_x, signal_y, signal_width, signal_height, 
                                     signal_type='hit', color=(0, 255, 255))  # Hit: Yellow
                
                self._draw_signal_wave(overlay_frame, signal_history, 
                                     signal_x, signal_y + signal_height + 20, signal_width, signal_height, 
                                     signal_type='bounce', color=(255, 0, 255))  # Bounce: Magenta
            
            # 現在の確率値をテキストで表示
            hit_prob = prediction.get('hit_probability', 0.0)
            bounce_prob = prediction.get('bounce_probability', 0.0)
            
            cv2.putText(overlay_frame, f"Hit: {hit_prob:.3f}", 
                       (signal_x, signal_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(overlay_frame, f"Bounce: {bounce_prob:.3f}", 
                       (signal_x, signal_y + signal_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # イベント検知を赤い点で強調
            if prediction.get('hit_detected', False):
                cv2.circle(overlay_frame, (signal_x + signal_width - 10, signal_y + signal_height // 2), 
                          8, (0, 0, 255), -1)
                
            if prediction.get('bounce_detected', False):
                cv2.circle(overlay_frame, (signal_x + signal_width - 10, signal_y + signal_height + 20 + signal_height // 2), 
                          8, (0, 0, 255), -1)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"オーバーレイエラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return frame

    def _draw_signal_wave(self, frame: np.ndarray, signal_history: List[Tuple[float, float]], 
                         x: int, y: int, width: int, height: int, 
                         signal_type: str, color: Tuple[int, int, int]):
        """
        信号波を描画します。
        
        Args:
            frame: 描画対象フレーム
            signal_history: 信号履歴
            x, y: 描画開始位置
            width, height: 描画領域サイズ
            signal_type: 'hit' or 'bounce'
            color: 描画色 (B, G, R)
        """
        if len(signal_history) < 2:
            return
            
        signal_idx = 0 if signal_type == 'hit' else 1
        
        # 信号値を正規化してピクセル座標に変換
        points = []
        for i, (hit_val, bounce_val) in enumerate(signal_history):
            signal_val = hit_val if signal_type == 'hit' else bounce_val
            
            # X座標: 時間軸
            px = x + int((i / (len(signal_history) - 1)) * width)
            
            # Y座標: 信号値 (0-1 を height に変換、上下反転)
            py = y + height - int(signal_val * height)
            
            points.append((px, py))
        
        # 連続した線で波形を描画
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        
        # 中央線（基準線）を描画
        cv2.line(frame, (x, y + height // 2), (x + width, y + height // 2), 
                 (128, 128, 128), 1)

    def reset_history(self):
        """信号履歴をリセットします。"""
        self.signal_history.clear()
        logger.info("イベント検知の信号履歴をリセットしました")


# ユーティリティ関数
def create_event_predictor(litmodule, **kwargs) -> EventPredictor:
    """
    EventPredictorのファクトリ関数。
    
    Args:
        litmodule: 初期化済みのモデルインスタンス（例: LitTransformerV2）
        **kwargs: その他のパラメータ（device, confidence_threshold, smoothing_window, debug, use_half など）
        
    Returns:
        EventPredictor: 初期化されたEventPredictor
    """
    return EventPredictor(litmodule=litmodule, **kwargs) 