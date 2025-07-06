#!/usr/bin/env python
"""
テニスイベント検出用 事前処理パイプライン

長時間動画からイベント候補を検出し、短い動画クリップと初期アノテーションJSONを自動生成します。
このスクリプトは既存のボール検出・イベント検出モデルを使用して、
アノテーション作業の効率化を図ります。
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import torch
import numpy as np
import subprocess
from dataclasses import dataclass
from collections import defaultdict

# 既存モデルのインポート
try:
    from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss
    from src.event.model.transformer_v2 import EventTransformerV2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(e)
    print(f"モジュールのインポートに失敗しました: {e}")


@dataclass
class ClipConfig:
    """クリップ生成の設定"""
    clip_duration_sec: float = 4.0  # クリップの長さ（秒）
    clip_overlap_sec: float = 0.5  # クリップ間のオーバーラップ（秒）
    img_width: int = 640
    img_height: int = 360
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelInferenceEngine:
    """
    モデル推論エンジン
    ボール検出とイベント検出の推論を統合的に行います
    """
    
    def __init__(self, config: ClipConfig, ball_ckpt: str, event_ckpt: str):
        """
        Args:
            config: クリップ生成設定
            ball_ckpt: ボール検出モデルのチェックポイントパス
            event_ckpt: イベント検出モデルのチェックポイントパス
        """
        self.config = config
        self.ball_model = None
        self.event_model = None
        self.ball_transform = None
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        if IMPORT_SUCCESS:
            self._load_models(ball_ckpt, event_ckpt)
        else:
            raise ImportError(f"必要なモジュールがインポートできません: {IMPORT_ERROR_MESSAGE}")
    
    def _load_models(self, ball_ckpt: str, event_ckpt: str) -> None:
        """モデルのロード"""
        try:
            # ボール検出モデル
            self.ball_model = LitLiteTracknetFocalLoss.load_from_checkpoint(
                ball_ckpt, map_location=self.config.device
            ).model.to(self.config.device).eval()
            
            # ボール検出用の変換
            self.ball_transform = A.ReplayCompose([
                A.Resize(height=self.config.img_height, width=self.config.img_width),
                A.Normalize(),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format="xy"))
            
            # イベント検出モデル（簡略化）
            # 実際の実装では、EventTransformerV2を適切にロードする必要があります
            self.event_model = None  # TODO: 適切なロード処理を実装
            
            self.logger.info("モデルのロードが完了しました")
        except Exception as e:
            self.logger.error(f"モデルのロードに失敗しました: {e}")
            raise e
    
    def infer_ball_sequence(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        フレーム系列からボール検出を実行
        
        Args:
            frames: RGB画像フレームのリスト
            
        Returns:
            ボール検出結果のリスト
        """
        if not self.ball_model:
            return []
        
        ball_results = []
        frame_buffer = []
        
        with torch.no_grad():
            for frame_idx, frame in enumerate(frames):
                try:
                    # フレームの前処理
                    transformed = self.ball_transform(image=frame)
                    frame_tensor = transformed['image'].unsqueeze(0).to(self.config.device)
                    
                    # 3フレームバッファの管理
                    frame_buffer.append(frame_tensor)
                    if len(frame_buffer) > 3:
                        frame_buffer.pop(0)
                    
                    # 3フレーム以上の場合のみ推論実行
                    if len(frame_buffer) == 3:
                        input_tensor = torch.cat(frame_buffer, dim=1)
                        heatmap = self.ball_model(input_tensor)
                        
                        # ヒートマップから最大値の位置を取得
                        max_val = torch.max(heatmap).item()
                        max_pos = torch.unravel_index(torch.argmax(heatmap), heatmap.shape)
                        
                        # 元の画像サイズに変換
                        y_norm = max_pos[-2].item() / heatmap.shape[-2]
                        x_norm = max_pos[-1].item() / heatmap.shape[-1]
                        
                        ball_result = {
                            "keypoint": [x_norm, y_norm] if max_val > 0.5 else None,
                            "visibility": 2 if max_val > 0.5 else 0,
                            "probability": max_val,
                            "is_interpolated": False
                        }
                    else:
                        # バッファが満たない間は初期値
                        ball_result = {
                            "keypoint": None,
                            "visibility": 0,
                            "probability": 0.0,
                            "is_interpolated": False
                        }
                    
                    ball_results.append(ball_result)
                    
                except Exception as e:
                    self.logger.warning(f"フレーム {frame_idx} のボール検出でエラー: {e}")
                    ball_results.append({
                        "keypoint": None,
                        "visibility": 0,
                        "probability": 0.0,
                        "is_interpolated": False
                    })
        
        return ball_results
    
    def infer_event_sequence(self, ball_results: List[Dict], frames: List[np.ndarray]) -> List[int]:
        """
        イベント検出の簡易実装
        実際の実装では、EventTransformerV2を使用してより精密な検出を行います
        
        Args:
            ball_results: ボール検出結果
            frames: フレーム系列
            
        Returns:
            各フレームのイベントステータス (0: なし, 1: ヒット, 2: バウンド)
        """
        event_results = []
        
        # 簡易的なイベント検出ロジック
        prev_visibility = 0
        for i, ball_result in enumerate(ball_results):
            current_visibility = ball_result.get('visibility', 0)
            probability = ball_result.get('probability', 0.0)
            
            # 簡単なヒューリスティック
            if probability > 0.8 and prev_visibility == 0 and current_visibility > 0:
                # ボールが突然現れた場合 -> ヒット候補
                event_results.append(1)
            elif probability > 0.6 and i > 0 and ball_results[i-1].get('probability', 0) < 0.3:
                # 確率が急上昇した場合 -> バウンド候補
                event_results.append(2)
            else:
                event_results.append(0)
            
            prev_visibility = current_visibility
        
        return event_results


class EventCandidateDetector:
    """
    イベント候補の特定とクラスタリング
    """
    
    def __init__(self, event_threshold: float = 0.3, min_gap_frames: int = 30):
        """
        Args:
            event_threshold: イベント検出の閾値
            min_gap_frames: イベント間の最小間隔（フレーム数）
        """
        self.event_threshold = event_threshold
        self.min_gap_frames = min_gap_frames
        self.logger = logging.getLogger(__name__)
    
    def find_event_candidates(self, event_probs: List[float], ball_results: List[Dict]) -> List[int]:
        """
        イベント候補フレームを特定
        
        Args:
            event_probs: 各フレームのイベント確率
            ball_results: ボール検出結果
            
        Returns:
            イベント候補フレームのインデックスリスト
        """
        candidates = []
        
        for i, prob in enumerate(event_probs):
            if prob > self.event_threshold:
                # ボールの可視性も考慮
                ball_vis = ball_results[i].get('visibility', 0) if i < len(ball_results) else 0
                if ball_vis > 0:
                    candidates.append(i)
        
        # 近接するフレームをクラスタリング
        clustered_candidates = self._cluster_candidates(candidates)
        
        self.logger.info(f"検出されたイベント候補: {len(clustered_candidates)} 個")
        return clustered_candidates
    
    def _cluster_candidates(self, candidates: List[int]) -> List[int]:
        """
        近接するイベント候補をクラスタリングし、代表フレームを選択
        """
        if not candidates:
            return []
        
        clustered = []
        current_cluster = [candidates[0]]
        
        for candidate in candidates[1:]:
            if candidate - current_cluster[-1] <= self.min_gap_frames:
                current_cluster.append(candidate)
            else:
                # クラスタの中央値を代表フレームとして選択
                representative = current_cluster[len(current_cluster) // 2]
                clustered.append(representative)
                current_cluster = [candidate]
        
        # 最後のクラスタを処理
        if current_cluster:
            representative = current_cluster[len(current_cluster) // 2]
            clustered.append(representative)
        
        return clustered


class ClipGenerator:
    """
    動画クリップの生成
    """
    
    def __init__(self, config: ClipConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_clips(
        self, 
        video_path: str, 
        event_frames: List[int], 
        fps: float, 
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        イベント候補フレーム周辺の動画クリップを生成
        
        Args:
            video_path: 入力動画のパス
            event_frames: イベント候補フレームのリスト
            fps: 動画のFPS
            output_dir: 出力ディレクトリ
            
        Returns:
            生成されたクリップの情報リスト
        """
        clips_info = []
        
        for i, center_frame in enumerate(event_frames):
            try:
                clip_info = self._generate_single_clip(
                    video_path, center_frame, fps, output_dir, i
                )
                if clip_info:
                    clips_info.append(clip_info)
            except Exception as e:
                self.logger.error(f"クリップ {i} の生成に失敗しました: {e}")
        
        return clips_info
    
    def _generate_single_clip(
        self, 
        video_path: str, 
        center_frame: int, 
        fps: float, 
        output_dir: Path, 
        clip_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        単一のクリップを生成
        """
        # クリップの開始・終了フレームを計算
        half_duration_frames = int((self.config.clip_duration_sec / 2) * fps)
        start_frame = max(0, center_frame - half_duration_frames)
        end_frame = center_frame + half_duration_frames
        
        # 開始・終了時間を計算
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # クリップファイル名
        clip_name = f"clip_{clip_index:03d}"
        clip_path = output_dir / f"{clip_name}.mp4"
        
        # FFmpegでクリップを生成
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(self.config.clip_duration_sec),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            str(clip_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"クリップ生成完了: {clip_path}")
            
            # クリップ情報を返す
            return {
                "clip_name": clip_name,
                "clip_path": str(clip_path),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "center_frame": center_frame,
                "start_time": start_time,
                "end_time": end_time
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpegエラー: {e.stderr}")
            return None


class AnnotationGenerator:
    """
    初期アノテーションJSONの生成
    """
    
    def __init__(self, config: ClipConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_annotation_json(
        self,
        clip_info: Dict[str, Any],
        ball_results: List[Dict],
        event_results: List[int],
        video_path: str,
        output_dir: Path
    ) -> str:
        """
        クリップ用の初期アノテーションJSONを生成
        
        Args:
            clip_info: クリップ情報
            ball_results: ボール検出結果
            event_results: イベント検出結果
            video_path: 元動画のパス
            output_dir: 出力ディレクトリ
            
        Returns:
            生成されたJSONファイルのパス
        """
        # 動画情報を取得
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # フレーム範囲の計算
        start_frame = clip_info["start_frame"]
        end_frame = clip_info["end_frame"]
        
        # アノテーションデータの構築
        annotation_data = {
            "clip_info": {
                "source_video": str(video_path),
                "clip_name": clip_info["clip_name"],
                "clip_path": clip_info["clip_path"],
                "fps": fps,
                "width": width,
                "height": height
            },
            "frames": []
        }
        
        # 各フレームのアノテーションを生成
        for local_frame_idx in range(end_frame - start_frame):
            global_frame_idx = start_frame + local_frame_idx
            
            # ボール情報を取得
            ball_info = ball_results[global_frame_idx] if global_frame_idx < len(ball_results) else {
                "keypoint": None,
                "visibility": 0,
                "is_interpolated": False
            }
            
            # イベント情報を取得
            event_status = event_results[global_frame_idx] if global_frame_idx < len(event_results) else 0
            
            # 正規化座標を絶対座標に変換
            keypoint = ball_info.get("keypoint")
            if keypoint:
                keypoint = [keypoint[0] * width, keypoint[1] * height]
            
            frame_annotation = {
                "frame_number": local_frame_idx,
                "ball": {
                    "keypoint": keypoint,
                    "visibility": ball_info.get("visibility", 0),
                    "is_interpolated": ball_info.get("is_interpolated", False)
                },
                "event_status": event_status
            }
            
            annotation_data["frames"].append(frame_annotation)
        
        # JSONファイルに保存
        json_path = output_dir / f"{clip_info['clip_name']}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"アノテーションJSON生成完了: {json_path}")
        return str(json_path)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="テニス動画からイベント候補クリップを自動生成")
    parser.add_argument("--input_video", "-i", required=True, help="入力動画ファイル")
    parser.add_argument("--output_dir", "-o", required=True, help="出力ディレクトリ")
    parser.add_argument("--event_threshold", "-t", type=float, default=0.3, help="イベント検出閾値")
    parser.add_argument("--clip_duration", "-d", type=float, default=4.0, help="クリップの長さ（秒）")
    parser.add_argument("--ball_ckpt", default="checkpoints/ball/lit_lite_tracknet/best_model.ckpt", help="ボール検出モデル")
    parser.add_argument("--event_ckpt", default="checkpoints/event/transformer_v2/epoch=18-step=532.ckpt", help="イベント検出モデル")
    
    args = parser.parse_args()
    
    # 設定
    config = ClipConfig(
        clip_duration_sec=args.clip_duration,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"事前処理パイプライン開始: {args.input_video}")
    
    try:
        # 1. モデルの初期化
        logger.info("モデルをロード中...")
        inference_engine = ModelInferenceEngine(config, args.ball_ckpt, args.event_ckpt)
        
        # 2. 動画の読み込みと全フレーム推論
        logger.info("動画を読み込み、推論を実行中...")
        cap = cv2.VideoCapture(args.input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        logger.info(f"総フレーム数: {len(frames)}")
        
        # 3. ボール検出
        ball_results = inference_engine.infer_ball_sequence(frames)
        
        # 4. イベント検出
        event_results = inference_engine.infer_event_sequence(ball_results, frames)
        
        # 5. イベント候補の特定
        detector = EventCandidateDetector(event_threshold=args.event_threshold)
        event_probs = [1.0 if e > 0 else 0.0 for e in event_results]  # 簡易的な確率変換
        event_candidates = detector.find_event_candidates(event_probs, ball_results)
        
        # 6. クリップ生成
        logger.info("クリップを生成中...")
        clip_generator = ClipGenerator(config)
        clips_info = clip_generator.generate_clips(args.input_video, event_candidates, fps, output_dir)
        
        # 7. アノテーションJSON生成
        logger.info("アノテーションJSONを生成中...")
        annotation_generator = AnnotationGenerator(config)
        
        for clip_info in clips_info:
            annotation_generator.generate_annotation_json(
                clip_info, ball_results, event_results, args.input_video, output_dir
            )
        
        logger.info(f"処理完了: {len(clips_info)} 個のクリップを生成しました")
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        raise e


if __name__ == "__main__":
    main()