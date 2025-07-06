#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎾 テニスボール検出・追跡・可視化システム (バッチ処理対応版)

このスクリプトは、動画ファイルからテニスボールを検出・追跡し、
結果を可視化した新しい動画ファイルを生成します。
バッチ処理により効率的な推論を実現します。

使用方法:
    python batch_ball_tracker.py --input_video path/to/input.mp4 [--output_video path/to/output.mp4] [--batch_size 8]

依存関係:
    - opencv-python
    - torch
    - torchvision
    - omegaconf
    - numpy
    - PIL (Pillow)
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from collections import deque
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import DictConfig

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# WASB-SBDTライブラリのインポート
try:
    from third_party.WASB_SBDT import create_model_package, load_default_config
    from third_party.WASB_SBDT.src.utils.utils import read_image
    from third_party.WASB_SBDT.src.dataloaders.dataset_loader import get_transform
    from third_party.WASB_SBDT.src.utils.image import get_affine_transform, affine_transform
    logger_import = logging.getLogger(__name__)
    logger_import.info("WASB-SBDTライブラリのインポート成功")
except ImportError as e:
    logger_import = logging.getLogger(__name__)
    logger_import.error(f"WASB-SBDTライブラリのインポートに失敗: {e}")
    sys.exit(1)

# ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchTennisBallTracker:
    """バッチ処理対応テニスボール検出・追跡システムのメインクラス"""
    
    def __init__(self, model_path: str = None, batch_size: int = 8):
        """
        初期化
        
        Args:
            model_path: 学習済みモデルのパス（Noneの場合はデフォルトモデルを使用）
            batch_size: バッチサイズ
        """
        logger.info(f"BatchTennisBallTrackerの初期化を開始 - バッチサイズ: {batch_size}")
        
        self.batch_size = batch_size
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用デバイス: {self.device}")
            
            # GPU使用時はメモリ情報を表示
            if self.device.type == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU メモリ: {gpu_memory:.1f} GB")
                
        except Exception as e:
            logger.error(f"デバイス設定に失敗: {e}")
            raise
        
        # デフォルト設定の読み込み
        try:
            self.config = load_default_config()
            logger.debug("デフォルト設定の読み込み完了")
        except Exception as e:
            logger.error(f"デフォルト設定の読み込みに失敗: {e}")
            raise
        
        # モデルパスの設定
        if model_path is None:
            model_path = str(project_root / "third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar")
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            logger.info(f"モデルファイル: {model_path}")
        except Exception as e:
            logger.error(f"モデルファイルの確認に失敗: {e}")
            raise
        
        # モデル、ポストプロセッサ、トラッカーの初期化
        try:
            logger.debug("モデル・ポストプロセッサ・トラッカーの初期化を開始")
            self.model, self.postprocessor, self.tracker = create_model_package(
                self.config, model_path
            )
            logger.info("モデル・ポストプロセッサ・トラッカーの初期化完了")
        except Exception as e:
            logger.error(f"モデルの初期化に失敗しました: {e}")
            raise
        
        # モデル設定の取得
        try:
            self.frames_in = self.config.model.frames_in  # 通常は3
            self.input_size = (self.config.model.inp_width, self.config.model.inp_height)  # (512, 288)
            logger.debug(f"モデル設定 - frames_in: {self.frames_in}, input_size: {self.input_size}")
        except Exception as e:
            logger.error(f"モデル設定の取得に失敗: {e}")
            raise
        
        # フレームバッファとバッチ処理用の変数
        self.frame_buffer = deque(maxlen=self.frames_in)
        self.batch_buffer = []
        self.batch_frame_info = []  # バッチ内の各フレームの情報を保存
        
        logger.info("BatchTennisBallTrackerの初期化完了")
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        フレームの前処理
        
        Args:
            frame: OpenCVで読み込んだフレーム (BGR)
            
        Returns:
            前処理済みのテンソルとアフィン変換行列
        """
        logger.debug("フレーム前処理を開始")
        
        try:
            # BGR -> RGB変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"色空間変換に失敗: {e}")
            raise
        
        try:
            # アフィン変換の準備
            h, w = frame_rgb.shape[:2]
            c = np.array([w / 2., h / 2.], dtype=np.float32)
            s = max(h, w) * 1.0
            trans = get_affine_transform(c, s, 0, self.input_size, inv=0)
        except Exception as e:
            logger.error(f"アフィン変換パラメータの計算に失敗: {e}")
            raise
      
        try:
            # アフィン変換の適用
            warped = cv2.warpAffine(frame_rgb, trans, self.input_size, flags=cv2.INTER_LINEAR)
        except Exception as e:
            logger.error(f"アフィン変換の適用に失敗: {e}")
            raise
        
        try:
            # 正規化とテンソル化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            tensor = torch.from_numpy(warped).permute(2, 0, 1).float() / 255.0
            for i, (m, s) in enumerate(zip(mean, std)):
                tensor[i] = (tensor[i] - m) / s
            
            return tensor, trans
        except Exception as e:
            logger.error(f"テンソル変換・正規化に失敗: {e}")
            raise
    
    def prepare_batch_input(self, frame_sequences: List[List[np.ndarray]]) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        バッチ入力の準備
        
        Args:
            frame_sequences: フレームシーケンスのリスト
            
        Returns:
            バッチテンソルとアフィン変換行列の辞書
        """
        batch_tensors = []
        affine_mats = {}
        
        for batch_idx, frames in enumerate(frame_sequences):
            try:
                # フレームの前処理
                processed_frames = []
                affine_matrices = []
                
                for frame in frames:
                    tensor, trans = self.preprocess_frame(frame)
                    processed_frames.append(tensor)
                    affine_matrices.append(trans)
                
                # シーケンスのテンソル化
                sequence_tensor = torch.cat(processed_frames, dim=0)
                batch_tensors.append(sequence_tensor)
                
                # アフィン変換行列の保存
                affine_tensor = torch.from_numpy(np.stack(affine_matrices)).float().to(self.device)
                affine_mats[batch_idx] = affine_tensor
                
            except Exception as e:
                logger.error(f"バッチ{batch_idx}の前処理に失敗: {e}")
                raise
        
        try:
            # バッチテンソルの作成
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            logger.debug(f"バッチテンソル作成完了 - shape: {batch_tensor.shape}")
            return batch_tensor, affine_mats
        except Exception as e:
            logger.error(f"バッチテンソル作成に失敗: {e}")
            raise
    
    def process_batch(self, frame_sequences: List[List[np.ndarray]]) -> List[Dict[str, Any]]:
        """
        バッチ処理でボール検出を実行
        
        Args:
            frame_sequences: フレームシーケンスのリスト
            
        Returns:
            各シーケンスのボール検出結果
        """
        logger.debug(f"バッチ処理開始 - バッチサイズ: {len(frame_sequences)}")
        
        try:
            # バッチ入力の準備
            batch_tensor, affine_mats = self.prepare_batch_input(frame_sequences)
            
            # モデルでの推論
            with torch.no_grad():
                start_time = time.time()
                predictions = self.model(batch_tensor)
                inference_time = time.time() - start_time
                logger.debug(f"バッチ推論完了 - 時間: {inference_time:.3f}秒")
            
            # ポストプロセッサでの処理
            postprocess_results = self.postprocessor.run(predictions, affine_mats)
            print(postprocess_results)
            
            # 各シーケンスの結果を抽出
            results = []
            for batch_idx in range(len(frame_sequences)):
                try:
                    # 各シーケンスの最後のフレームの結果を取得
                    frame_detections = []
                    if batch_idx in postprocess_results and (self.frames_in - 1) in postprocess_results[batch_idx]:
                        last_frame_result = postprocess_results[batch_idx][self.frames_in - 1]
                        if 0 in last_frame_result:
                            xys = last_frame_result[0]['xys']
                            scores = last_frame_result[0]['scores']
                            for xy, score in zip(xys, scores):
                                if not isinstance(xy, np.ndarray):
                                    xy = np.array(xy, dtype=np.float32)
                                frame_detections.append({'xy': xy, 'score': score})
                    
                    # トラッカーでの追跡
                    track_result = self.tracker.update(frame_detections)
                    results.append(track_result)
                    
                except Exception as e:
                    logger.warning(f"バッチ{batch_idx}の結果処理に失敗: {e}")
                    results.append({'x': -1, 'y': -1, 'visi': False, 'score': 0})
            
            logger.debug(f"バッチ処理完了 - {len(results)}件の結果")
            return results
            
        except Exception as e:
            logger.error(f"バッチ処理に失敗: {e}")
            # エラー時は空の結果を返す
            return [{'x': -1, 'y': -1, 'visi': False, 'score': 0} for _ in range(len(frame_sequences))]
    
    def process_video(self, input_path: str, output_path: str) -> None:
        """
        動画ファイルを処理してボール追跡結果を可視化（バッチ処理版）
        
        Args:
            input_path: 入力動画のパス
            output_path: 出力動画のパス
        """
        logger.info(f"バッチ動画処理開始: {input_path}")
        
        # 動画の読み込み
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"動画ファイルを開けません: {input_path}")
        except Exception as e:
            logger.error(f"動画ファイルの読み込みに失敗: {e}")
            raise
        
        # 動画プロパティの取得
        try:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"動画情報: {width}x{height}, {fps}FPS, {total_frames}フレーム")
        except Exception as e:
            logger.error(f"動画プロパティの取得に失敗: {e}")
            cap.release()
            raise
        
        # 動画ライターの初期化
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise ValueError(f"出力動画ファイルを作成できません: {output_path}")
        except Exception as e:
            logger.error(f"動画ライターの初期化に失敗: {e}")
            cap.release()
            raise
        
        # トラッカーのリセット
        try:
            self.tracker.refresh()
        except Exception as e:
            logger.warning(f"トラッカーのリセットに失敗: {e}")
        
        # 全フレームを読み込み
        logger.info("全フレームを読み込み中...")
        all_frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            logger.info(f"フレーム読み込み完了: {len(all_frames)}フレーム")
        except Exception as e:
            logger.error(f"フレーム読み込み中にエラー: {e}")
            cap.release()
            raise
        
        # バッチ処理の実行
        logger.info("バッチ処理を開始...")
        processed_results = []
        
        try:
            # 初期フレームの処理（frames_in-1フレームは推論なし）
            for i in range(min(self.frames_in - 1, len(all_frames))):
                processed_results.append(None)
                self.frame_buffer.append(all_frames[i])
            
            # バッチ処理用のシーケンス作成
            batch_sequences = []
            batch_frame_indices = []
            
            for i in range(self.frames_in - 1, len(all_frames)):
                # フレームバッファを更新
                self.frame_buffer.append(all_frames[i])
                
                # 現在のシーケンスをバッチに追加
                current_sequence = list(self.frame_buffer)
                batch_sequences.append(current_sequence)
                batch_frame_indices.append(i)
                
                # バッチサイズに達したら処理実行
                if len(batch_sequences) == self.batch_size:
                    batch_results = self.process_batch(batch_sequences)
                    processed_results.extend(batch_results)
                    
                    # バッチをクリア
                    batch_sequences = []
                    batch_frame_indices = []
                    
                    # 進捗表示
                    progress = len(processed_results) / len(all_frames) * 100
                    logger.info(f"処理進捗: {len(processed_results)}/{len(all_frames)} ({progress:.1f}%)")
            
            # 残りのシーケンスを処理
            if batch_sequences:
                batch_results = self.process_batch(batch_sequences)
                processed_results.extend(batch_results)
            
            logger.info("バッチ処理完了")
            
        except Exception as e:
            logger.error(f"バッチ処理中にエラー: {e}")
            # エラー時は残りのフレームに空の結果を追加
            while len(processed_results) < len(all_frames):
                processed_results.append({'x': -1, 'y': -1, 'visi': False, 'score': 0})
        
        # 結果の描画と動画出力
        logger.info("結果の描画と動画出力を開始...")
        try:
            for i, (frame, result) in enumerate(zip(all_frames, processed_results)):
                result_frame = self.draw_results(frame, result, i)
                out.write(result_frame)
                
                if i % 100 == 0:
                    logger.info(f"出力進捗: {i}/{len(all_frames)} ({i/len(all_frames)*100:.1f}%)")
            
            out.release()
            logger.info(f"動画処理完了: {output_path}")
            
        except Exception as e:
            logger.error(f"動画出力中にエラー: {e}")
            out.release()
            raise
        
        # 処理統計の表示
        total_detection_frames = len(all_frames) - (self.frames_in - 1)
        detected_frames = sum(1 for r in processed_results[self.frames_in-1:] if r and r['visi'])
        detection_rate = detected_frames / total_detection_frames * 100 if total_detection_frames > 0 else 0
        
        logger.info(f"📊 処理統計:")
        logger.info(f"  - 総フレーム数: {len(all_frames)}")
        logger.info(f"  - 検出対象フレーム数: {total_detection_frames}")
        logger.info(f"  - ボール検出フレーム数: {detected_frames}")
        logger.info(f"  - 検出率: {detection_rate:.1f}%")
    
    def draw_results(self, frame: np.ndarray, ball_position: Optional[Dict[str, Any]], frame_idx: int) -> np.ndarray:
        """
        フレームに検出結果を描画
        
        Args:
            frame: 元フレーム
            ball_position: ボール位置情報
            frame_idx: フレーム番号
            
        Returns:
            描画済みフレーム
        """
        try:
            result_frame = frame.copy()
        except Exception as e:
            logger.error(f"フレームのコピーに失敗: {e}")
            return frame
        
        # フレーム番号の描画
        try:
            cv2.putText(result_frame, f"Frame: {frame_idx}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except Exception as e:
            logger.warning(f"フレーム番号の描画に失敗: {e}")
        
        # バッチサイズの表示
        try:
            cv2.putText(result_frame, f"Batch Size: {self.batch_size}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            logger.warning(f"バッチサイズの描画に失敗: {e}")
        
        # ボール位置の描画
        try:
            if ball_position and ball_position['visi']:
                x, y = int(ball_position['x']), int(ball_position['y'])
                score = ball_position['score']
                
                # ボール位置に円を描画
                cv2.circle(result_frame, (x, y), 8, (0, 255, 0), -1)  # 緑色の塗りつぶし円
                cv2.circle(result_frame, (x, y), 12, (0, 255, 0), 2)  # 緑色の枠線
                
                # スコアの表示
                cv2.putText(result_frame, f"Score: {score:.2f}", 
                           (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 座標の表示
                cv2.putText(result_frame, f"({x}, {y})", 
                           (x + 15, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            else:
                # ボールが検出されない場合
                cv2.putText(result_frame, "Ball: Not Detected", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            logger.warning(f"ボール位置の描画に失敗: {e}")
            try:
                cv2.putText(result_frame, "Ball: Draw Error", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except:
                pass
            
        return result_frame


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="🎾 テニスボール検出・追跡・可視化システム (バッチ処理対応版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python batch_ball_tracker.py --input_video samples/tennis_match.mp4
    python batch_ball_tracker.py --input_video input.mp4 --output_video result.mp4 --batch_size 16
    
バッチサイズの推奨値:
    - CPU使用時: 1-4
    - GPU使用時: 8-32 (GPUメモリに応じて調整)
        """
    )
    
    parser.add_argument(
        "--input_video", 
        required=True,
        help="入力動画ファイルのパス"
    )
    
    parser.add_argument(
        "--output_video", 
        help="出力動画ファイルのパス（省略時は入力ファイル名_batch_outputが使用されます）"
    )
    
    parser.add_argument(
        "--model_path",
        help="学習済みモデルのパス（省略時はデフォルトモデルを使用）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="バッチサイズ（デフォルト: 8）"
    )
    
    try:
        args = parser.parse_args()
        logger.debug(f"コマンドライン引数の解析完了: {args}")
    except Exception as e:
        logger.error(f"コマンドライン引数の解析に失敗: {e}")
        sys.exit(1)
    
    # バッチサイズの検証
    if args.batch_size < 1:
        logger.error("バッチサイズは1以上である必要があります")
        sys.exit(1)
    
    # 入力ファイルの存在確認
    try:
        if not os.path.exists(args.input_video):
            logger.error(f"入力動画ファイルが見つかりません: {args.input_video}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"入力ファイルの確認中にエラー: {e}")
        sys.exit(1)
    
    # 出力ファイル名の生成
    if args.output_video is None:
        try:
            input_path = Path(args.input_video)
            output_path = input_path.parent / f"{input_path.stem}_batch_output{input_path.suffix}"
            args.output_video = str(output_path)
        except Exception as e:
            logger.error(f"出力ファイル名の生成に失敗: {e}")
            sys.exit(1)
    
    logger.info(f"入力動画: {args.input_video}")
    logger.info(f"出力動画: {args.output_video}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    
    try:
        # トラッカーの初期化
        start_time = time.time()
        tracker = BatchTennisBallTracker(model_path=args.model_path, batch_size=args.batch_size)
        init_time = time.time() - start_time
        logger.info(f"初期化時間: {init_time:.2f}秒")
        
        # 動画処理の実行
        process_start_time = time.time()
        tracker.process_video(args.input_video, args.output_video)
        process_time = time.time() - process_start_time
        
        logger.info(f"🎾 バッチボール追跡処理が正常に完了しました！")
        logger.info(f"⏱️  処理時間: {process_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()