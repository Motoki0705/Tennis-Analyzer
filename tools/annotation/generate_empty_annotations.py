#!/usr/bin/env python
"""
空のアノテーションJSON自動生成スクリプト

手動で配置された動画クリップファイルから、空のアノテーションJSONファイルを自動生成します。
モデルは一切使用せず、動画のメタデータのみを使用して初期アノテーション構造を作成します。
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any
import cv2

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmptyAnnotationGenerator:
    """
    手動配置されたクリップから空のアノテーションJSONを生成するクラス
    """
    
    def __init__(self, clips_dir: Path, annotations_dir: Path):
        """
        Args:
            clips_dir: クリップファイルが配置されたディレクトリ
            annotations_dir: アノテーションJSONファイルの出力ディレクトリ
        """
        self.clips_dir = clips_dir
        self.annotations_dir = annotations_dir
        self.logger = logging.getLogger(__name__)
        
        # 出力ディレクトリの作成
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """
        動画ファイルからメタデータを取得
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            メタデータ辞書
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"動画ファイルを開けません: {video_path}")
            
            # メタデータの取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            metadata = {
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "duration": duration
            }
            
            self.logger.info(f"メタデータ取得完了 {video_path.name}: {frame_count}フレーム, {fps:.2f}fps, {width}x{height}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"メタデータ取得エラー {video_path}: {e}")
            raise e
    
    def generate_empty_frames_data(self, frame_count: int) -> List[Dict[str, Any]]:
        """
        指定されたフレーム数分の空のフレームアノテーションデータを生成
        
        Args:
            frame_count: フレーム数
            
        Returns:
            空のフレームアノテーションデータのリスト
        """
        frames_data = []
        
        for frame_number in range(frame_count):
            frame_annotation = {
                "frame_number": frame_number,
                "ball": {
                    "keypoint": None,
                    "visibility": 0,
                    "is_interpolated": False
                },
                "event_status": 0
            }
            frames_data.append(frame_annotation)
        
        return frames_data
    
    def generate_annotation_json(self, clip_path: Path, source_video: str = "") -> bool:
        """
        単一のクリップファイルから空のアノテーションJSONを生成
        
        Args:
            clip_path: クリップファイルのパス
            source_video: 元動画のパス（オプション）
            
        Returns:
            生成成功フラグ
        """
        try:
            clip_name = clip_path.stem
            self.logger.info(f"アノテーションJSON生成中: {clip_name}")
            
            # 動画メタデータの取得
            metadata = self.get_video_metadata(clip_path)
            
            # 空のフレームデータ生成
            frames_data = self.generate_empty_frames_data(metadata["frame_count"])
            
            # アノテーションデータの構築
            annotation_data = {
                "clip_info": {
                    "source_video": source_video,
                    "clip_name": clip_name,
                    "clip_path": str(clip_path),
                    "fps": metadata["fps"],
                    "width": metadata["width"],
                    "height": metadata["height"]
                },
                "frames": frames_data
            }
            
            # JSONファイルの保存
            json_path = self.annotations_dir / f"{clip_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"アノテーションJSON生成完了: {json_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"アノテーションJSON生成エラー {clip_path}: {e}")
            return False
    
    def find_clip_files(self) -> List[Path]:
        """
        クリップディレクトリ内の動画ファイルを検索
        
        Returns:
            見つかった動画ファイルのパスリスト
        """
        if not self.clips_dir.exists():
            self.logger.error(f"クリップディレクトリが存在しません: {self.clips_dir}")
            return []
        
        # サポートする動画形式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        clip_files = []
        
        for ext in video_extensions:
            clip_files.extend(self.clips_dir.glob(f"*{ext}"))
            clip_files.extend(self.clips_dir.glob(f"*{ext.upper()}"))
        
        # ファイル名でソート
        clip_files.sort(key=lambda x: x.name)
        
        self.logger.info(f"発見されたクリップファイル数: {len(clip_files)}")
        return clip_files
    
    def process_all_clips(self, source_video: str = "", overwrite: bool = False) -> Dict[str, Any]:
        """
        すべてのクリップファイルを処理してアノテーションJSONを生成
        
        Args:
            source_video: 元動画のパス
            overwrite: 既存のアノテーションファイルを上書きするか
            
        Returns:
            処理結果の統計情報
        """
        clip_files = self.find_clip_files()
        
        if not clip_files:
            self.logger.warning("処理対象のクリップファイルが見つかりませんでした")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for clip_path in clip_files:
            clip_name = clip_path.stem
            json_path = self.annotations_dir / f"{clip_name}.json"
            
            # 既存ファイルのチェック
            if json_path.exists() and not overwrite:
                self.logger.info(f"スキップ（既存）: {clip_name}")
                skipped_count += 1
                continue
            
            # アノテーションJSON生成
            success = self.generate_annotation_json(clip_path, source_video)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        
        # 結果の報告
        results = {
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "total_clips": len(clip_files)
        }
        
        self.logger.info(f"処理完了: 成功 {processed_count}, スキップ {skipped_count}, 失敗 {failed_count}")
        return results
    
    def validate_generated_annotations(self) -> bool:
        """
        生成されたアノテーションファイルの妥当性をチェック
        
        Returns:
            妥当性チェック結果
        """
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        if not annotation_files:
            self.logger.warning("検証対象のアノテーションファイルが見つかりません")
            return False
        
        valid_count = 0
        invalid_count = 0
        
        for json_path in annotation_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 必要なフィールドの存在チェック
                required_fields = ["clip_info", "frames"]
                if all(field in data for field in required_fields):
                    clip_info = data["clip_info"]
                    required_clip_fields = ["clip_name", "clip_path", "fps", "width", "height"]
                    if all(field in clip_info for field in required_clip_fields):
                        valid_count += 1
                    else:
                        self.logger.warning(f"clip_info フィールド不足: {json_path.name}")
                        invalid_count += 1
                else:
                    self.logger.warning(f"必要なフィールド不足: {json_path.name}")
                    invalid_count += 1
                    
            except Exception as e:
                self.logger.error(f"ファイル検証エラー {json_path}: {e}")
                invalid_count += 1
        
        self.logger.info(f"検証結果: 有効 {valid_count}, 無効 {invalid_count}")
        return invalid_count == 0


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="手動配置されたクリップから空のアノテーションJSONを自動生成"
    )
    parser.add_argument(
        "--clips_dir", "-c",
        default="./datasets/annotation_workspace/clips",
        help="クリップファイルが配置されたディレクトリ"
    )
    parser.add_argument(
        "--annotations_dir", "-a",
        default="./datasets/annotation_workspace/annotations",
        help="アノテーションJSONファイルの出力ディレクトリ"
    )
    parser.add_argument(
        "--source_video", "-s",
        default="",
        help="元動画のパス（オプション、記録用）"
    )
    parser.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="既存のアノテーションファイルを上書き"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="生成後に妥当性チェックを実行"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログの出力"
    )
    
    args = parser.parse_args()
    
    # ログレベルの設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    clips_dir = Path(args.clips_dir)
    annotations_dir = Path(args.annotations_dir)
    
    logger.info(f"空のアノテーションJSON生成開始:")
    logger.info(f"  クリップディレクトリ: {clips_dir}")
    logger.info(f"  アノテーション出力ディレクトリ: {annotations_dir}")
    logger.info(f"  元動画: {args.source_video or '未指定'}")
    logger.info(f"  上書きモード: {args.overwrite}")
    
    try:
        # ジェネレーターの初期化
        generator = EmptyAnnotationGenerator(clips_dir, annotations_dir)
        
        # すべてのクリップを処理
        results = generator.process_all_clips(
            source_video=args.source_video,
            overwrite=args.overwrite
        )
        
        if results["processed"] == 0 and results["total_clips"] > 0:
            logger.warning("新しく処理されたクリップがありません")
            if not args.overwrite:
                logger.info("既存ファイルを上書きする場合は --overwrite オプションを使用してください")
        
        # 妥当性チェック
        if args.validate and results["processed"] > 0:
            logger.info("生成されたアノテーションファイルの妥当性をチェック中...")
            is_valid = generator.validate_generated_annotations()
            if is_valid:
                logger.info("すべてのアノテーションファイルが有効です")
            else:
                logger.error("一部のアノテーションファイルに問題があります")
                return 1
        
        # 結果の要約
        logger.info(f"処理完了:")
        logger.info(f"  総クリップ数: {results['total_clips']}")
        logger.info(f"  新規生成: {results['processed']}")
        logger.info(f"  スキップ: {results['skipped']}")
        logger.info(f"  失敗: {results['failed']}")
        
        if results["failed"] > 0:
            logger.warning("一部のクリップでエラーが発生しました")
            return 1
        
        if results["processed"] > 0:
            logger.info("空のアノテーションJSON生成が正常に完了しました")
            logger.info(f"Webアノテーションツールで {annotations_dir} 内のファイルを編集できます")
        
        return 0
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 