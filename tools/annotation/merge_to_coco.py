#!/usr/bin/env python
"""
アノテーションマージスクリプト

複数の中間アノテーションJSONファイルを単一のCOCO形式データセットに変換・統合します。
設計書の「3.2. 最終出力フォーマット (COCO-like JSON)」仕様に準拠します。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import cv2
import os
from collections import defaultdict

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnotationMerger:
    """
    中間アノテーションファイルをCOCO形式に変換・統合するクラス
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # グローバルカウンター
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        
        # COCO形式のベース構造
        self.coco_data = {
            "info": {
                "description": "Tennis Event Detection Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Tennis Analyzer Team",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "ball",
                    "supercategory": "object",
                    "keypoints": ["ball_center"],
                    "skeleton": []
                }
            ],
            "images": [],
            "annotations": []
        }
    
    def find_annotation_files(self, input_dir: Path) -> List[Path]:
        """
        入力ディレクトリから中間アノテーションファイルを検索
        
        Args:
            input_dir: 検索対象ディレクトリ
            
        Returns:
            見つかったJSONファイルのパスリスト
        """
        annotation_files = []
        
        # clip_*.json パターンでファイルを検索
        for json_file in input_dir.rglob("clip_*.json"):
            if json_file.is_file():
                annotation_files.append(json_file)
                self.logger.debug(f"アノテーションファイル発見: {json_file}")
        
        # ファイル名でソート（一貫した順序で処理）
        annotation_files.sort(key=lambda x: x.name)
        
        self.logger.info(f"合計 {len(annotation_files)} 個のアノテーションファイルを発見しました")
        return annotation_files
    
    def load_annotation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        中間アノテーションファイルを読み込み
        
        Args:
            file_path: アノテーションファイルのパス
            
        Returns:
            読み込まれたアノテーションデータ、エラー時はNone
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 必要なフィールドの存在チェック
            required_fields = ["clip_info", "frames"]
            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"必要なフィールドがありません: {field} in {file_path}")
                    return None
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONパースエラー {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"ファイル読み込みエラー {file_path}: {e}")
            return None
    
    def extract_frames_from_clip(self, clip_path: str) -> Tuple[bool, List[str]]:
        """
        クリップから個別フレーム画像を抽出（必要に応じて）
        
        Args:
            clip_path: クリップ動画のパス
            
        Returns:
            (成功フラグ, フレームファイルパスのリスト)
        """
        if not os.path.exists(clip_path):
            self.logger.warning(f"クリップファイルが見つかりません: {clip_path}")
            return False, []
        
        # フレーム保存用ディレクトリ
        clip_name = Path(clip_path).stem
        frames_dir = Path("temp_frames") / clip_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        
        try:
            cap = cv2.VideoCapture(clip_path)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                frame_idx += 1
            
            cap.release()
            self.logger.info(f"フレーム抽出完了: {len(frame_paths)} フレーム from {clip_path}")
            return True, frame_paths
            
        except Exception as e:
            self.logger.error(f"フレーム抽出エラー {clip_path}: {e}")
            return False, []
    
    def create_bbox_from_keypoint(self, keypoint: List[float], bbox_size: int = 10) -> List[float]:
        """
        キーポイントから固定サイズのBBoxを生成
        
        Args:
            keypoint: [x, y] 座標
            bbox_size: BBoxのサイズ（一辺の長さ）
            
        Returns:
            [x, y, width, height] 形式のBBox
        """
        if not keypoint or len(keypoint) < 2:
            return [0, 0, bbox_size, bbox_size]
        
        x, y = keypoint
        half_size = bbox_size // 2
        
        return [
            max(0, x - half_size),  # x
            max(0, y - half_size),  # y
            bbox_size,              # width
            bbox_size               # height
        ]
    
    def process_single_annotation_file(self, annotation_data: Dict[str, Any]) -> bool:
        """
        単一の中間アノテーションファイルを処理してCOCO形式に変換
        
        Args:
            annotation_data: 中間アノテーションデータ
            
        Returns:
            処理成功フラグ
        """
        try:
            clip_info = annotation_data["clip_info"]
            frames_data = annotation_data["frames"]
            
            clip_name = clip_info["clip_name"]
            clip_path = clip_info["clip_path"]
            fps = clip_info.get("fps", 30.0)
            width = clip_info.get("width", 1280)
            height = clip_info.get("height", 720)
            
            self.logger.info(f"処理中: {clip_name} ({len(frames_data)} フレーム)")
            
            # クリップが実際に存在するかチェック
            if not os.path.exists(clip_path):
                self.logger.warning(f"クリップファイルが見つかりません: {clip_path}")
                # 仮想的なパスを生成（テスト用）
                frames_paths = [f"virtual_frame_{i:06d}.jpg" for i in range(len(frames_data))]
                use_virtual_paths = True
            else:
                # 実際のフレーム抽出
                success, frames_paths = self.extract_frames_from_clip(clip_path)
                if not success:
                    return False
                use_virtual_paths = False
            
            # 各フレームを処理
            for frame_data in frames_data:
                frame_number = frame_data["frame_number"]
                ball_data = frame_data.get("ball", {})
                event_status = frame_data.get("event_status", 0)
                
                # 画像エントリの作成
                if frame_number < len(frames_paths):
                    image_path = frames_paths[frame_number]
                else:
                    self.logger.warning(f"フレーム {frame_number} のパスが見つかりません")
                    continue
                
                image_entry = {
                    "id": self.image_id_counter,
                    "file_name": os.path.basename(image_path),
                    "width": width,
                    "height": height,
                    "license": 1,
                    "clip_name": clip_name,
                    "frame_number": frame_number,
                    "source_video": clip_info.get("source_video", ""),
                    "fps": fps
                }
                self.coco_data["images"].append(image_entry)
                
                # ボールアノテーションの処理
                keypoint = ball_data.get("keypoint")
                visibility = ball_data.get("visibility", 0)
                is_interpolated = ball_data.get("is_interpolated", False)
                
                if keypoint and visibility > 0:
                    # キーポイント形式: [x, y, visibility]
                    keypoints = [keypoint[0], keypoint[1], visibility]
                    
                    # BBoxの自動生成
                    bbox = self.create_bbox_from_keypoint(keypoint, bbox_size=10)
                    
                    # アノテーションエントリの作成
                    annotation_entry = {
                        "id": self.annotation_id_counter,
                        "image_id": self.image_id_counter,
                        "category_id": 1,  # ball category
                        "keypoints": keypoints,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],  # width * height
                        "iscrowd": 0,
                        "event_status": event_status,
                        "is_interpolated": is_interpolated
                    }
                    
                    self.coco_data["annotations"].append(annotation_entry)
                    self.annotation_id_counter += 1
                
                self.image_id_counter += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"アノテーション処理エラー: {e}")
            return False
    
    def merge_annotations(self, input_dir: Path) -> bool:
        """
        複数のアノテーションファイルをマージ
        
        Args:
            input_dir: 入力ディレクトリ
            
        Returns:
            マージ成功フラグ
        """
        annotation_files = self.find_annotation_files(input_dir)
        
        if not annotation_files:
            self.logger.error("アノテーションファイルが見つかりませんでした")
            return False
        
        processed_count = 0
        failed_count = 0
        
        for annotation_file in annotation_files:
            self.logger.info(f"処理中: {annotation_file}")
            
            # アノテーションファイルの読み込み
            annotation_data = self.load_annotation_file(annotation_file)
            if annotation_data is None:
                failed_count += 1
                continue
            
            # COCO形式への変換
            success = self.process_single_annotation_file(annotation_data)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        
        self.logger.info(f"マージ完了: 成功 {processed_count} 件, 失敗 {failed_count} 件")
        self.logger.info(f"総画像数: {len(self.coco_data['images'])}")
        self.logger.info(f"総アノテーション数: {len(self.coco_data['annotations'])}")
        
        return processed_count > 0
    
    def save_coco_dataset(self, output_file: Path) -> bool:
        """
        COCO形式データセットをファイルに保存
        
        Args:
            output_file: 出力ファイルパス
            
        Returns:
            保存成功フラグ
        """
        try:
            # 出力ディレクトリの作成
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # バックアップの作成（既存ファイルがある場合）
            if output_file.exists():
                backup_file = output_file.with_suffix('.json.bak')
                import shutil
                shutil.copy2(output_file, backup_file)
                self.logger.info(f"既存ファイルをバックアップしました: {backup_file}")
            
            # JSON形式で保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.coco_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"COCO形式データセット保存完了: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"ファイル保存エラー {output_file}: {e}")
            return False
    
    def validate_coco_dataset(self) -> bool:
        """
        生成されたCOCO形式データセットの妥当性をチェック
        
        Returns:
            妥当性チェック結果
        """
        try:
            # 基本構造のチェック
            required_keys = ["info", "licenses", "categories", "images", "annotations"]
            for key in required_keys:
                if key not in self.coco_data:
                    self.logger.error(f"必要なキーが不足しています: {key}")
                    return False
            
            # 画像とアノテーションの整合性チェック
            image_ids = {img["id"] for img in self.coco_data["images"]}
            annotation_image_ids = {ann["image_id"] for ann in self.coco_data["annotations"]}
            
            # 孤立したアノテーション（対応する画像がない）をチェック
            orphaned_annotations = annotation_image_ids - image_ids
            if orphaned_annotations:
                self.logger.warning(f"対応する画像がないアノテーション: {len(orphaned_annotations)} 件")
            
            # アノテーションの有無
            images_with_annotations = len(annotation_image_ids)
            total_images = len(image_ids)
            
            self.logger.info(f"妥当性チェック結果:")
            self.logger.info(f"  - 総画像数: {total_images}")
            self.logger.info(f"  - アノテーション付き画像数: {images_with_annotations}")
            self.logger.info(f"  - アノテーションカバー率: {images_with_annotations/total_images*100:.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"妥当性チェックエラー: {e}")
            return False
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        データセット統計情報を生成
        
        Returns:
            統計情報辞書
        """
        stats = {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "clips": defaultdict(int),
            "event_status_distribution": defaultdict(int),
            "interpolated_annotations": 0
        }
        
        # クリップ別統計
        for img in self.coco_data["images"]:
            clip_name = img.get("clip_name", "unknown")
            stats["clips"][clip_name] += 1
        
        # イベントステータス分布
        for ann in self.coco_data["annotations"]:
            event_status = ann.get("event_status", 0)
            stats["event_status_distribution"][event_status] += 1
            
            if ann.get("is_interpolated", False):
                stats["interpolated_annotations"] += 1
        
        # 辞書型に変換（JSON serializable）
        stats["clips"] = dict(stats["clips"])
        stats["event_status_distribution"] = dict(stats["event_status_distribution"])
        
        return stats


def cleanup_temp_files():
    """一時的なフレームファイルをクリーンアップ"""
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("一時ファイルをクリーンアップしました")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="中間アノテーションファイルをCOCO形式データセットに統合"
    )
    parser.add_argument(
        "--input_dir", "-i", 
        required=True, 
        help="中間アノテーションファイルが格納されたディレクトリ"
    )
    parser.add_argument(
        "--output_file", "-o", 
        required=True, 
        help="出力するCOCO形式JSONファイルのパス"
    )
    parser.add_argument(
        "--stats_file", "-s", 
        help="統計情報を出力するJSONファイルのパス（オプション）"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true", 
        help="実行後に一時ファイルを削除"
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
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    # 入力ディレクトリの存在チェック
    if not input_dir.exists():
        logger.error(f"入力ディレクトリが存在しません: {input_dir}")
        return 1
    
    logger.info(f"アノテーションマージ開始:")
    logger.info(f"  入力ディレクトリ: {input_dir}")
    logger.info(f"  出力ファイル: {output_file}")
    
    try:
        # マージ処理の実行
        merger = AnnotationMerger()
        
        # アノテーションファイルのマージ
        success = merger.merge_annotations(input_dir)
        if not success:
            logger.error("アノテーションマージに失敗しました")
            return 1
        
        # 妥当性チェック
        if not merger.validate_coco_dataset():
            logger.error("データセットの妥当性チェックに失敗しました")
            return 1
        
        # COCO形式データセットの保存
        if not merger.save_coco_dataset(output_file):
            logger.error("データセット保存に失敗しました")
            return 1
        
        # 統計情報の生成・保存
        stats = merger.generate_statistics()
        logger.info(f"データセット統計:")
        logger.info(f"  - 総画像数: {stats['total_images']}")
        logger.info(f"  - 総アノテーション数: {stats['total_annotations']}")
        logger.info(f"  - 補間アノテーション数: {stats['interpolated_annotations']}")
        logger.info(f"  - クリップ数: {len(stats['clips'])}")
        
        if args.stats_file:
            stats_file = Path(args.stats_file)
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"統計情報を保存しました: {stats_file}")
        
        # 一時ファイルのクリーンアップ
        if args.cleanup:
            cleanup_temp_files()
        
        logger.info("アノテーションマージが正常に完了しました")
        return 0
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main())