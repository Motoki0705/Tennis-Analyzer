#!/usr/bin/env python
"""
テニスイベントアノテーション用 Web Application Backend

FastAPIを使用したWebベースのアノテーションツールです。
フロントエンドのReactアプリケーションと連携し、
効率的なアノテーション作業を支援します。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import uvicorn
import asyncio
import subprocess
import uuid
import shutil
from datetime import datetime
import aiofiles
import ffmpeg
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import os

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPIアプリの初期化
app = FastAPI(
    title="Tennis Event Annotation Tool",
    description="テニスイベント検出用高効率アノテーションシステム",
    version="1.0.0"
)

# CORS設定（開発環境用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# データモデル定義
class BallAnnotation(BaseModel):
    """ボールアノテーション"""
    keypoint: Optional[List[float]] = Field(None, description="[x, y] 座標")
    visibility: int = Field(0, description="0: なし, 1: 隠れている, 2: 見える")
    is_interpolated: bool = Field(False, description="補間により生成されたか")


class FrameAnnotation(BaseModel):
    """フレームアノテーション"""
    frame_number: int = Field(description="フレーム番号")
    ball: BallAnnotation = Field(description="ボールアノテーション")
    event_status: int = Field(0, description="0: なし, 1: ヒット, 2: バウンド")


class ClipInfo(BaseModel):
    """クリップ情報"""
    source_video: str = Field(description="元動画のパス")
    clip_name: str = Field(description="クリップ名")
    clip_path: str = Field(description="クリップファイルパス")
    fps: float = Field(description="FPS")
    width: int = Field(description="動画幅")
    height: int = Field(description="動画高さ")


class AnnotationData(BaseModel):
    """アノテーションデータ全体"""
    clip_info: ClipInfo = Field(description="クリップ情報")
    frames: List[FrameAnnotation] = Field(description="フレームアノテーションリスト")


class ClipListItem(BaseModel):
    """クリップリスト項目"""
    clip_name: str = Field(description="クリップ名")
    status: str = Field(description="ステータス: 'pending', 'in_progress', 'completed'")
    has_annotations: bool = Field(description="アノテーションが存在するか")
    clip_path: str = Field(description="クリップファイルパス")


class VideoMetadata(BaseModel):
    """動画メタデータ"""
    filename: str = Field(description="ファイル名")
    file_path: str = Field(description="ファイルパス")
    duration: float = Field(description="長さ（秒）")
    fps: float = Field(description="FPS")
    width: int = Field(description="幅")
    height: int = Field(description="高さ")
    total_frames: int = Field(description="総フレーム数")
    file_size: int = Field(description="ファイルサイズ（バイト）")
    upload_date: str = Field(description="アップロード日時")


class ClipRequest(BaseModel):
    """クリップ生成リクエスト"""
    video_path: str = Field(description="元動画のパス")
    clip_name: str = Field(description="クリップ名")
    start_time: float = Field(description="開始時間（秒）")
    end_time: float = Field(description="終了時間（秒）")


class ClipCreationTask(BaseModel):
    """クリップ生成タスク"""
    task_id: str = Field(description="タスクID")
    video_path: str = Field(description="元動画のパス")
    clip_name: str = Field(description="クリップ名")
    start_time: float = Field(description="開始時間（秒）")
    end_time: float = Field(description="終了時間（秒）")
    status: str = Field(description="ステータス: 'pending', 'processing', 'completed', 'failed'")
    progress: float = Field(default=0.0, description="進捗（0.0-1.0）")
    error_message: Optional[str] = Field(None, description="エラーメッセージ")
    created_at: str = Field(description="作成日時")
    completed_at: Optional[str] = Field(None, description="完了日時")


# グローバル設定
class AppConfig:
    """アプリケーション設定"""
    def __init__(self):
        self.data_dir = Path("data/annotation_workspace")
        self.clips_dir = self.data_dir / "clips"
        self.annotations_dir = self.data_dir / "annotations"
        self.videos_dir = self.data_dir / "videos"
        self.raw_videos_dir = self.data_dir / "raw_videos"  # 新規追加：元動画ディレクトリ
        self.temp_dir = self.data_dir / "temp"  # 新規追加：一時ファイル用
        
        # 許可する動画形式
        self.allowed_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # アップロードファイルサイズ制限（バイト）
        self.max_upload_size = 10 * 1024 * 1024 * 1024  # 10GB
        
        # ディレクトリの作成
        for dir_path in [self.data_dir, self.clips_dir, self.annotations_dir, 
                         self.videos_dir, self.raw_videos_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


config = AppConfig()

# グローバル変数：クリップ生成タスクの管理
clip_creation_tasks: Dict[str, ClipCreationTask] = {}

# 静的ファイルの提供（フロントエンド用）
if os.path.exists("tools/annotation/web_app/static"):
    app.mount("/static", StaticFiles(directory="tools/annotation/web_app/static"), name="static")


class VideoProcessor:
    """動画処理クラス"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """
        動画のメタデータを取得
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            動画メタデータ
        """
        try:
            # OpenCVで動画情報を取得
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"動画ファイルを開けません: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            duration = frame_count / fps if fps > 0 else 0
            file_size = video_path.stat().st_size
            
            return VideoMetadata(
                filename=video_path.name,
                file_path=str(video_path),
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=frame_count,
                file_size=file_size,
                upload_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"動画メタデータ取得エラー {video_path}: {e}")
            raise HTTPException(status_code=400, detail=f"動画メタデータ取得に失敗しました: {str(e)}")
    
    async def create_clip(self, video_path: str, clip_name: str, start_time: float, end_time: float) -> str:
        """
        FFmpegを使用してクリップを生成
        
        Args:
            video_path: 元動画のパス
            clip_name: クリップ名
            start_time: 開始時間（秒）
            end_time: 終了時間（秒）
            
        Returns:
            生成されたクリップのパス
        """
        try:
            input_path = Path(video_path)
            if not input_path.exists():
                raise FileNotFoundError(f"元動画が見つかりません: {video_path}")
            
            # 出力パスを設定
            output_path = self.config.clips_dir / f"{clip_name}.mp4"
            
            # FFmpegでクリップを生成
            (
                ffmpeg
                .input(str(input_path), ss=start_time, t=end_time - start_time)
                .output(str(output_path), vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            if not output_path.exists():
                raise Exception("クリップファイルの生成に失敗しました")
            
            self.logger.info(f"クリップ生成完了: {output_path}")
            return str(output_path)
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"FFmpegエラー: {error_message}")
            raise Exception(f"動画処理エラー: {error_message}")
        except Exception as e:
            self.logger.error(f"クリップ生成エラー: {e}")
            raise Exception(f"クリップ生成に失敗しました: {str(e)}")
    
    def validate_video_file(self, file: UploadFile) -> bool:
        """
        アップロードされた動画ファイルの検証
        
        Args:
            file: アップロードファイル
            
        Returns:
            検証結果
        """
        try:
            # ファイル拡張子チェック
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in self.config.allowed_video_extensions:
                raise ValueError(f"サポートされていないファイル形式です: {file_extension}")
            
            # ファイルサイズチェック
            if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
                file.file.seek(0, 2)  # ファイル末尾に移動
                file_size = file.file.tell()
                file.file.seek(0)  # ファイル先頭に戻る
                
                if file_size > self.config.max_upload_size:
                    raise ValueError(f"ファイルサイズが上限を超えています: {file_size} bytes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"動画ファイル検証エラー: {e}")
            raise HTTPException(status_code=400, detail=str(e))


class AnnotationManager:
    """アノテーション管理クラス"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_clip_list(self) -> List[ClipListItem]:
        """
        アノテーション対象のクリップリストを取得（手動配置クリップ対応）
        
        Returns:
            クリップ情報のリスト
        """
        clips = []
        
        # 手動配置されたクリップを、アノテーションJSONから取得
        if not self.config.annotations_dir.exists():
            self.logger.warning(f"アノテーションディレクトリが存在しません: {self.config.annotations_dir}")
            return clips
        
        for annotation_file in self.config.annotations_dir.glob("*.json"):
            clip_name = annotation_file.stem
            
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                # クリップファイルパスの取得
                clip_info = annotation_data.get("clip_info", {})
                recorded_path = clip_info.get("clip_path", "")
                
                # 実際のクリップファイルを検索
                clip_file = None
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                
                # 1. 記録されたパスを最初に確認
                if recorded_path and Path(recorded_path).exists():
                    clip_file = Path(recorded_path)
                else:
                    # 2. クリップディレクトリ内で検索
                    for ext in video_extensions:
                        test_path = self.config.clips_dir / f"{clip_name}{ext}"
                        if test_path.exists():
                            clip_file = test_path
                            break
                
                # ステータス判定（手動アノテーション重視）
                frames = annotation_data.get("frames", [])
                if frames:
                    # 手動でアノテーションが加えられたフレーム数をカウント
                    annotated_frames = sum(
                        1 for frame in frames 
                        if (frame.get("ball", {}).get("keypoint") is not None and 
                            frame.get("ball", {}).get("visibility", 0) > 0) or 
                           frame.get("event_status", 0) > 0
                    )
                    
                    if annotated_frames >= len(frames) * 0.8:  # 80%以上完了で完了とみなす
                        status = "completed"
                        has_annotations = True
                    elif annotated_frames > 0:
                        status = "in_progress"
                        has_annotations = True
                    else:
                        status = "pending"
                        has_annotations = False
                else:
                    status = "pending"
                    has_annotations = False
                
                # クリップファイルが見つからない場合は警告を出すが、リストには含める
                if not clip_file:
                    clip_file = self.config.clips_dir / f"{clip_name}.mp4"  # デフォルトパス
                    self.logger.warning(f"クリップファイルが見つかりません: {clip_name}")
                
                clips.append(ClipListItem(
                    clip_name=clip_name,
                    status=status,
                    has_annotations=has_annotations,
                    clip_path=str(clip_file)
                ))
                
            except Exception as e:
                self.logger.error(f"アノテーションファイル読み込みエラー {annotation_file}: {e}")
                # エラーがあってもリストには含める
                clips.append(ClipListItem(
                    clip_name=clip_name,
                    status="error",
                    has_annotations=False,
                    clip_path=str(self.config.clips_dir / f"{clip_name}.mp4")
                ))
        
        # ステータス順、名前順でソート
        clips.sort(key=lambda x: (x.status != "pending", x.status != "in_progress", x.clip_name))
        
        self.logger.info(f"クリップリスト生成完了: {len(clips)} 件")
        return clips
    
    def load_annotation(self, clip_name: str) -> AnnotationData:
        """
        指定されたクリップのアノテーションを読み込み
        
        Args:
            clip_name: クリップ名
            
        Returns:
            アノテーションデータ
        """
        annotation_file = self.config.annotations_dir / f"{clip_name}.json"
        
        if not annotation_file.exists():
            raise HTTPException(status_code=404, detail=f"アノテーションファイルが見つかりません: {clip_name}")
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Pydanticモデルに変換
            return AnnotationData(**raw_data)
        
        except Exception as e:
            self.logger.error(f"アノテーション読み込みエラー {annotation_file}: {e}")
            raise HTTPException(status_code=500, detail=f"アノテーション読み込みに失敗しました: {e}")
    
    def save_annotation(self, clip_name: str, annotation_data: AnnotationData) -> bool:
        """
        アノテーションデータを保存
        
        Args:
            clip_name: クリップ名
            annotation_data: 保存するアノテーションデータ
            
        Returns:
            保存成功フラグ
        """
        annotation_file = self.config.annotations_dir / f"{clip_name}.json"
        
        try:
            # バックアップファイルの作成
            if annotation_file.exists():
                backup_file = annotation_file.with_suffix('.json.bak')
                import shutil
                shutil.copy2(annotation_file, backup_file)
            
            # JSON形式で保存
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data.dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"アノテーション保存完了: {annotation_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"アノテーション保存エラー {annotation_file}: {e}")
            raise HTTPException(status_code=500, detail=f"アノテーション保存に失敗しました: {e}")
    
    def create_empty_annotation_for_clip(self, clip_name: str, clip_path: str, source_video: str) -> bool:
        """
        クリップ用の空のアノテーションファイルを生成
        
        Args:
            clip_name: クリップ名
            clip_path: クリップファイルパス
            source_video: 元動画のパス
            
        Returns:
            生成成功フラグ
        """
        try:
            # クリップの動画情報を取得
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                raise ValueError(f"クリップファイルを開けません: {clip_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # クリップ情報を作成
            clip_info = ClipInfo(
                source_video=source_video,
                clip_name=clip_name,
                clip_path=clip_path,
                fps=fps,
                width=width,
                height=height
            )
            
            # 各フレーム用の空のアノテーションを作成
            frames = []
            for frame_num in range(frame_count):
                frame_annotation = FrameAnnotation(
                    frame_number=frame_num,
                    ball=BallAnnotation(
                        keypoint=None,
                        visibility=0,
                        is_interpolated=False
                    ),
                    event_status=0
                )
                frames.append(frame_annotation)
            
            # アノテーションデータを作成
            annotation_data = AnnotationData(
                clip_info=clip_info,
                frames=frames
            )
            
            # ファイルに保存
            annotation_file = self.config.annotations_dir / f"{clip_name}.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data.model_dump(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"空のアノテーションファイルを生成しました: {annotation_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"空のアノテーション生成エラー {clip_name}: {e}")
            return False


# インスタンス作成
annotation_manager = AnnotationManager(config)
video_processor = VideoProcessor(config)


# APIエンドポイント定義

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Tennis Event Annotation Tool API", "version": "1.0.0"}


@app.get("/api/clips", response_model=List[ClipListItem])
async def get_clips():
    """
    アノテーション対象のクリップリストを取得
    
    Returns:
        クリップ情報のリスト
    """
    try:
        clips = annotation_manager.get_clip_list()
        logger.info(f"クリップリスト取得: {len(clips)} 件")
        return clips
    except Exception as e:
        logger.error(f"クリップリスト取得エラー: {e}")
        raise HTTPException(status_code=500, detail="クリップリスト取得に失敗しました")


@app.get("/api/annotation/{clip_name}", response_model=AnnotationData)
async def get_annotation(clip_name: str):
    """
    指定されたクリップのアノテーションを取得
    
    Args:
        clip_name: クリップ名
        
    Returns:
        アノテーションデータ
    """
    try:
        annotation_data = annotation_manager.load_annotation(clip_name)
        logger.info(f"アノテーション取得: {clip_name}")
        return annotation_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"アノテーション取得エラー {clip_name}: {e}")
        raise HTTPException(status_code=500, detail="アノテーション取得に失敗しました")


@app.post("/api/annotation/{clip_name}")
async def save_annotation(clip_name: str, annotation_data: AnnotationData):
    """
    アノテーションデータを保存
    
    Args:
        clip_name: クリップ名
        annotation_data: 保存するアノテーションデータ
        
    Returns:
        保存結果
    """
    try:
        success = annotation_manager.save_annotation(clip_name, annotation_data)
        if success:
            return {"message": "アノテーション保存完了", "clip_name": clip_name}
        else:
            raise HTTPException(status_code=500, detail="アノテーション保存に失敗しました")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"アノテーション保存エラー {clip_name}: {e}")
        raise HTTPException(status_code=500, detail="アノテーション保存に失敗しました")


@app.get("/api/video/{clip_name}")
async def get_video(clip_name: str):
    """
    クリップ動画ファイルを取得（複数形式対応）
    
    Args:
        clip_name: クリップ名
        
    Returns:
        動画ファイル
    """
    # 対応する動画形式を検索
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    clip_file = None
    
    for ext in video_extensions:
        test_path = config.clips_dir / f"{clip_name}{ext}"
        if test_path.exists():
            clip_file = test_path
            break
    
    # アノテーションファイルから記録されたパスも確認
    if not clip_file:
        annotation_file = config.annotations_dir / f"{clip_name}.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                recorded_path = annotation_data.get("clip_info", {}).get("clip_path", "")
                if recorded_path and Path(recorded_path).exists():
                    clip_file = Path(recorded_path)
            except Exception:
                pass
    
    if not clip_file or not clip_file.exists():
        raise HTTPException(status_code=404, detail=f"動画ファイルが見つかりません: {clip_name}")
    
    # メディアタイプの決定
    extension = clip_file.suffix.lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.flv': 'video/x-flv',
        '.wmv': 'video/x-ms-wmv'
    }
    media_type = media_type_map.get(extension, 'video/mp4')
    
    return FileResponse(
        str(clip_file),
        media_type=media_type,
        filename=clip_file.name
    )


@app.post("/api/interpolate")
async def interpolate_keypoints(
    clip_name: str,
    start_frame: int,
    end_frame: int,
    start_keypoint: List[float],
    end_keypoint: List[float]
):
    """
    指定された範囲のキーポイントを線形補間
    
    Args:
        clip_name: クリップ名
        start_frame: 開始フレーム
        end_frame: 終了フレーム
        start_keypoint: 開始キーポイント [x, y]
        end_keypoint: 終了キーポイント [x, y]
        
    Returns:
        補間されたキーポイントのリスト
    """
    try:
        if start_frame >= end_frame:
            raise HTTPException(status_code=400, detail="無効なフレーム範囲です")
        
        interpolated_points = []
        frame_count = end_frame - start_frame + 1
        
        for i in range(frame_count):
            t = i / (frame_count - 1) if frame_count > 1 else 0
            x = start_keypoint[0] + t * (end_keypoint[0] - start_keypoint[0])
            y = start_keypoint[1] + t * (end_keypoint[1] - start_keypoint[1])
            
            interpolated_points.append({
                "frame_number": start_frame + i,
                "keypoint": [x, y],
                "is_interpolated": True
            })
        
        logger.info(f"キーポイント補間: {clip_name} frames {start_frame}-{end_frame}")
        return {"interpolated_points": interpolated_points}
        
    except Exception as e:
        logger.error(f"キーポイント補間エラー: {e}")
        raise HTTPException(status_code=500, detail="キーポイント補間に失敗しました")


@app.get("/api/stats")
async def get_annotation_stats():
    """
    アノテーション統計情報を取得
    
    Returns:
        統計情報
    """
    try:
        clips = annotation_manager.get_clip_list()
        
        stats = {
            "total_clips": len(clips),
            "pending": len([c for c in clips if c.status == "pending"]),
            "in_progress": len([c for c in clips if c.status == "in_progress"]),
            "completed": len([c for c in clips if c.status == "completed"]),
        }
        
        # 進捗率
        if stats["total_clips"] > 0:
            stats["completion_rate"] = stats["completed"] / stats["total_clips"] * 100
        else:
            stats["completion_rate"] = 0
        
        return stats
        
    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        raise HTTPException(status_code=500, detail="統計情報取得に失敗しました")


# === 新規機能: 動画アップロードとクリップ生成 ===

@app.get("/api/raw_videos", response_model=List[VideoMetadata])
async def get_raw_videos():
    """
    アップロード済み元動画のリストを取得
    
    Returns:
        動画メタデータのリスト
    """
    try:
        videos = []
        
        for video_file in config.raw_videos_dir.glob("*"):
            if video_file.suffix.lower() in config.allowed_video_extensions:
                try:
                    metadata = video_processor.get_video_metadata(video_file)
                    videos.append(metadata)
                except Exception as e:
                    logger.warning(f"動画メタデータ取得スキップ {video_file}: {e}")
        
        # ファイル名でソート
        videos.sort(key=lambda x: x.filename)
        logger.info(f"元動画リスト取得: {len(videos)} 件")
        return videos
        
    except Exception as e:
        logger.error(f"元動画リスト取得エラー: {e}")
        raise HTTPException(status_code=500, detail="元動画リスト取得に失敗しました")


@app.post("/api/upload_video")
async def upload_video(video_file: UploadFile = File(...)):
    """
    動画ファイルをアップロード
    
    Args:
        video_file: アップロードする動画ファイル
        
    Returns:
        アップロード結果とメタデータ
    """
    try:
        # ファイル検証
        video_processor.validate_video_file(video_file)
        
        # ファイル名の安全化
        safe_filename = "".join(c for c in video_file.filename if c.isalnum() or c in "._-")
        if not safe_filename:
            safe_filename = f"uploaded_video_{uuid.uuid4().hex[:8]}.mp4"
        
        # 保存パス
        save_path = config.raw_videos_dir / safe_filename
        
        # 既存ファイルがある場合は番号を付加
        counter = 1
        while save_path.exists():
            name_parts = safe_filename.rsplit('.', 1)
            if len(name_parts) == 2:
                new_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
            else:
                new_name = f"{safe_filename}_{counter}"
            save_path = config.raw_videos_dir / new_name
            counter += 1
        
        # ファイル保存
        async with aiofiles.open(save_path, 'wb') as f:
            content = await video_file.read()
            await f.write(content)
        
        # メタデータ取得
        metadata = video_processor.get_video_metadata(save_path)
        
        logger.info(f"動画アップロード完了: {save_path}")
        return {
            "message": "動画アップロード完了",
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"動画アップロードエラー: {e}")
        raise HTTPException(status_code=500, detail=f"動画アップロードに失敗しました: {str(e)}")


@app.post("/api/create_clip")
async def create_clip_endpoint(clip_request: ClipRequest, background_tasks: BackgroundTasks):
    """
    クリップ生成を開始（非同期）
    
    Args:
        clip_request: クリップ生成リクエスト
        background_tasks: バックグラウンドタスク
        
    Returns:
        タスクID
    """
    try:
        # 入力検証
        if not Path(clip_request.video_path).exists():
            raise HTTPException(status_code=404, detail="元動画ファイルが見つかりません")
        
        if clip_request.start_time >= clip_request.end_time:
            raise HTTPException(status_code=400, detail="無効な時間範囲です")
        
        # 重複するクリップ名をチェック
        existing_clip = config.clips_dir / f"{clip_request.clip_name}.mp4"
        if existing_clip.exists():
            raise HTTPException(status_code=400, detail="同名のクリップが既に存在します")
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # タスクを作成
        task = ClipCreationTask(
            task_id=task_id,
            video_path=clip_request.video_path,
            clip_name=clip_request.clip_name,
            start_time=clip_request.start_time,
            end_time=clip_request.end_time,
            status="pending",
            created_at=datetime.now().isoformat()
        )
        
        # グローバル変数に保存
        clip_creation_tasks[task_id] = task
        
        # バックグラウンドでクリップ生成を実行
        background_tasks.add_task(process_clip_creation, task_id)
        
        logger.info(f"クリップ生成タスク開始: {task_id} - {clip_request.clip_name}")
        return {"task_id": task_id, "message": "クリップ生成を開始しました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"クリップ生成開始エラー: {e}")
        raise HTTPException(status_code=500, detail=f"クリップ生成開始に失敗しました: {str(e)}")


@app.get("/api/clip_tasks/{task_id}", response_model=ClipCreationTask)
async def get_clip_task_status(task_id: str):
    """
    クリップ生成タスクの状態を取得
    
    Args:
        task_id: タスクID
        
    Returns:
        タスク状態
    """
    if task_id not in clip_creation_tasks:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    
    return clip_creation_tasks[task_id]


@app.get("/api/clip_tasks", response_model=List[ClipCreationTask])
async def get_all_clip_tasks():
    """
    全てのクリップ生成タスクを取得
    
    Returns:
        タスクリスト
    """
    tasks = list(clip_creation_tasks.values())
    # 作成日時で降順ソート
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return tasks


async def process_clip_creation(task_id: str):
    """
    クリップ生成を実際に処理する非同期関数
    
    Args:
        task_id: タスクID
    """
    task = clip_creation_tasks.get(task_id)
    if not task:
        logger.error(f"タスクが見つかりません: {task_id}")
        return
    
    try:
        # ステータス更新
        task.status = "processing"
        task.progress = 0.1
        
        # クリップ生成
        clip_path = await video_processor.create_clip(
            task.video_path,
            task.clip_name,
            task.start_time,
            task.end_time
        )
        
        task.progress = 0.8
        
        # 空のアノテーションファイル生成
        annotation_created = annotation_manager.create_empty_annotation_for_clip(
            task.clip_name,
            clip_path,
            task.video_path
        )
        
        if not annotation_created:
            logger.warning(f"アノテーションファイル生成に失敗: {task.clip_name}")
        
        # 完了
        task.status = "completed"
        task.progress = 1.0
        task.completed_at = datetime.now().isoformat()
        
        logger.info(f"クリップ生成完了: {task.clip_name}")
        
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now().isoformat()
        logger.error(f"クリップ生成失敗 {task.clip_name}: {e}")


@app.get("/api/raw_video/{filename}")
async def get_raw_video(filename: str):
    """
    元動画ファイルを取得（プレビュー用）
    
    Args:
        filename: ファイル名
        
    Returns:
        動画ファイル
    """
    video_path = config.raw_videos_dir / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="動画ファイルが見つかりません")
    
    # セキュリティチェック（ディレクトリトラバーサル対策）
    try:
        video_path.resolve().relative_to(config.raw_videos_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="アクセスが拒否されました")
    
    # メディアタイプの決定
    extension = video_path.suffix.lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.flv': 'video/x-flv',
        '.wmv': 'video/x-ms-wmv'
    }
    media_type = media_type_map.get(extension, 'video/mp4')
    
    return FileResponse(
        str(video_path),
        media_type=media_type,
        filename=video_path.name
    )


# ヘルスチェック
@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "service": "Tennis Event Annotation Tool"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tennis Event Annotation Web App")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--data_dir", default="data/annotation_workspace", help="Data directory")
    
    args = parser.parse_args()
    
    # データディレクトリの設定
    config.data_dir = Path(args.data_dir)
    config.clips_dir = config.data_dir / "clips"
    config.annotations_dir = config.data_dir / "annotations"
    
    # ディレクトリの作成
    for dir_path in [config.data_dir, config.clips_dir, config.annotations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"サーバー起動: http://{args.host}:{args.port}")
    logger.info(f"データディレクトリ: {config.data_dir}")
    
    uvicorn.run(app, host=args.host, port=args.port) 