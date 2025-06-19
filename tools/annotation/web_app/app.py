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
from fastapi import FastAPI, HTTPException, UploadFile, File
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


# グローバル設定
class AppConfig:
    """アプリケーション設定"""
    def __init__(self):
        self.data_dir = Path("data/annotation_workspace")
        self.clips_dir = self.data_dir / "clips"
        self.annotations_dir = self.data_dir / "annotations"
        self.videos_dir = self.data_dir / "videos"
        
        # ディレクトリの作成
        for dir_path in [self.data_dir, self.clips_dir, self.annotations_dir, self.videos_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


config = AppConfig()

# 静的ファイルの提供（フロントエンド用）
if os.path.exists("tools/annotation/web_app/static"):
    app.mount("/static", StaticFiles(directory="tools/annotation/web_app/static"), name="static")


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


# アノテーション管理インスタンス
annotation_manager = AnnotationManager(config)


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