"""
高度拡張性を持つストリーミング処理パイプライン - DirectoryInputHandler

ディレクトリ内の画像ファイルからフレームを読み込むInputHandler実装。
"""

import cv2
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

from ..core.interfaces import InputHandler, ItemId

logger = logging.getLogger(__name__)


class DirectoryInputHandler(InputHandler):
    """
    ディレクトリ内の画像ファイルからフレームを読み込むInputHandler。
    
    指定されたディレクトリ内の画像ファイルを順次読み込みます。
    """
    
    # サポートする画像拡張子
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, directory_path: Union[str, Path], 
                 file_pattern: str = "*", 
                 sort_files: bool = True):
        """
        Args:
            directory_path: 画像ファイルがあるディレクトリのパス
            file_pattern: ファイル名のパターン（glob形式）
            sort_files: ファイルをソートするかどうか
        """
        self.directory_path = Path(directory_path)
        self.file_pattern = file_pattern 
        self.sort_files = sort_files
        self.image_files = []
        self._properties = None
        
        # ディレクトリの存在確認
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")
        
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.directory_path}")
        
        # 画像ファイルリストの構築
        self._build_file_list()
    
    def _build_file_list(self) -> None:
        """画像ファイルのリストを構築します。"""
        try:
            # パターンに一致するファイルを取得
            all_files = list(self.directory_path.glob(self.file_pattern))
            
            # 画像ファイルのみをフィルタリング
            self.image_files = [
                f for f in all_files 
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            
            # ソートが必要な場合
            if self.sort_files:
                self.image_files.sort()
            
            logger.info(f"Found {len(self.image_files)} image files in {self.directory_path}")
            
            if not self.image_files:
                logger.warning(f"No supported image files found in {self.directory_path}")
        
        except Exception as e:
            logger.error(f"Error building file list: {e}")
            raise
    
    def __iter__(self) -> Iterator[Tuple[ItemId, Any]]:
        """
        画像ファイルのイテレータを返します。
        
        Yields:
            Tuple[int, np.ndarray]: (画像インデックス, 画像データ)
        """
        try:
            for idx, image_path in enumerate(self.image_files):
                try:
                    # 画像を読み込み
                    image = cv2.imread(str(image_path))
                    
                    if image is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    yield idx, image
                
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error iterating image files: {e}")
            raise
    
    def get_properties(self) -> Dict[str, Any]:
        """
        ディレクトリ内画像ファイルのプロパティを返します。
        
        Returns:
            Dict[str, Any]: 画像ファイル群のメタ情報
        """
        if self._properties is None:
            self._properties = self._load_properties()
        
        return self._properties
    
    def _load_properties(self) -> Dict[str, Any]:
        """画像ファイル群のプロパティを読み込みます。"""
        try:
            properties = {
                "source_path": str(self.directory_path),
                "source_type": "directory",
                "file_pattern": self.file_pattern,
                "total_files": len(self.image_files),
                "effective_total_frames": len(self.image_files),
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS)
            }
            
            # 最初の画像から形状情報を取得
            if self.image_files:
                try:
                    first_image = cv2.imread(str(self.image_files[0]))
                    if first_image is not None:
                        properties.update({
                            "height": first_image.shape[0],
                            "width": first_image.shape[1],
                            "channels": first_image.shape[2] if len(first_image.shape) > 2 else 1
                        })
                except Exception as e:
                    logger.warning(f"Failed to get image properties from first file: {e}")
            
            return properties
        
        except Exception as e:
            logger.error(f"Error loading directory properties: {e}")
            # フォールバック値を返す
            return {
                "source_path": str(self.directory_path),
                "source_type": "directory",
                "file_pattern": self.file_pattern,
                "total_files": len(self.image_files),
                "effective_total_frames": len(self.image_files),
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
                "height": 0,
                "width": 0,
                "channels": 3
            }
    
    def get_file_list(self) -> List[Path]:
        """
        処理対象の画像ファイルリストを返します。
        
        Returns:
            List[Path]: 画像ファイルのパスのリスト
        """
        return self.image_files.copy()
    
    def close(self) -> None:
        """リソースを解放します（この実装では何もしません）。"""
        pass 