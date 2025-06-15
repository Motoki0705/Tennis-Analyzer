"""
高度拡張性を持つストリーミング処理パイプライン - CallbackOutputHandler

コールバック関数を呼び出すOutputHandler実装。
"""

import logging
from typing import Any, Callable, Dict

from ..core.interfaces import OutputHandler, ItemId

logger = logging.getLogger(__name__)


class CallbackOutputHandler(OutputHandler):
    """
    コールバック関数を呼び出すOutputHandler。
    
    処理結果をユーザー定義のコールバック関数に渡します。
    """
    
    def __init__(self, callback: Callable[[ItemId, Any, Dict[str, Any]], None],
                 start_callback: Callable[[Dict[str, Any]], None] = None,
                 finish_callback: Callable[[], None] = None):
        """
        Args:
            callback: 各結果を処理するコールバック関数
            start_callback: 出力開始時に呼ばれるコールバック関数（オプション）
            finish_callback: 出力終了時に呼ばれるコールバック関数（オプション）
        """
        self.callback = callback
        self.start_callback = start_callback
        self.finish_callback = finish_callback
        self._processed_count = 0
        self._error_count = 0
    
    def start_output(self, properties: Dict[str, Any]) -> None:
        """
        出力を開始します。
        
        Args:
            properties: 入力のプロパティ情報
        """
        try:
            self._processed_count = 0
            self._error_count = 0
            
            # 開始コールバックが設定されている場合は呼び出し
            if self.start_callback:
                self.start_callback(properties)
            
            logger.info("Started callback output handler")
        
        except Exception as e:
            logger.error(f"Error starting callback output: {e}")
            raise
    
    def handle_result(self, item_id: ItemId, 
                     original_item: Any, 
                     results: Dict[str, Any]) -> None:
        """
        処理結果をコールバック関数に渡します。
        
        Args:
            item_id: アイテムID
            original_item: 元のアイテム
            results: 各ワーカーからの処理結果
        """
        try:
            # コールバック関数を呼び出し
            self.callback(item_id, original_item, results)
            self._processed_count += 1
            
            if self._processed_count % 100 == 0:
                logger.debug(f"Processed {self._processed_count} items via callback")
        
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in callback for item {item_id}: {e}")
            
            # エラーが多すぎる場合は警告
            if self._error_count > 10 and self._error_count / self._processed_count > 0.1:
                logger.warning(f"High error rate in callback: {self._error_count} errors "
                              f"out of {self._processed_count} processed items")
    
    def finish_output(self) -> None:
        """出力を終了します。"""
        try:
            # 終了コールバックが設定されている場合は呼び出し
            if self.finish_callback:
                self.finish_callback()
            
            logger.info(f"Finished callback output handler: {self._processed_count} items processed, "
                       f"{self._error_count} errors")
        
        except Exception as e:
            logger.error(f"Error finishing callback output: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        処理統計を返します。
        
        Returns:
            処理統計の辞書
        """
        return {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "success_count": self._processed_count - self._error_count
        }
    
    def close(self) -> None:
        """リソースを解放します（この実装では何もしません）。"""
        pass 