"""
高度拡張性を持つストリーミング処理パイプライン - インターフェース定義

このモジュールは、パイプライン内の各コンポーネントの
抽象的なインターフェースを定義します。
"""

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# 型エイリアス
ItemId = Union[str, int]
TopicName = str
TaskData = Any
ResultData = Any


class InputHandler(ABC):
    """
    データソースから処理単位（データアイテム）を取り出すハンドラの基底クラス。
    
    様々な入力形式（動画ファイル、フレームリスト、ディレクトリなど）を
    統一的なインターフェースで扱えるよう抽象化します。
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[ItemId, Any]]:
        """
        データアイテムのイテレータを返します。
        
        Returns:
            Iterator[Tuple[ItemId, Any]]: (item_id, data) のタプルのイテレータ
                item_id: 一意な識別子（例: フレームインデックス）
                data: 処理対象データ（例: np.ndarrayフレーム）
        """
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        データソースのメタ情報を返します。
        
        Returns:
            Dict[str, Any]: データソースのプロパティ辞書
                例: {"fps": 30, "width": 1920, "height": 1080, "total_frames": 1000}
        """
        pass
    
    def close(self) -> None:
        """
        リソースを解放します。デフォルト実装では何もしません。
        必要に応じて継承クラスでオーバーライドしてください。
        """
        pass


class OutputHandler(ABC):
    """
    パイプラインの最終成果物を処理するハンドラの基底クラス。
    
    様々な出力形式（動画ファイル、JSON、リアルタイム表示など）を
    統一的なインターフェースで扱えるよう抽象化します。
    """
    
    @abstractmethod
    def setup(self, properties: Dict[str, Any]) -> None:
        """
        InputHandlerのプロパティを元に初期設定を行います。
        
        Args:
            properties: InputHandler.get_properties()から取得したプロパティ
        """
        pass
    
    @abstractmethod
    def handle_result(self, item_id: ItemId, final_result: Dict[str, Any], 
                     original_data: Any) -> None:
        """
        最終成果物を処理します。
        
        Args:
            item_id: アイテムの一意識別子
            final_result: パイプラインで生成された最終結果
            original_data: 元のデータ（必要に応じて）
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        リソースを解放します（例: ファイルを閉じる）。
        """
        pass


class BaseWorker(ABC):
    @abstractmethod
    def get_published_topic(self) -> TopicName: pass

    @abstractmethod
    def get_dependencies(self) -> List[TopicName]: pass

    @abstractmethod
    def submit_task(self, item_id: ItemId, task_data: TaskData,
                   dependencies: Dict[TopicName, ResultData]) -> bool: pass


class TaskManagerInterface(ABC):
    """タスクマネージャーのインターフェース"""
    
    @abstractmethod
    def setup(self, workers: List[BaseWorker]) -> None:
        """ワーカー群から依存関係グラフを構築します。"""
        pass
    
    @abstractmethod
    def submit_item(self, item_id: ItemId, data: Any) -> None:
        """新しいアイテムの処理を開始します。"""
        pass
    
    @abstractmethod
    def notify_result_available(self, item_id: ItemId, topic: TopicName) -> None:
        """結果が利用可能になったことを通知します。"""
        pass


class ResultManagerInterface(ABC):
    """結果マネージャーのインターフェース"""
    
    @abstractmethod
    def start(self) -> None:
        """結果処理ループを開始します。"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """結果処理ループを停止します。"""
        pass
    
    @abstractmethod
    def get_completed_results(self) -> Iterator[Tuple[ItemId, Dict[str, Any], Any]]:
        """完成した最終成果物のイテレータを返します。"""
        pass 