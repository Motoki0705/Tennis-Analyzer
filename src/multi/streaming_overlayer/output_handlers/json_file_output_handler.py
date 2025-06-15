"""
高度拡張性を持つストリーミング処理パイプライン - JsonFileOutputHandler

処理結果をJSONファイルに出力するOutputHandler実装。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from ..core.interfaces import OutputHandler, ItemId

logger = logging.getLogger(__name__)


class JsonFileOutputHandler(OutputHandler):
    """
    処理結果をJSONファイルに出力するOutputHandler。
    
    全ての処理結果を集約し、構造化されたJSONファイルとして保存します。
    """
    
    def __init__(self, output_path: Union[str, Path], 
                 pretty_print: bool = True,
                 include_metadata: bool = True):
        """
        Args:
            output_path: 出力JSONファイルのパス
            pretty_print: 見やすくフォーマットするかどうか
            include_metadata: メタデータを含めるかどうか
        """
        self.output_path = Path(output_path)
        self.pretty_print = pretty_print
        self.include_metadata = include_metadata
        self.results_data = []
        self.metadata = {}
        
        # 出力ディレクトリの作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_output(self, properties: Dict[str, Any]) -> None:
        """
        出力を開始します。
        
        Args:
            properties: 入力のプロパティ情報
        """
        try:
            # メタデータとして入力プロパティを保存
            if self.include_metadata:
                self.metadata = {
                    "input_properties": properties,
                    "processing_info": {
                        "timestamp": None,  # 後で設定
                        "total_items": 0,
                        "processed_items": 0
                    }
                }
            
            # 結果データを初期化
            self.results_data = []
            
            logger.info(f"Started JSON output: {self.output_path}")
        
        except Exception as e:
            logger.error(f"Error starting JSON output: {e}")
            raise
    
    def handle_result(self, item_id: ItemId, 
                     original_item: Any, 
                     results: Dict[str, Any]) -> None:
        """
        処理結果をJSON形式でバッファに保存します。
        
        Args:
            item_id: アイテムID
            original_item: 元のアイテム（この実装では使用しない）
            results: 各ワーカーからの処理結果
        """
        try:
            # 結果をシリアライズ可能な形式に変換
            serializable_results = self._serialize_results(results)
            
            # アイテムデータを作成
            item_data = {
                "item_id": item_id,
                "results": serializable_results,
                "timestamp": None  # 必要に応じて設定
            }
            
            # 結果データに追加
            self.results_data.append(item_data)
            
            # 処理済みアイテム数を更新
            if self.include_metadata:
                self.metadata["processing_info"]["processed_items"] = len(self.results_data)
            
            if len(self.results_data) % 100 == 0:
                logger.debug(f"Collected {len(self.results_data)} results")
        
        except Exception as e:
            logger.error(f"Error handling result for item {item_id}: {e}")
    
    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        処理結果をJSONシリアライズ可能な形式に変換します。
        
        Args:
            results: 処理結果の辞書
        
        Returns:
            シリアライズ可能な結果辞書
        """
        serializable_results = {}
        
        for worker_name, result in results.items():
            try:
                # カスタムオブジェクトをシリアライズ
                serializable_results[worker_name] = self._serialize_object(result)
            
            except Exception as e:
                logger.warning(f"Failed to serialize result from {worker_name}: {e}")
                serializable_results[worker_name] = {
                    "error": f"Serialization failed: {str(e)}",
                    "type": str(type(result))
                }
        
        return serializable_results
    
    def _serialize_object(self, obj: Any) -> Any:
        """
        オブジェクトをJSONシリアライズ可能な形式に変換します。
        
        Args:
            obj: シリアライズ対象のオブジェクト
        
        Returns:
            シリアライズ可能な値
        """
        # 基本型はそのまま返す
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # リストの場合は各要素を再帰的に処理
        if isinstance(obj, list):
            return [self._serialize_object(item) for item in obj]
        
        # 辞書の場合は各値を再帰的に処理
        if isinstance(obj, dict):
            return {key: self._serialize_object(value) for key, value in obj.items()}
        
        # NumPy配列の場合
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        
        # カスタムオブジェクトの場合、属性を辞書に変換
        if hasattr(obj, '__dict__'):
            obj_dict = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # プライベート属性は除外
                    try:
                        obj_dict[key] = self._serialize_object(value)
                    except Exception:
                        obj_dict[key] = str(value)  # 文字列として保存
            return obj_dict
        
        # その他の場合は文字列に変換
        return str(obj)
    
    def finish_output(self) -> None:
        """出力を終了し、JSONファイルに保存します。"""
        try:
            import datetime
            
            # メタデータの最終更新
            if self.include_metadata:
                self.metadata["processing_info"]["timestamp"] = datetime.datetime.now().isoformat()
                self.metadata["processing_info"]["total_items"] = len(self.results_data)
            
            # 出力データの構築
            output_data = {
                "results": self.results_data
            }
            
            # メタデータを含める場合
            if self.include_metadata:
                output_data["metadata"] = self.metadata
            
            # JSONファイルに保存
            with open(self.output_path, 'w', encoding='utf-8') as f:
                if self.pretty_print:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(output_data, f, ensure_ascii=False)
            
            logger.info(f"Finished JSON output: {self.output_path} "
                       f"({len(self.results_data)} items)")
        
        except Exception as e:
            logger.error(f"Error finishing JSON output: {e}")
            raise
    
    def get_results_data(self) -> List[Dict[str, Any]]:
        """
        現在の結果データを返します。
        
        Returns:
            結果データのリスト
        """
        return self.results_data.copy()
    
    def clear_results(self) -> None:
        """結果データをクリアします。"""
        self.results_data.clear()
        if self.include_metadata:
            self.metadata["processing_info"]["processed_items"] = 0
    
    def close(self) -> None:
        """リソースを解放します。"""
        # この実装では特に解放するリソースはありません
        pass 