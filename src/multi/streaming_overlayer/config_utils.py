"""
キューシステム設定ユーティリティ
"""
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig, OmegaConf
import logging

from .queue_manager import QueueConfig

logger = logging.getLogger(__name__)


def create_queue_configs_from_hydra_config(queue_config: DictConfig) -> Dict[str, Dict[str, Any]]:
    """
    Hydra設定からキューコンフィグを作成
    
    Args:
        queue_config: Hydra queue設定
        
    Returns:
        QueueManagerで使用可能なキューコンフィグ辞書
    """
    queue_configs = {}
    
    # 基本キューの設定
    if hasattr(queue_config, 'base_queue_sizes'):
        base_sizes = OmegaConf.to_container(queue_config.base_queue_sizes)
        queue_types = OmegaConf.to_container(queue_config.queue_types) if hasattr(queue_config, 'queue_types') else {}
        
        for queue_name, size in base_sizes.items():
            queue_configs[queue_name] = {
                "maxsize": size,
                "queue_type": queue_types.get(queue_name, "Queue"),
                "description": f"基本{queue_name}キュー"
            }
    
    # 拡張キューの設定
    if hasattr(queue_config, 'worker_extended_queues'):
        extended_queues = OmegaConf.to_container(queue_config.worker_extended_queues)
        queue_types = OmegaConf.to_container(queue_config.queue_types) if hasattr(queue_config, 'queue_types') else {}
        
        for worker_name, worker_queues in extended_queues.items():
            for queue_name, size in worker_queues.items():
                queue_configs[queue_name] = {
                    "maxsize": size,
                    "queue_type": queue_types.get(queue_name, "Queue"),
                    "description": f"{worker_name}ワーカー専用{queue_name}キュー"
                }
    
    # カスタムキューの設定
    if hasattr(queue_config, 'custom_queues'):
        custom_queues = OmegaConf.to_container(queue_config.custom_queues)
        
        for queue_name, config_data in custom_queues.items():
            queue_configs[queue_name] = {
                "maxsize": config_data.get("maxsize", 16),
                "queue_type": config_data.get("queue_type", "Queue"),
                "description": config_data.get("description", f"カスタム{queue_name}キュー")
            }
    
    return queue_configs


def get_worker_extended_queue_names(queue_config: DictConfig, worker_name: str) -> Optional[List[str]]:
    """
    指定されたワーカーの拡張キュー名リストを取得
    
    Args:
        queue_config: Hydra queue設定
        worker_name: ワーカー名
        
    Returns:
        拡張キュー名のリスト、または None
    """
    if not hasattr(queue_config, 'worker_extended_queues'):
        return None
    
    extended_queues = OmegaConf.to_container(queue_config.worker_extended_queues)
    worker_queues = extended_queues.get(worker_name, {})
    
    return list(worker_queues.keys()) if worker_queues else []


def apply_performance_settings(queue_config: DictConfig) -> Dict[str, Any]:
    """
    パフォーマンス設定を抽出
    
    Args:
        queue_config: Hydra queue設定
        
    Returns:
        パフォーマンス設定辞書
    """
    performance_settings = {}
    
    if hasattr(queue_config, 'performance'):
        performance = OmegaConf.to_container(queue_config.performance)
        performance_settings.update(performance)
    
    return performance_settings


def validate_queue_config(queue_config: DictConfig) -> bool:
    """
    キュー設定の妥当性を検証
    
    Args:
        queue_config: Hydra queue設定
        
    Returns:
        設定が妥当かどうか
    """
    try:
        # 必須設定の確認
        if not hasattr(queue_config, 'base_queue_sizes'):
            logger.warning("base_queue_sizes設定が見つかりません")
            return False
        
        base_sizes = OmegaConf.to_container(queue_config.base_queue_sizes)
        required_queues = ['preprocess', 'inference', 'postprocess', 'results']
        
        for queue_name in required_queues:
            if queue_name not in base_sizes:
                logger.error(f"必須キュー '{queue_name}' の設定が見つかりません")
                return False
            
            if not isinstance(base_sizes[queue_name], int) or base_sizes[queue_name] <= 0:
                logger.error(f"キュー '{queue_name}' のサイズが無効です: {base_sizes[queue_name]}")
                return False
        
        # キュータイプの確認
        if hasattr(queue_config, 'queue_types'):
            queue_types = OmegaConf.to_container(queue_config.queue_types)
            valid_types = ['Queue', 'PriorityQueue', 'LifoQueue']
            
            for queue_name, queue_type in queue_types.items():
                if queue_type not in valid_types:
                    logger.error(f"無効なキュータイプ '{queue_type}' がキュー '{queue_name}' に指定されています")
                    return False
        
        logger.info("キュー設定の検証に成功しました")
        return True
        
    except Exception as e:
        logger.error(f"キュー設定の検証中にエラーが発生しました: {e}")
        return False


def log_queue_configuration(queue_config: DictConfig):
    """
    キュー設定をログに出力
    
    Args:
        queue_config: Hydra queue設定
    """
    logger.info("=== キューシステム設定 ===")
    
    # 基本キューサイズ
    if hasattr(queue_config, 'base_queue_sizes'):
        logger.info("基本キューサイズ:")
        base_sizes = OmegaConf.to_container(queue_config.base_queue_sizes)
        for queue_name, size in base_sizes.items():
            logger.info(f"  {queue_name}: {size}")
    
    # 拡張キュー
    if hasattr(queue_config, 'worker_extended_queues'):
        logger.info("ワーカー拡張キュー:")
        extended_queues = OmegaConf.to_container(queue_config.worker_extended_queues)
        for worker_name, worker_queues in extended_queues.items():
            logger.info(f"  {worker_name}:")
            for queue_name, size in worker_queues.items():
                logger.info(f"    {queue_name}: {size}")
    
    # カスタムキュー
    if hasattr(queue_config, 'custom_queues'):
        logger.info("カスタムキュー:")
        custom_queues = OmegaConf.to_container(queue_config.custom_queues)
        for queue_name, config_data in custom_queues.items():
            logger.info(f"  {queue_name}: size={config_data['maxsize']}, type={config_data['queue_type']}")
    
    # パフォーマンス設定
    if hasattr(queue_config, 'performance'):
        logger.info("パフォーマンス設定:")
        performance = OmegaConf.to_container(queue_config.performance)
        for key, value in performance.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=========================") 