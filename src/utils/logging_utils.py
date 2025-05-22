import logging
from typing import Type


def setup_logger(cls: Type) -> logging.Logger:
    """
    クラス用のロガーを設定し、ハンドラーが存在しない場合は追加します。
    
    Args:
        cls: ロガーを設定するクラス
        
    Returns:
        設定されたロガーインスタンス
    """
    logger = logging.getLogger(cls.__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
