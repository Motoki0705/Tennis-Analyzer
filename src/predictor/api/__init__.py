"""
Predictor API Module

Provides command-line interfaces for tennis ball detection and video processing.
"""

# モジュールレベルのインポートを避けてRuntimeWarningを防止
def get_inference_main():
    """Inference main function getter to avoid import-time execution"""
    from .inference import main
    return main

def get_batch_main():
    """Batch process main function getter to avoid import-time execution"""
    from .batch_process import main
    return main

# 後方互換性のため
def inference_main(*args, **kwargs):
    return get_inference_main()(*args, **kwargs)

def batch_main(*args, **kwargs):
    return get_batch_main()(*args, **kwargs)

__all__ = [
    'inference_main',
    'batch_main',
    'get_inference_main',
    'get_batch_main',
] 