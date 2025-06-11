"""
拡張可能なキューシステムの高度な使用例デモ

このデモは以下の拡張性を実証します：
1. カスタムキュー設定
2. 動的ワーカー追加
3. 特殊用途のキュー構成
4. パフォーマンス監視機能
"""
import queue
import threading
import time
from typing import Dict, Any, Optional

from src.multi.streaming_overlayer.queue_manager import (
    QueueManager, 
    QueueConfig, 
    create_queue_manager_for_video_predictor
)


def demo_basic_queue_configuration():
    """基本的なキュー設定のデモ"""
    print("=" * 60)
    print("1. 基本的なキュー設定デモ")
    print("=" * 60)
    
    # 標準構成でのQueueManager作成
    worker_names = ["ball", "court", "pose"]
    queue_manager = create_queue_manager_for_video_predictor(worker_names)
    
    print(f"✅ 基本ワーカー数: {len(queue_manager.worker_queue_sets)}")
    
    # 各ワーカーのキュー構成を表示
    for worker_name in worker_names:
        queue_set = queue_manager.get_worker_queue_set(worker_name)
        base_queues = len(queue_set.base_queues)
        extended_queues = len(queue_set.extended_queues)
        print(f"  {worker_name}: 基本キュー {base_queues}個, 拡張キュー {extended_queues}個")
    
    print()


def demo_custom_queue_configuration():
    """カスタムキュー設定のデモ"""
    print("=" * 60)
    print("2. カスタムキュー設定デモ")
    print("=" * 60)
    
    # カスタム設定の定義
    custom_configs = {
        "high_priority_inference": {
            "maxsize": 128,
            "queue_type": "PriorityQueue",
            "description": "高優先度推論専用キュー"
        },
        "batch_processing": {
            "maxsize": 64,
            "queue_type": "Queue",
            "description": "バッチ処理用大容量キュー"
        },
        "emergency_queue": {
            "maxsize": 8,
            "queue_type": "LifoQueue",
            "description": "緊急処理用LIFO キュー"
        }
    }
    
    # カスタム設定でQueueManager作成
    worker_names = ["ball"]
    queue_manager = create_queue_manager_for_video_predictor(worker_names, custom_configs)
    
    # カスタム設定の確認
    for config_name, config_data in custom_configs.items():
        if config_name in queue_manager.queue_configs:
            config = queue_manager.queue_configs[config_name]
            print(f"✅ {config_name}: maxsize={config.maxsize}, type={config.queue_type}")
            print(f"   説明: {config.description}")
        else:
            print(f"❌ {config_name}: 設定が見つかりません")
    
    print()


def demo_dynamic_worker_addition():
    """動的ワーカー追加のデモ"""
    print("=" * 60)
    print("3. 動的ワーカー追加デモ")
    print("=" * 60)
    
    # 初期構成
    queue_manager = QueueManager()
    queue_manager.initialize_results_queue()
    
    # 段階的にワーカーを追加
    worker_configs = [
        ("basic_worker", None),
        ("detection_worker", ["detection_inference", "detection_postprocess"]),
        ("pose_worker", ["detection_inference", "detection_postprocess", "pose_inference", "pose_postprocess"]),
        ("multi_stage_worker", ["detection_inference", "pose_inference", "ball_inference", "court_inference"])
    ]
    
    for worker_name, extended_queues in worker_configs:
        try:
            queue_manager.initialize_worker_queues(worker_name, extended_queues)
            queue_set = queue_manager.get_worker_queue_set(worker_name)
            base_count = len(queue_set.base_queues)
            extended_count = len(queue_set.extended_queues)
            print(f"✅ {worker_name}: 基本 {base_count}個, 拡張 {extended_count}個")
        except Exception as e:
            print(f"❌ {worker_name}: エラー - {e}")
    
    print()


def demo_performance_monitoring():
    """パフォーマンス監視のデモ"""
    print("=" * 60)
    print("4. パフォーマンス監視デモ")
    print("=" * 60)
    
    # QueueManagerの作成
    worker_names = ["ball", "pose"]
    queue_manager = create_queue_manager_for_video_predictor(worker_names)
    
    # キューにテストデータを追加
    def populate_queues():
        """キューにテストデータを格納"""
        for i in range(10):
            # Ball worker キュー
            ball_preprocess_q = queue_manager.get_queue("ball", "preprocess")
            ball_preprocess_q.put(f"ball_task_{i}")
            
            # Pose worker 拡張キュー
            pose_detection_q = queue_manager.get_queue("pose", "detection_inference")
            pose_pose_q = queue_manager.get_queue("pose", "pose_inference")
            pose_detection_q.put(f"detection_task_{i}")
            pose_pose_q.put(f"pose_task_{i}")
            
            # Results queue
            results_q = queue_manager.get_results_queue()
            results_q.put((i, "test", f"result_{i}"))
    
    # キューにデータを格納
    populate_queues()
    
    # 状態監視
    status = queue_manager.get_queue_status()
    print(f"結果キューサイズ: {status['results_queue_size']}")
    print()
    
    for worker_name, worker_status in status['workers'].items():
        print(f"[{worker_name.upper()}]")
        print("  基本キュー:")
        for queue_name, size in worker_status['base_queues'].items():
            print(f"    {queue_name}: {size} items")
        
        if worker_status['extended_queues']:
            print("  拡張キュー:")
            for queue_name, size in worker_status['extended_queues'].items():
                print(f"    {queue_name}: {size} items")
        print()


def demo_queue_type_variations():
    """異なるキュータイプのデモ"""
    print("=" * 60)
    print("5. キュータイプバリエーションデモ")
    print("=" * 60)
    
    queue_manager = QueueManager()
    
    # 異なるタイプのキューを作成
    queue_types = [
        ("standard_queue", "Queue"),
        ("priority_queue", "results"),  # PriorityQueue
        ("lifo_queue", "emergency_queue")  # LifoQueue (custom config needed)
    ]
    
    # LIFO用のカスタム設定を追加
    lifo_config = QueueConfig("emergency_queue", 16, "LifoQueue", "緊急用LIFOキュー")
    queue_manager.add_queue_config(lifo_config)
    
    for queue_name, config_name in queue_types:
        try:
            q = queue_manager.create_queue_from_config(config_name)
            queue_type = type(q).__name__
            print(f"✅ {queue_name}: {queue_type}")
            
            # 各キューの動作テスト
            if queue_type == "PriorityQueue":
                # 優先度付きアイテムを追加 (priority, item)
                q.put((1, "high_priority"))
                q.put((3, "low_priority"))
                q.put((2, "medium_priority"))
                
                print("   優先度順取得テスト:")
                while not q.empty():
                    priority, item = q.get()
                    print(f"     Priority {priority}: {item}")
            
            elif queue_type == "LifoQueue":
                # LIFO順でアイテムを追加
                q.put("first")
                q.put("second")
                q.put("third")
                
                print("   LIFO順取得テスト:")
                while not q.empty():
                    item = q.get()
                    print(f"     {item}")
            
            else:  # Standard Queue
                # FIFO順でアイテムを追加
                q.put("first")
                q.put("second")
                q.put("third")
                
                print("   FIFO順取得テスト:")
                while not q.empty():
                    item = q.get()
                    print(f"     {item}")
            
        except Exception as e:
            print(f"❌ {queue_name}: エラー - {e}")
        
        print()


def demo_scalability_test():
    """スケーラビリティテストのデモ"""
    print("=" * 60)
    print("6. スケーラビリティテストデモ")
    print("=" * 60)
    
    # 大量のワーカーを持つ構成をテスト
    worker_names = [f"worker_{i}" for i in range(10)]
    
    start_time = time.time()
    queue_manager = create_queue_manager_for_video_predictor(worker_names)
    init_time = time.time() - start_time
    
    print(f"✅ {len(worker_names)}個のワーカー初期化時間: {init_time:.4f}秒")
    
    # 状態監視のパフォーマンステスト
    start_time = time.time()
    status = queue_manager.get_queue_status()
    status_time = time.time() - start_time
    
    print(f"✅ 状態監視実行時間: {status_time:.4f}秒")
    print(f"✅ 管理中のワーカー数: {len(status['workers'])}")
    
    # メモリ使用量の概算
    total_queues = 0
    for worker_status in status['workers'].values():
        total_queues += len(worker_status['base_queues'])
        total_queues += len(worker_status['extended_queues'])
    
    print(f"✅ 総キュー数: {total_queues}")
    print()


def main():
    """メインデモンストレーション実行"""
    print("🚀 拡張可能キューシステム - 高度な使用例デモ")
    print("このデモはvideo_predictor側でのキュー初期化と将来の拡張性を実証します")
    print()
    
    # 各デモを実行
    demo_basic_queue_configuration()
    demo_custom_queue_configuration()
    demo_dynamic_worker_addition()
    demo_performance_monitoring()
    demo_queue_type_variations()
    demo_scalability_test()
    
    print("=" * 60)
    print("🎉 デモ完了 - 拡張可能なキューシステムの柔軟性を確認！")
    print("=" * 60)
    
    print("\n📝 主な拡張性ポイント:")
    print("  ✅ video_predictor側でのキュー管理")
    print("  ✅ predictor数に制限されない柔軟なキュー構成")
    print("  ✅ カスタムキュー設定による特殊用途対応")
    print("  ✅ 動的ワーカー追加機能")
    print("  ✅ 複数のキュータイプサポート")
    print("  ✅ リアルタイム状態監視")
    print("  ✅ スケーラブルな設計")


if __name__ == "__main__":
    main() 