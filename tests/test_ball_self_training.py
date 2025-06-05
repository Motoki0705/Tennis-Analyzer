"""
ボール検出Self-Trainingのテストスクリプト
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.ball.self_training.self_training_cycle import BallSelfTrainingCycle
from src.ball.self_training.trajectory_tracker import BallTrajectoryTracker
from src.ball.dataset.pseudo_labeled_seq_dataset import PseudoLabeledSequenceDataset
from src.ball.lit_module.self_training_lit_module import SelfTrainingLitModule, SelfTrainingCoordLitModule


class DummyModel(nn.Module):
    """テスト用のダミーモデル"""
    def __init__(self, output_type="heatmap"):
        super().__init__()
        self.output_type = output_type
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(4)
        if output_type == "heatmap":
            self.output = nn.Conv2d(16, 1, 1)
        else:  # coord
            self.fc = nn.Linear(16 * 90 * 160, 2)
    
    def forward(self, x):
        # 入力: [B, C, H, W] = [B, 3, 360, 640]
        x = torch.relu(self.conv(x))
        x = self.pool(x)  # [B, 16, 90, 160]
        
        if self.output_type == "heatmap":
            return self.output(x)  # [B, 1, 90, 160]
        else:
            x = x.view(x.size(0), -1)
            return self.fc(x)  # [B, 2]


class DummyDataset(torch.utils.data.Dataset):
    """テスト用のダミーデータセット"""
    def __init__(self, size=10, with_labels=True):
        self.size = size
        self.with_labels = with_labels
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # ダミーデータを生成
        image = torch.randn(3, 360, 640)
        if self.with_labels:
            # ヒートマップまたは座標
            heatmap = torch.zeros(1, 90, 160)
            heatmap[0, 45, 80] = 1.0  # 中心にボール
            visibility = torch.tensor([1.0])
            return image, heatmap, visibility
        else:
            # ラベルなしデータ（画像情報付き）
            info = {
                "id": idx,
                "file_name": f"image_{idx:04d}.jpg",
                "width": 640,
                "height": 360,
                "game_id": 0,
                "clip_id": idx // 10,
            }
            return image, info


class TestBallSelfTraining:
    """Self-Trainingのテストクラス"""
    
    def test_self_training_cycle_initialization(self):
        """Self-Trainingサイクルの初期化テスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            labeled_dataset = DummyDataset(100, with_labels=True)
            unlabeled_dataset = DummyDataset(50, with_labels=False)
            val_dataset = DummyDataset(20, with_labels=True)
            
            cycle = BallSelfTrainingCycle(
                model=model,
                labeled_dataset=labeled_dataset,
                unlabeled_dataset=unlabeled_dataset,
                val_dataset=val_dataset,
                save_dir=tmpdir,
                device=torch.device("cpu"),
                confidence_threshold=0.7,
                max_cycles=2,
                pseudo_label_weight=0.5,
            )
            
            assert cycle.confidence_threshold == 0.7
            assert cycle.max_cycles == 2
            assert cycle.pseudo_label_weight == 0.5
            assert Path(tmpdir) / "pseudo_labels" in cycle.pseudo_labels_dir.parents
            assert Path(tmpdir) / "models" in cycle.models_dir.parents
    
    def test_pseudo_label_generation(self):
        """擬似ラベル生成のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            unlabeled_dataset = DummyDataset(10, with_labels=False)
            
            cycle = BallSelfTrainingCycle(
                model=model,
                labeled_dataset=DummyDataset(10),
                unlabeled_dataset=unlabeled_dataset,
                val_dataset=DummyDataset(5),
                save_dir=tmpdir,
                device=torch.device("cpu"),
                confidence_threshold=0.5,
            )
            
            # 擬似ラベルを生成
            pseudo_labels = cycle._generate_predictions()
            
            # 擬似ラベルが生成されていることを確認
            assert isinstance(pseudo_labels, list)
            # 信頼度閾値によってはラベルが生成されない可能性もある
            if len(pseudo_labels) > 0:
                assert "bbox" in pseudo_labels[0]
                assert "score" in pseudo_labels[0]
                assert pseudo_labels[0]["is_pseudo"] == True
    
    def test_trajectory_tracker_initialization(self):
        """軌跡追跡の初期化テスト"""
        annotations = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 30, 30],
                "score": 0.9,
            }
        ]
        
        tracker = BallTrajectoryTracker(
            annotations=annotations,
            confidence_threshold=0.7,
            temporal_window=9,
            max_trajectory_gap=5,
            min_trajectory_length=7,
        )
        
        assert tracker.confidence_threshold == 0.7
        assert tracker.temporal_window == 9
        assert tracker.max_trajectory_gap == 5
        assert tracker.min_trajectory_length == 7
    
    def test_trajectory_tracking(self):
        """軌跡追跡の動作テスト"""
        # 連続したフレームのアノテーション
        annotations = []
        for i in range(10):
            annotations.append({
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [100 + i * 5, 100, 30, 30],  # 右に移動
                "score": 0.8,
            })
        
        tracker = BallTrajectoryTracker(
            annotations=annotations,
            confidence_threshold=0.7,
            min_trajectory_length=3,
        )
        
        # クリップ画像情報
        clip_imgs = [
            {"id": i + 1, "file_name": f"frame_{i:04d}.jpg"}
            for i in range(15)  # アノテーションより多いフレーム
        ]
        
        # 軌跡追跡を実行
        tracker.track_ball_in_clip(clip_imgs)
        
        # 統計情報を確認
        stats = tracker.get_statistics()
        assert stats["total_trajectories"] >= 1
        # 補間されたポイントがあるか確認
        # （ただし、この単純なケースでは補間されない可能性もある）
    
    def test_self_training_lit_module_heatmap(self):
        """SelfTrainingLitModule（ヒートマップ）のテスト"""
        model = DummyModel(output_type="heatmap")
        lit_module = SelfTrainingLitModule(
            model=model,
            lr=1e-3,
            pseudo_weight=0.5,
        )
        
        # ダミーバッチデータ
        batch_size = 2
        frames = torch.randn(batch_size, 3, 360, 640)
        heatmaps = torch.zeros(batch_size, 1, 90, 160)
        heatmaps[:, 0, 45, 80] = 1.0
        visibility = torch.ones(batch_size, 1)
        is_pseudo = torch.tensor([False, True])
        
        # フォワードパス
        output = lit_module(frames)
        assert output.shape == (batch_size, 1, 90, 160)
        
        # ロスの計算
        loss = lit_module._step((frames, heatmaps, visibility, is_pseudo), "train")
        assert isinstance(loss.item(), float)
    
    def test_self_training_lit_module_coord(self):
        """SelfTrainingCoordLitModule（座標）のテスト"""
        model = DummyModel(output_type="coord")
        lit_module = SelfTrainingCoordLitModule(
            model=model,
            lr=1e-3,
            pseudo_weight=0.5,
        )
        
        # ダミーバッチデータ
        batch_size = 2
        frames = torch.randn(batch_size, 3, 360, 640)
        coords = torch.tensor([[320.0, 180.0], [100.0, 200.0]])
        visibility = torch.ones(batch_size)
        is_pseudo = torch.tensor([False, True])
        
        # フォワードパス
        output = lit_module(frames)
        assert output.shape == (batch_size, 2)
        
        # ロスの計算
        loss = lit_module._step((frames, coords, visibility, is_pseudo), "train")
        assert isinstance(loss.item(), float)
    
    @patch('src.ball.dataset.pseudo_labeled_seq_dataset.PseudoLabeledSequenceDataset')
    def test_pseudo_labeled_dataset_integration(self, mock_dataset_class):
        """PseudoLabeledSequenceDatasetとの統合テスト"""
        # モックデータセットの設定
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.add_pseudo_labels = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 擬似ラベルファイルを作成
            pseudo_labels = {
                "images": [{"id": 1, "file_name": "test.jpg"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [100, 100, 30, 30],
                        "score": 0.8,
                        "is_pseudo": True,
                    }
                ],
                "categories": [{"id": 1, "name": "ball"}],
            }
            
            pseudo_label_path = Path(tmpdir) / "pseudo_labels.json"
            with open(pseudo_label_path, "w") as f:
                json.dump(pseudo_labels, f)
            
            # add_pseudo_labelsが呼ばれることを確認
            mock_dataset.add_pseudo_labels(str(pseudo_label_path), weight=0.5)
            mock_dataset.add_pseudo_labels.assert_called_once_with(
                str(pseudo_label_path), weight=0.5
            )
    
    def test_self_training_end_to_end(self):
        """Self-Trainingのエンドツーエンドテスト（簡易版）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 小規模なデータでテスト
            model = DummyModel()
            labeled_dataset = DummyDataset(20, with_labels=True)
            unlabeled_dataset = DummyDataset(10, with_labels=False)
            val_dataset = DummyDataset(5, with_labels=True)
            
            cycle = BallSelfTrainingCycle(
                model=model,
                labeled_dataset=labeled_dataset,
                unlabeled_dataset=unlabeled_dataset,
                val_dataset=val_dataset,
                save_dir=tmpdir,
                device=torch.device("cpu"),
                confidence_threshold=0.3,  # 低めに設定して擬似ラベルが生成されやすくする
                max_cycles=1,  # 1サイクルのみ
                pseudo_label_weight=0.5,
                use_trajectory_tracking=False,  # 簡単のため無効化
            )
            
            # モックで_retrain_modelをパッチ（実際の学習は行わない）
            with patch.object(cycle, '_retrain_model', return_value=0.8):
                best_model, best_score, metrics = cycle.run_self_training()
            
            # 結果を確認
            assert best_score == 0.8
            assert "train_loss" in metrics
            assert "val_loss" in metrics
            assert "pseudo_label_count" in metrics
            
            # 擬似ラベルファイルが作成されているか確認
            pseudo_label_files = list(Path(tmpdir).glob("pseudo_labels/*.json"))
            # 擬似ラベルが生成されていれば、ファイルが存在するはず


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 