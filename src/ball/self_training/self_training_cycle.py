import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ball.self_training.trajectory_tracker import BallTrajectoryTracker


class BallSelfTrainingCycle:
    """
    ボール検出のための自己学習サイクルを実装するクラス。
    1. 現在のモデルで未ラベルデータに予測を行う
    2. 高信頼度の予測から擬似ラベルを生成
    3. 軌跡追跡による擬似ラベルの洗練
    4. 擬似ラベルとオリジナルラベルを組み合わせて再学習
    5. 新しいモデルで1に戻る
    """

    def __init__(
        self,
        model: nn.Module,
        labeled_dataset,
        unlabeled_dataset,
        val_dataset,
        save_dir: Union[str, Path],
        device: torch.device,
        confidence_threshold: float = 0.7,
        max_cycles: int = 3,
        pseudo_label_weight: float = 0.5,
    ):
        """
        初期化

        Parameters
        ----------
        model : 学習済みのボール検出モデル
        labeled_dataset : ラベル付きデータセット
        unlabeled_dataset : ラベルなしデータセット
        val_dataset : 検証用データセット
        save_dir : モデルと擬似ラベルを保存するディレクトリ
        device : 計算デバイス
        confidence_threshold : 擬似ラベルとして採用する信頼度の閾値
        max_cycles : 最大自己学習サイクル数
        pseudo_label_weight : 擬似ラベルの重み付け係数
        """
        self.model = model
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_cycles = max_cycles
        self.pseudo_label_weight = pseudo_label_weight

        # 保存ディレクトリを作成
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pseudo_labels_dir = self.save_dir / "pseudo_labels"
        self.pseudo_labels_dir.mkdir(exist_ok=True)
        self.models_dir = self.save_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # 現在のサイクル
        self.current_cycle = 0
        self.best_val_score = 0.0
        self.best_model_path = None

    def run_self_training(self):
        """
        自己学習サイクルを実行する
        """
        print(f"Starting self-training cycle with {self.max_cycles} iterations")
        
        while self.current_cycle < self.max_cycles:
            print(f"\n--- Cycle {self.current_cycle + 1}/{self.max_cycles} ---")
            
            # 1. 未ラベルデータに予測を行う
            pseudo_labels = self._generate_predictions()
            
            # 2. 軌跡追跡で擬似ラベルを洗練
            refined_pseudo_labels = self._refine_pseudo_labels(pseudo_labels)
            
            # 3. 擬似ラベルを保存
            pseudo_label_path = self._save_pseudo_labels(refined_pseudo_labels)
            
            # 4. 擬似ラベルとオリジナルラベルを組み合わせて再学習
            combined_dataset = self._combine_datasets(pseudo_label_path)
            
            # 5. モデルの再学習
            val_score = self._retrain_model(combined_dataset)
            
            # 6. 最良モデルの保存
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_model_path = self.models_dir / f"model_cycle_{self.current_cycle + 1}.pt"
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved: {self.best_model_path} with score {val_score:.4f}")
            
            self.current_cycle += 1
        
        # 最終結果
        print("\n--- Self-Training Complete ---")
        print(f"Best validation score: {self.best_val_score:.4f}")
        print(f"Best model path: {self.best_model_path}")
        
        # 最良モデルを読み込む
        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path))
        
        return self.model, self.best_val_score

    def _generate_predictions(self) -> List[Dict]:
        """
        現在のモデルで未ラベルデータに予測を行う

        Returns
        -------
        pseudo_labels : 擬似ラベルのリスト
        """
        print("Generating predictions on unlabeled data...")
        self.model.eval()
        
        dataloader = DataLoader(
            self.unlabeled_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        pseudo_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                images, image_infos = batch
                images = images.to(self.device)
                
                # モデルの出力形式に応じて予測を取得
                outputs = self.model(images)
                
                # 出力形式に応じて擬似ラベルを抽出
                # ここではモデルがヒートマップを出力すると仮定
                batch_pseudo_labels = self._extract_pseudo_labels_from_outputs(outputs, image_infos)
                pseudo_labels.extend(batch_pseudo_labels)
        
        return pseudo_labels

    def _extract_pseudo_labels_from_outputs(self, outputs, image_infos) -> List[Dict]:
        """
        モデル出力から擬似ラベルを抽出する

        Parameters
        ----------
        outputs : モデルの出力
        image_infos : 画像情報

        Returns
        -------
        pseudo_labels : 擬似ラベルのリスト
        """
        # モデルの出力形式に応じて適切に実装する必要があります
        # 以下は一例で、ヒートマップ出力を仮定しています
        batch_size = outputs.size(0)
        pseudo_labels = []
        
        for i in range(batch_size):
            heatmap = outputs[i].cpu().numpy()
            info = image_infos[i]
            
            # ヒートマップから最大値の位置とスコアを取得
            score = float(heatmap.max())
            if score >= self.confidence_threshold:
                # 最大値の位置（y, x）を取得
                y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                
                # 画像サイズに合わせてスケーリング
                orig_h, orig_w = info.get("height", heatmap.shape[0]), info.get("width", heatmap.shape[1])
                scale_y = orig_h / heatmap.shape[0]
                scale_x = orig_w / heatmap.shape[1]
                
                center_x = x * scale_x
                center_y = y * scale_y
                
                # ボールのサイズを推定（ここでは固定値を使用）
                ball_w, ball_h = 30, 30  # 実際の用途に合わせて調整
                
                # 擬似ラベル（COCO形式）
                pseudo_label = {
                    "image_id": info.get("id", 0),
                    "category_id": 1,  # ボールカテゴリ
                    "bbox": [center_x - ball_w/2, center_y - ball_h/2, ball_w, ball_h],
                    "score": score,
                    "is_pseudo": True
                }
                
                pseudo_labels.append(pseudo_label)
        
        return pseudo_labels

    def _refine_pseudo_labels(self, pseudo_labels: List[Dict]) -> List[Dict]:
        """
        軌跡追跡で擬似ラベルを洗練する

        Parameters
        ----------
        pseudo_labels : 擬似ラベルのリスト

        Returns
        -------
        refined_pseudo_labels : 洗練された擬似ラベルのリスト
        """
        print("Refining pseudo labels with trajectory tracking...")
        
        # 画像情報を取得
        image_infos = []
        for item in self.unlabeled_dataset:
            if isinstance(item, tuple) and len(item) >= 2:
                _, info = item
                image_infos.append(info)
        
        # 一時的なCOCOフォーマットのアノテーションを作成
        temp_coco = {
            "images": image_infos,
            "annotations": pseudo_labels,
            "categories": [{"id": 1, "name": "ball"}]
        }
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(temp_coco, temp_file)
        
        # 軌跡追跡を実行
        output_path = Path(temp_path).with_suffix(".refined.json")
        tracker = BallTrajectoryTracker(
            annotations=pseudo_labels,
            confidence_threshold=self.confidence_threshold,
            temporal_window=9,
            max_trajectory_gap=5,
            min_trajectory_length=7,
        )
        
        # 画像をゲームIDとクリップIDでグループ化
        clips = {}
        for img in image_infos:
            key = (img.get("game_id", 0), img.get("clip_id", 0))
            if key not in clips:
                clips[key] = []
            clips[key].append(img)
        
        # 各クリップで軌跡追跡を実行
        for clip_imgs in clips.values():
            tracker.track_ball_in_clip(clip_imgs)
        
        # 一時ファイルを削除
        os.unlink(temp_path)
        
        return pseudo_labels  # tracker によって更新されたラベル

    def _save_pseudo_labels(self, pseudo_labels: List[Dict]) -> Path:
        """
        擬似ラベルをファイルに保存する

        Parameters
        ----------
        pseudo_labels : 擬似ラベルのリスト

        Returns
        -------
        output_path : 保存したファイルのパス
        """
        output_path = self.pseudo_labels_dir / f"pseudo_labels_cycle_{self.current_cycle + 1}.json"
        
        # 画像情報を取得
        image_infos = []
        for item in self.unlabeled_dataset:
            if isinstance(item, tuple) and len(item) >= 2:
                _, info = item
                image_infos.append(info)
        
        # COCOフォーマットで保存
        coco_data = {
            "images": image_infos,
            "annotations": pseudo_labels,
            "categories": [{"id": 1, "name": "ball"}]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved pseudo labels to {output_path}")
        return output_path

    def _combine_datasets(self, pseudo_label_path: Path):
        """
        擬似ラベルとオリジナルラベルを組み合わせたデータセットを作成

        Parameters
        ----------
        pseudo_label_path : 擬似ラベルのファイルパス

        Returns
        -------
        combined_dataset : 組み合わせたデータセット
        """
        print("Combining labeled and pseudo-labeled datasets...")
        
        # ここでは、データセットクラスが擬似ラベルを読み込めるメソッドがあると仮定
        # 実際の実装ではデータセットクラスに応じて適切に実装する必要があります
        if hasattr(self.labeled_dataset, "add_pseudo_labels"):
            combined_dataset = copy.deepcopy(self.labeled_dataset)
            combined_dataset.add_pseudo_labels(pseudo_label_path, weight=self.pseudo_label_weight)
        else:
            # 擬似ラベル追加メソッドがない場合はオリジナルデータセットをそのまま返す
            print("Warning: Dataset does not support adding pseudo labels. Using original dataset.")
            combined_dataset = self.labeled_dataset
        
        return combined_dataset

    def _retrain_model(self, combined_dataset):
        """
        モデルを再学習する

        Parameters
        ----------
        combined_dataset : 組み合わせたデータセット

        Returns
        -------
        val_score : 検証スコア
        """
        print("Retraining model with combined dataset...")
        
        # トレーニング用データローダー
        train_loader = DataLoader(
            combined_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 検証用データローダー
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # オプティマイザとスケジューラの設定
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        
        # 損失関数（タスクに応じて適切に選択）
        criterion = nn.MSELoss()
        
        # トレーニングループ
        num_epochs = 10  # サイクルごとのエポック数
        best_val_score = 0.0
        
        for epoch in range(num_epochs):
            # トレーニングフェーズ
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, targets = batch[:2]  # データセットの形式に応じて調整
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # 擬似ラベルのサンプルには重みを適用
                is_pseudo = batch[2] if len(batch) > 2 else torch.zeros(images.size(0), dtype=torch.bool)
                sample_weights = torch.ones(images.size(0), device=self.device)
                sample_weights[is_pseudo] = self.pseudo_label_weight
                
                # 重み付き損失計算
                loss = criterion(outputs, targets)
                weighted_loss = (loss * sample_weights.view(-1, 1, 1, 1)).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                train_loss += weighted_loss.item()
            
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            # 検証フェーズ
            val_score = self._evaluate(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Score: {val_score:.4f}")
            
            # スケジューラ更新
            scheduler.step(val_score)
            
            # 最良モデルの保存（サイクル内）
            if val_score > best_val_score:
                best_val_score = val_score
                cycle_best_path = self.models_dir / f"model_cycle_{self.current_cycle + 1}_epoch_{epoch + 1}.pt"
                torch.save(self.model.state_dict(), cycle_best_path)
                print(f"New best model saved: {cycle_best_path}")
        
        return best_val_score

    def _evaluate(self, val_loader) -> float:
        """
        モデルを評価する

        Parameters
        ----------
        val_loader : 検証用データローダー

        Returns
        -------
        score : 評価スコア
        """
        self.model.eval()
        val_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images, targets = batch[:2]
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # ここでは単純に損失の逆数をスコアとしていますが、
        # 実際のタスクに応じた評価指標（AP、IoUなど）を使用すべきです
        score = 1.0 / (val_loss + 1e-8)
        
        return score


# 使用例
if __name__ == "__main__":
    import numpy as np
    
    # 仮のデータセットとモデルを作成
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
            self.data = [(torch.randn(3, 224, 224), torch.randn(1, 56, 56), {"id": i}) for i in range(size)]
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return self.data[idx]
            
        def add_pseudo_labels(self, path, weight=0.5):
            print(f"Added pseudo labels from {path} with weight {weight}")
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
            self.pool = nn.MaxPool2d(4)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = self.conv3(x)
            return x
    
    # ダミーデータとモデルでテスト
    model = DummyModel()
    labeled_dataset = DummyDataset(100)
    unlabeled_dataset = DummyDataset(50)
    val_dataset = DummyDataset(20)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self_training = BallSelfTrainingCycle(
        model=model.to(device),
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        val_dataset=val_dataset,
        save_dir="outputs/ball/self_training",
        device=device,
        max_cycles=2,
    )
    
    best_model, score = self_training.run_self_training()
    print(f"Final score: {score:.4f}") 