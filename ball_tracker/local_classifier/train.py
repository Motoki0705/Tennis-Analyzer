"""
Local Ball Classifier Training
ローカル分類器の学習スクリプト
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available. Installing required packages:")
    print("pip install scikit-learn seaborn")

from .model import create_local_classifier
from .dataset import create_dataloaders, BallPatchDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_local_classifier(
    annotation_file: str,
    images_dir: str,
    output_dir: str = "./local_classifier_checkpoints",
    model_type: str = "standard",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patch_size: int = 16,
    device: str = "cuda"
):
    """
    ローカル分類器の学習実行
    
    Args:
        annotation_file (str): COCO形式アノテーションファイル
        images_dir (str): 画像ディレクトリ
        output_dir (str): チェックポイント保存ディレクトリ
        model_type (str): モデルタイプ
        epochs (int): エポック数
        batch_size (int): バッチサイズ
        learning_rate (float): 学習率
        weight_decay (float): 重み減衰
        patch_size (int): パッチサイズ
        device (str): デバイス
    """
    
    # Setup directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Device setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_local_classifier(model_type, input_size=patch_size)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {model_type} with {total_params:,} parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        annotation_file=annotation_file,
        images_dir=images_dir,
        batch_size=batch_size,
        patch_size=patch_size
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    print("\n" + "="*80)
    print("🚀 ローカル分類器学習開始")
    print("="*80)
    print(f"📊 データセット: 学習 {len(train_loader.dataset)}, 検証 {len(val_loader.dataset)}")
    print(f"🏗️  モデル: {model_type} ({total_params:,} パラメータ)")
    print(f"⚙️  設定: エポック {epochs}, バッチサイズ {batch_size}, 学習率 {learning_rate}")
    print("="*80)
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation  
        val_loss, val_acc, val_f1, val_report = validate_epoch(model, val_loader, criterion, device, 
                                                               return_detailed=True)
        
        # Scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Validation', val_f1, epoch)
        writer.add_scalar('Learning_Rate', new_lr, epoch)
        
        # 詳細表示
        print(f"🏋️  学習  | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"✅ 検証  | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
        if old_lr != new_lr:
            print(f"📉 学習率変更: {old_lr:.6f} → {new_lr:.6f}")
        
        # Validation詳細結果
        if val_report:
            print(f"📊 詳細結果:")
            print(f"   - Positive (ボールあり): Precision {val_report['1']['precision']:.3f}, "
                  f"Recall {val_report['1']['recall']:.3f}")
            print(f"   - Negative (ボールなし): Precision {val_report['0']['precision']:.3f}, "
                  f"Recall {val_report['0']['recall']:.3f}")
        
        # Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, output_dir / "best_model.pth")
            print(f"🏆 新ベストモデル保存! 検証精度: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏰ 早期停止: {patience}エポック改善なし")
                break
        
        # Progress visualization
        if (epoch + 1) % 5 == 0:
            plot_training_progress(train_losses, val_losses, train_accs, val_accs, 
                                 output_dir / f"training_progress_epoch_{epoch+1}.png")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_acc, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"💾 チェックポイント保存: epoch_{epoch+1}")
    
    print(f"\n🎯 最終結果: 最高検証精度 {best_val_acc:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, epochs-1, val_accs[-1] if val_accs else 0, 
                   output_dir / "final_model.pth")
    
    # Final evaluation and visualization
    print("\n" + "="*80)
    print("📊 最終評価・可視化生成")
    print("="*80)
    
    # Plot final training curves
    plot_training_progress(train_losses, val_losses, train_accs, val_accs, 
                         output_dir / "final_training_curves.png")
    print(f"✅ 学習曲線保存: final_training_curves.png")
    
    # Plot confusion matrix on validation set
    plot_confusion_matrix(model, val_loader, device, 
                         output_dir / "confusion_matrix.png")
    print(f"✅ 混同行列保存: confusion_matrix.png")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'total_epochs': len(train_losses),
        'model_type': model_type,
        'total_parameters': total_params
    }
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    
    print(f"\n🎉 学習完了!")
    print(f"🏆 最高検証精度: {best_val_acc:.4f}")
    print(f"📈 最終検証精度: {val_accs[-1] if val_accs else 0:.4f}")
    print(f"📁 結果保存先: {output_dir}")
    print("="*80)
    
    return model, history


def train_epoch(model, dataloader, criterion, optimizer, device):
    """単一エポックの学習"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="🏋️  学習中", ncols=100)
    
    for batch_idx, (patches, labels) in enumerate(pbar):
        patches = patches.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(patches)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # For F1 calculation
        all_predictions.extend(predicted.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = correct / total
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}', 
            'Acc': f'{accuracy:.4f}',
            'Batch': f'{batch_idx+1}/{len(dataloader)}'
        })
    
    # Calculate F1 score
    if SKLEARN_AVAILABLE:
        f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    else:
        f1 = 0.0
    
    return total_loss / len(dataloader), correct / total, f1


def validate_epoch(model, dataloader, criterion, device, return_detailed=False):
    """単一エポックの検証"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="✅ 検証中", ncols=100)
        
        for patches, labels in pbar:
            patches = patches.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # For detailed metrics
            all_predictions.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Update progress bar
            avg_loss = total_loss / len(pbar)
            accuracy = correct / total
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}', 
                'Acc': f'{accuracy:.4f}',
                'Batch': f'{len(pbar.iterable) - len(pbar.iterable) + pbar.n}/{len(dataloader)}'
            })
    
    # Calculate detailed metrics
    if SKLEARN_AVAILABLE:
        f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
        
        detailed_report = None
        if return_detailed:
            try:
                detailed_report = classification_report(all_labels, all_predictions, 
                                                      target_names=['0', '1'], 
                                                      output_dict=True, zero_division=0)
            except:
                detailed_report = None
    else:
        f1 = 0.0
        detailed_report = None
    
    if return_detailed:
        return total_loss / len(dataloader), correct / total, f1, detailed_report
    else:
        return total_loss / len(dataloader), correct / total, f1


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """チェックポイントの保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)


def plot_training_progress(train_losses, val_losses, train_accs, val_accs, save_path):
    """学習進行状況の可視化"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='学習Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='検証Loss', linewidth=2)
    ax1.set_title('学習・検証Loss推移', fontsize=14)
    ax1.set_xlabel('エポック')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='学習Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='検証Accuracy', linewidth=2)
    ax2.set_title('学習・検証Accuracy推移', fontsize=14)
    ax2.set_xlabel('エポック')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add best accuracy annotations
    if val_accs:
        best_epoch = np.argmax(val_accs) + 1
        best_acc = max(val_accs)
        ax2.annotate(f'最高精度: {best_acc:.3f}\n(Epoch {best_epoch})', 
                    xy=(best_epoch, best_acc), xytext=(best_epoch+2, best_acc-0.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(model, dataloader, device, save_path):
    """混同行列の可視化"""
    if not SKLEARN_AVAILABLE:
        print("⚠️ sklearn not available. Skipping confusion matrix plot.")
        return
        
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for patches, labels in dataloader:
            patches = patches.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(patches)
            predicted = (outputs > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ボールなし', 'ボールあり'],
                yticklabels=['ボールなし', 'ボールあり'])
    plt.title('混同行列 (Confusion Matrix)', fontsize=14)
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Train Local Ball Classifier")
    parser.add_argument("--annotation_file", required=True, help="COCO annotation file")
    parser.add_argument("--images_dir", required=True, help="Images directory")
    parser.add_argument("--output_dir", default="./local_classifier_checkpoints", help="Output directory")
    parser.add_argument("--model_type", default="standard", choices=["standard", "efficient"], help="Model type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--device", default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.annotation_file):
        logger.error(f"Annotation file not found: {args.annotation_file}")
        return
        
    if not os.path.exists(args.images_dir):
        logger.error(f"Images directory not found: {args.images_dir}")
        return
    
    # Start training
    logger.info("Starting local classifier training...")
    logger.info(f"Annotation file: {args.annotation_file}")
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    model, history = train_local_classifier(
        annotation_file=args.annotation_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        device=args.device
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 