"""
Train TRM on Full Black Swan Dataset (All 12 Crisis Periods)

This script trains the TRM model on all 1,201 samples across 12 black swan periods:
- Asian Financial Crisis (1997)
- Dot-com Crash (2000-2002)
- 9/11 Attacks (2001)
- Financial Crisis (2008-2009)
- Flash Crash (2010)
- European Debt Crisis (2011-2012)
- China Selloff (2015)
- Brexit (2016)
- COVID-19 Crash (2020)
- GameStop Short Squeeze (2021)
- Russia-Ukraine (2022)
- SVB Banking Crisis (2023)

Target: >65% accuracy on 8-way strategy classification
"""

import sys
from pathlib import Path
import logging
import torch
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models.trm_model import TinyRecursiveModel
from models.trm_config import TRMConfig
from training.trm_data_loader import TRMDataModule
from training.trm_trainer import TRMTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trm_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Train TRM on full black swan dataset"""

    print("=" * 80)
    print("TRM TRAINING - FULL BLACK SWAN DATASET")
    print("=" * 80)
    print()

    # Configuration
    print("[1/7] Loading configuration...")
    config_path = Path(__file__).parent.parent.parent / 'config' / 'trm_config.json'

    if config_path.exists():
        config = TRMConfig.load(str(config_path))
        print(f"[OK] Configuration loaded from {config_path}")
    else:
        config = TRMConfig()
        print("[OK] Using default configuration")

    print(f"  - Model: {config.model.input_dim} input -> {config.model.hidden_dim} hidden -> {config.model.output_dim} output")
    print(f"  - Recursion: T={config.model.num_recursion_cycles} cycles x n={config.model.num_latent_steps} steps = {config.model.effective_depth} layers")
    print(f"  - Optimizer: {config.training.optimizer} (lr={config.training.learning_rate})")
    print(f"  - GrokFast: {'Enabled' if config.training.use_grokfast else 'Disabled'} (alpha={config.training.grokfast_alpha})")
    print()

    # Data Loading
    print("[2/7] Loading training data...")
    data_path = Path(__file__).parent.parent.parent / 'data' / 'trm_training' / 'black_swan_labels.parquet'

    if not data_path.exists():
        print(f"[ERROR] ERROR: Training data not found at {data_path}")
        print("Please run: python scripts/trm/generate_black_swan_labels.py")
        return

    # Load labels to show statistics
    labels_df = pd.read_parquet(data_path)
    print(f"[OK] Loaded {len(labels_df):,} training samples")
    print(f"  - Date range: {labels_df['date'].min()} to {labels_df['date'].max()}")
    print(f"  - Crisis periods: {labels_df['period_name'].nunique()}")
    print()

    # Strategy distribution
    print("Strategy Distribution:")
    strategy_names = config.strategies.strategy_names
    strategy_counts = labels_df['strategy_idx'].value_counts().sort_index()
    for idx, count in strategy_counts.items():
        pct = (count / len(labels_df)) * 100
        print(f"  {idx}: {strategy_names[idx]:20s} - {count:4d} samples ({pct:5.1f}%)")
    print()

    # Calculate class weights for imbalanced data
    # weight_i = total_samples / (num_classes * class_i_count)
    num_classes = len(config.strategies.strategy_names)
    total_samples = len(labels_df)
    class_weights = torch.zeros(num_classes, dtype=torch.float32)

    for idx in range(num_classes):
        count = strategy_counts.get(idx, 0)
        if count > 0:
            class_weights[idx] = total_samples / (num_classes * count)
        else:
            class_weights[idx] = 0.0  # No samples for this class

    print("Class Weights (for imbalance correction):")
    for idx in range(num_classes):
        if class_weights[idx] > 0:
            print(f"  {idx}: {strategy_names[idx]:20s} - weight={class_weights[idx]:.3f}")
    print()

    # Crisis period distribution
    print("Crisis Period Distribution:")
    for period_name, period_df in labels_df.groupby('period_name'):
        print(f"  {period_name:30s} - {len(period_df):4d} samples")
    print()

    # Create data module
    print("[3/7] Creating data loaders...")
    data_module = TRMDataModule(
        data_path=str(data_path),
        random_seed=42
    )

    train_loader, val_loader, test_loader = data_module.create_dataloaders(
        batch_size=config.training.batch_size,
        shuffle=True
    )

    print(f"[OK] Data loaders created")
    print(f"  - Training batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"  - Validation batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")
    print(f"  - Test batches: {len(test_loader)} ({len(test_loader.dataset)} samples)")
    print(f"  - Batch size: {config.training.batch_size}")
    print()

    # Save normalization parameters
    norm_path = Path(__file__).parent.parent.parent / 'config' / 'trm_normalization.json'
    data_module.save_normalization_params(str(norm_path))
    print(f"[OK] Normalization parameters saved to {norm_path}")
    print()

    # Model creation
    print("[4/7] Creating TRM model...")
    model = TinyRecursiveModel(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        num_latent_steps=config.model.num_latent_steps,
        num_recursion_cycles=config.model.num_recursion_cycles
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] TRM model created")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Effective depth: {config.model.effective_depth} layers")
    print()

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[OK] Using device: {device}")
    if device == 'cuda':
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Trainer creation
    print("[5/7] Creating trainer...")
    trainer = TRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config.training.learning_rate,
        alpha=config.training.grokfast_alpha if config.training.use_grokfast else 0.0,
        lambda_=0.1,  # GrokFast filter strength
        weight_decay=config.training.weight_decay if hasattr(config.training, 'weight_decay') else 0.01,
        T=config.model.num_recursion_cycles,
        n=config.model.num_latent_steps,
        class_weights=class_weights  # Add class weights for imbalanced data
    )
    print(f"[OK] Trainer created with GrokFast optimizer and class weighting")
    print()

    # Training
    print("[6/7] Starting training...")
    print("=" * 80)
    print()

    num_epochs = 100  # More epochs for full dataset
    early_stopping_patience = 15  # Patience for early stopping

    print(f"Training configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Early stopping patience: {early_stopping_patience}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Batch size: {config.training.batch_size}")
    print()

    try:
        # Train the model
        trainer.fit(
            num_epochs=num_epochs,
            patience=early_stopping_patience
        )

        print()
        print("=" * 80)
        print("[7/7] Training complete!")
        print("=" * 80)
        print()

        # Final evaluation (on validation set)
        print("Final Evaluation on Validation Set:")
        final_metrics = trainer.validate()

        print(f"  - Validation Loss: {final_metrics['loss']:.4f}")
        print(f"  - Validation Accuracy: {final_metrics['accuracy']:.2%}")
        print(f"  - Top-3 Accuracy: {final_metrics['top3_accuracy']:.2%}")
        print()
        print("Note: Test set evaluation will be performed separately after training")
        print()

        # Success criteria
        target_accuracy = 0.65
        if final_metrics['accuracy'] >= target_accuracy:
            print(f"[SUCCESS] SUCCESS: Achieved {final_metrics['accuracy']:.2%} accuracy (target: {target_accuracy:.0%})")
        else:
            print(f"[WARNING]  WARNING: Accuracy {final_metrics['accuracy']:.2%} below target {target_accuracy:.0%}")
        print()

        # Training summary
        print("Training Summary:")
        print(f"  - Best validation accuracy: {trainer.best_val_acc:.2f}%")
        print(f"  - Total epochs: {len(trainer.train_history)}")
        print(f"  - Final training loss: {trainer.train_history[-1]['loss']:.4f}")
        print(f"  - Final validation loss: {trainer.val_history[-1]['loss']:.4f}")
        print()

        # Save checkpoint
        checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / 'training_checkpoint.pkl'
        trainer.save_checkpoint(checkpoint_path, len(trainer.train_history) - 1, final_metrics)
        print(f"[OK] Model checkpoint saved to {checkpoint_path}")
        print()

        # Save training history
        history_path = checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            history = {
                'train_history': trainer.train_history,
                'val_history': trainer.val_history
            }
            json.dump(history, f, indent=2)
        print(f"[OK] Training history saved to {history_path}")
        print()

        # Save final metrics
        metrics_path = checkpoint_dir / 'final_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'val_loss': final_metrics['loss'],
                'val_accuracy': final_metrics['accuracy'],
                'top3_accuracy': final_metrics['top3_accuracy'],
                'best_val_acc': trainer.best_val_acc,
                'total_epochs': len(trainer.train_history),
                'date_trained': datetime.now().isoformat()
            }, f, indent=2)
        print(f"[OK] Final metrics saved to {metrics_path}")
        print()

        print("=" * 80)
        print("TRM TRAINING COMPLETE!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Review training curves in training_history.json")
        print("  2. Test model on historical backtests")
        print("  3. Integrate with trading_engine.py for paper trading")
        print("  4. Monitor performance in production")
        print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
        print()
        print("Saving current checkpoint...")
        checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / 'trm_interrupted.pth'
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"[OK] Checkpoint saved to {checkpoint_path}")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print(f"[ERROR] ERROR during training: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
