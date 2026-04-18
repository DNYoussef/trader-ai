"""
Analyze what the TRM model has learned during training.
Examines: feature importance, per-class performance, confusion patterns.
"""
import torch
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

from pathlib import Path
from collections import defaultdict

# Strategy names
STRATEGIES = [
    'ultra_defensive', 'defensive', 'balanced_defensive', 'balanced_growth',
    'growth', 'aggressive_growth', 'contrarian_long', 'contrarian_short'
]

def load_model_and_data():
    """Load the latest checkpoint and test data."""
    model_dir = Path('D:/Projects/trader-ai/models/trm_grokking')
    data_path = Path('D:/Projects/trader-ai/data/trm_training/labels_110_features_noisy.parquet')

    # Find latest checkpoint from current training run (Dec 16)
    checkpoints = sorted(model_dir.glob('checkpoint_epoch_*.pt'))
    dec16_checkpoints = [c for c in checkpoints if c.stat().st_size < 10_000_000]  # Current run has smaller files

    if dec16_checkpoints:
        latest = max(dec16_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Loading checkpoint: {latest.name}")
    else:
        latest = model_dir / 'final_model.pt'
        print(f"Loading final model")

    checkpoint = torch.load(latest, map_location='cpu', weights_only=False)

    # Load data
    df = pd.read_parquet(data_path)

    return checkpoint, df, latest.stem

def analyze_predictions(checkpoint, df):
    """Analyze model predictions on the data."""
    from src.models.trm_model import TinyRecursiveModel as TRMModel

    # Get model config from checkpoint
    model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Infer model dimensions from state dict
    input_dim = model_state['input_proj.weight'].shape[1]
    hidden_dim = model_state['input_proj.weight'].shape[0]

    print(f"\nModel architecture: input_dim={input_dim}, hidden_dim={hidden_dim}")

    # Create model - use correct params for TinyRecursiveModel
    model = TRMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=8
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Prepare features - features are stored as arrays in a single column
    X = np.stack(df['features'].values).astype(np.float32)
    y_true = df['strategy_idx'].values
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]

    # Load normalization params
    import json
    with open('D:/Projects/trader-ai/models/trm_grokking/normalization_params.json') as f:
        norm_params = json.load(f)

    mean = np.array(norm_params['mean'])
    std = np.array(norm_params['std'])
    std[std < 1e-7] = 1.0  # Avoid division by zero

    X_norm = (X - mean) / std

    # Get predictions - model returns dict with 'strategy_logits'
    with torch.no_grad():
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        output = model(X_tensor)
        logits = output['strategy_logits']
        probs = torch.softmax(logits, dim=-1)
        y_pred = logits.argmax(dim=-1).numpy()
        confidence = probs.max(dim=-1).values.numpy()

    return y_true, y_pred, probs.numpy(), confidence, feature_cols

def compute_feature_importance(checkpoint):
    """Compute feature importance from input projection weights."""
    model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Input projection: hidden_dim x input_dim
    weights = model_state['input_proj.weight'].numpy()

    # Feature importance = L2 norm of weights for each input feature
    importance = np.sqrt((weights ** 2).sum(axis=0))
    importance = importance / importance.sum()  # Normalize

    return importance

def main():
    print("=" * 60)
    print("TRM MODEL LEARNING ANALYSIS")
    print("=" * 60)

    checkpoint, df, ckpt_name = load_model_and_data()

    # Extract epoch from checkpoint name
    epoch = ckpt_name.split('_')[-1] if 'epoch' in ckpt_name else 'final'
    print(f"\nAnalyzing model at epoch: {epoch}")

    # Get predictions
    y_true, y_pred, probs, confidence, feature_cols = analyze_predictions(checkpoint, df)

    # Overall accuracy
    overall_acc = (y_true == y_pred).mean() * 100
    print(f"\nOverall Accuracy: {overall_acc:.1f}%")
    print(f"Average Confidence: {confidence.mean():.3f}")

    # Per-class analysis
    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Count':>6} {'Acc':>8} {'Conf':>8} {'Predicted As':>30}")
    print("-" * 80)

    for i, strategy in enumerate(STRATEGIES):
        mask = y_true == i
        count = mask.sum()
        if count == 0:
            continue

        acc = (y_pred[mask] == i).mean() * 100
        conf = confidence[mask].mean()

        # What does the model predict for this class?
        pred_dist = pd.Series(y_pred[mask]).value_counts(normalize=True)
        top_preds = [(STRATEGIES[idx][:8], f"{pct*100:.0f}%") for idx, pct in pred_dist.head(3).items()]
        top_preds_str = ", ".join([f"{n}:{p}" for n, p in top_preds])

        print(f"{strategy:<20} {count:>6} {acc:>7.1f}% {conf:>7.3f}  {top_preds_str}")

    # Confusion patterns
    print("\n" + "=" * 60)
    print("TOP CONFUSION PATTERNS (mistakes)")
    print("=" * 60)

    confusions = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            confusions[(STRATEGIES[true], STRATEGIES[pred])] += 1

    sorted_confusions = sorted(confusions.items(), key=lambda x: -x[1])[:10]
    for (true_s, pred_s), count in sorted_confusions:
        print(f"  {true_s:<20} -> {pred_s:<20} : {count} times")

    # Feature importance
    print("\n" + "=" * 60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 60)

    importance = compute_feature_importance(checkpoint)
    sorted_idx = np.argsort(importance)[::-1]

    for rank, idx in enumerate(sorted_idx[:20], 1):
        print(f"  {rank:>2}. {feature_cols[idx]:<35} {importance[idx]*100:>6.2f}%")

    # Strategy confidence distribution
    print("\n" + "=" * 60)
    print("STRATEGY PREDICTION CONFIDENCE")
    print("=" * 60)

    for i, strategy in enumerate(STRATEGIES):
        # How confident is the model when predicting this strategy?
        pred_mask = y_pred == i
        if pred_mask.sum() > 0:
            conf_when_pred = confidence[pred_mask].mean()
            correct_conf = confidence[pred_mask & (y_true == i)].mean() if (pred_mask & (y_true == i)).sum() > 0 else 0
            wrong_conf = confidence[pred_mask & (y_true != i)].mean() if (pred_mask & (y_true != i)).sum() > 0 else 0
            print(f"  {strategy:<20} Pred {pred_mask.sum():>5}x | Avg conf: {conf_when_pred:.3f} | Correct: {correct_conf:.3f} | Wrong: {wrong_conf:.3f}")

    # Training phase analysis
    print("\n" + "=" * 60)
    print("LEARNING STATE SUMMARY")
    print("=" * 60)

    if 'metrics_history' in checkpoint:
        metrics = checkpoint['metrics_history']
        recent = metrics[-10:] if len(metrics) >= 10 else metrics

        train_accs = [m['train_acc'] for m in recent]
        val_accs = [m['val_acc'] for m in recent]
        gaps = [m['generalization_gap'] for m in recent]

        print(f"  Recent train acc: {np.mean(train_accs):.1f}% (+/- {np.std(train_accs):.1f}%)")
        print(f"  Recent val acc:   {np.mean(val_accs):.1f}% (+/- {np.std(val_accs):.1f}%)")
        print(f"  Recent gap:       {np.mean(gaps):.1f}% (+/- {np.std(gaps):.1f}%)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
