"""
Test TRM Grokking on Modular Arithmetic (a + b mod 97)

This script tests whether the TRM architecture can grok at all by training
on the canonical grokking task from the original OpenAI paper.

If TRM can grok modular arithmetic in ~1000-5000 epochs with proper GrokFast
settings, the architecture is fine and the trading task is the problem.

If TRM cannot grok even this simple task, there's an architecture issue.

Expected results with correct GrokFast settings (lambda=2.0, alpha=0.98):
- Train acc should reach 100% within ~100-500 epochs (memorization)
- Val acc should suddenly jump from ~1% to ~95%+ after 500-5000 epochs (grokking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.trm_model import TinyRecursiveModel
from training.meta_grokfast import MetaGrokFast, GROKFAST_PAPER_CONFIG, TRM_AGGRESSIVE_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic: (a + b) mod p

    This is the canonical grokking task from the OpenAI paper.
    """

    def __init__(self, p: int = 97, operation: str = 'add', split: str = 'train',
                 train_frac: float = 0.5, seed: int = 42):
        """
        Args:
            p: Prime modulus (97 is standard)
            operation: 'add', 'subtract', 'multiply', or 'divide'
            split: 'train' or 'val'
            train_frac: Fraction of data for training (0.5 is optimal for grokking)
            seed: Random seed for reproducible splits
        """
        self.p = p
        self.operation = operation

        # Generate all pairs (a, b) where a, b in [0, p-1]
        all_pairs = []
        all_labels = []

        for a in range(p):
            for b in range(p):
                if operation == 'add':
                    c = (a + b) % p
                elif operation == 'subtract':
                    c = (a - b) % p
                elif operation == 'multiply':
                    c = (a * b) % p
                elif operation == 'divide':
                    if b == 0:
                        continue  # Skip division by zero
                    # Modular inverse
                    b_inv = pow(b, p - 2, p)  # Fermat's little theorem
                    c = (a * b_inv) % p
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                all_pairs.append((a, b))
                all_labels.append(c)

        # Shuffle and split
        torch.manual_seed(seed)
        n = len(all_pairs)
        indices = torch.randperm(n).tolist()

        n_train = int(n * train_frac)

        if split == 'train':
            selected = indices[:n_train]
        else:
            selected = indices[n_train:]

        self.pairs = [all_pairs[i] for i in selected]
        self.labels = [all_labels[i] for i in selected]

        logger.info(f"Created {split} dataset: {len(self.pairs)} samples, p={p}, op={operation}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        c = self.labels[idx]

        # One-hot encode inputs (dimension = p for each of a and b)
        # Total input dim = 2 * p = 194 for p=97
        x = torch.zeros(2 * self.p)
        x[a] = 1.0
        x[self.p + b] = 1.0

        return x, c


class TRMForModular(nn.Module):
    """
    Wrapper to adapt TRM for modular arithmetic task.

    Adapts input/output dimensions while keeping core TRM architecture.
    """

    def __init__(self, p: int = 97, hidden_dim: int = 256,
                 num_latent_steps: int = 6, num_recursion_cycles: int = 3):
        super().__init__()

        self.p = p
        input_dim = 2 * p  # One-hot encoded (a, b)
        output_dim = p     # Predict c in [0, p-1]

        # Use smaller TRM for this task (don't need 7M params)
        self.trm = TinyRecursiveModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_latent_steps=num_latent_steps,
            num_recursion_cycles=num_recursion_cycles,
            dropout=0.0  # No dropout for grokking test
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"TRMForModular: {total_params:,} parameters")

    def forward(self, x):
        output = self.trm(x)
        return output['strategy_logits']  # Shape: (batch, p)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


def detect_grokking(history, window=50):
    """
    Detect if grokking has occurred.

    Grokking = sudden jump in val_acc after train_acc is already high.
    """
    if len(history) < window:
        return False, "Not enough data"

    recent = history[-window:]

    # Check if train acc is high
    train_accs = [h['train_acc'] for h in recent]
    avg_train = sum(train_accs) / len(train_accs)

    # Check if val acc jumped recently
    val_accs = [h['val_acc'] for h in recent]
    val_start = val_accs[0]
    val_end = val_accs[-1]
    val_jump = val_end - val_start

    if avg_train > 95 and val_end > 90:
        return True, f"GROKKING DETECTED! Val acc jumped from {val_start:.1f}% to {val_end:.1f}%"
    elif avg_train > 95 and val_end < 20:
        return False, f"Memorized but not generalized (train={avg_train:.1f}%, val={val_end:.1f}%)"
    else:
        return False, f"Still training (train={avg_train:.1f}%, val={val_end:.1f}%)"


def main():
    # Configuration
    P = 97  # Prime modulus
    OPERATION = 'add'  # Start with addition (easiest)
    TRAIN_FRAC = 0.5   # 50% train, 50% val (optimal for grokking)
    HIDDEN_DIM = 256   # Smaller than full TRM
    MAX_EPOCHS = 10000
    BATCH_SIZE = 512   # Full batch often works better for grokking

    # Use paper-matching config (no bigeometric/muon extras)
    USE_PAPER_CONFIG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create datasets
    train_dataset = ModularArithmeticDataset(p=P, operation=OPERATION, split='train', train_frac=TRAIN_FRAC)
    val_dataset = ModularArithmeticDataset(p=P, operation=OPERATION, split='val', train_frac=TRAIN_FRAC)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = TRMForModular(p=P, hidden_dim=HIDDEN_DIM).to(device)

    # Create optimizer with GrokFast
    if USE_PAPER_CONFIG:
        logger.info("Using GROKFAST_PAPER_CONFIG (lambda=2.0, alpha=0.98, wd=1.0)")
        optimizer = MetaGrokFast(model.parameters(), config=GROKFAST_PAPER_CONFIG)
    else:
        logger.info("Using TRM_AGGRESSIVE_CONFIG")
        optimizer = MetaGrokFast(model.parameters(), config=TRM_AGGRESSIVE_CONFIG)

    # Training loop
    history = []
    best_val_acc = 0
    grokking_epoch = None

    print("\n" + "=" * 80)
    print("TRM GROKKING TEST ON MODULAR ARITHMETIC")
    print(f"Task: (a + b) mod {P}")
    print(f"GrokFast lambda={optimizer.config.grokfast_lambda}, alpha={optimizer.config.grokfast_alpha}")
    print(f"Weight decay={optimizer.config.weight_decay}")
    print("=" * 80)
    print("\nExpected behavior:")
    print("  - Train acc -> 100% in ~100-500 epochs (memorization)")
    print("  - Val acc stays ~1% for a while (random guessing)")
    print("  - Val acc suddenly jumps to ~95%+ (GROKKING!)")
    print("=" * 80 + "\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Detect grokking
        grokked, status = detect_grokking(history)

        # Determine phase
        if train_acc < 50:
            phase = "LEARNING"
        elif train_acc >= 95 and val_acc < 20:
            phase = "MEMORIZED"
        elif train_acc >= 95 and val_acc >= 80:
            phase = "GROKKED!"
        else:
            phase = "PLATEAU"

        # Log progress
        if epoch % 100 == 0 or epoch <= 10 or grokked:
            gap = train_acc - val_acc
            print(f"Epoch {epoch:5d} [{phase:10s}] | "
                  f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                  f"Gap: {gap:+.1f}%")

        # Track best and grokking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if val_acc > 90 and grokking_epoch is None:
                grokking_epoch = epoch
                print(f"\n*** GROKKING DETECTED AT EPOCH {epoch}! ***\n")

        # Early stop if grokked
        if grokked and val_acc > 95:
            print(f"\nGrokking confirmed! Stopping early at epoch {epoch}")
            break

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total epochs: {len(history)}")
    print(f"Best val acc: {best_val_acc:.2f}%")
    if grokking_epoch:
        print(f"Grokking occurred at epoch: {grokking_epoch}")
        print("\nCONCLUSION: TRM architecture CAN grok!")
        print("           The trading task may be too noisy or have no learnable pattern.")
    else:
        print(f"Grokking did NOT occur in {MAX_EPOCHS} epochs")
        print("\nCONCLUSION: TRM architecture may have issues preventing grokking.")
        print("           Consider: simpler architecture, different recursion settings.")
    print("=" * 80)

    # Save results
    results = {
        'task': f'({OPERATION}) mod {P}',
        'train_frac': TRAIN_FRAC,
        'hidden_dim': HIDDEN_DIM,
        'max_epochs': MAX_EPOCHS,
        'grokfast_lambda': optimizer.config.grokfast_lambda,
        'grokfast_alpha': optimizer.config.grokfast_alpha,
        'weight_decay': optimizer.config.weight_decay,
        'best_val_acc': best_val_acc,
        'grokking_epoch': grokking_epoch,
        'final_train_acc': history[-1]['train_acc'],
        'final_val_acc': history[-1]['val_acc'],
        'history': history[-100:]  # Last 100 epochs
    }

    output_path = Path(__file__).parent.parent / 'models' / 'trm_grokking' / 'modular_arithmetic_test.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
