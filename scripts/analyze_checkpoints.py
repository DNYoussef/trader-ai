"""
Analyze and visualize training checkpoint data for Enhanced HRM 32D model
Generates graphs showing training progress across checkpoints
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_checkpoint_data(checkpoint_dir="models"):
    """Load all checkpoint files and extract training metrics"""
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("enhanced_hrm_32d_checkpoint_*.pth"))

    # Also check for the main model file
    main_model = checkpoint_dir / "enhanced_hrm_32d_grokfast.pth"
    if main_model.exists():
        checkpoint_files.append(main_model)

    # Also check for any training history JSON
    history_file = checkpoint_dir / "enhanced_hrm_32d_history.json"

    checkpoint_data = []

    # Load data from checkpoints
    for checkpoint_path in checkpoint_files:
        try:
            print(f"Loading {checkpoint_path.name}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract iteration number from filename
            if "checkpoint_" in checkpoint_path.name:
                iteration = int(checkpoint_path.name.split("_")[-1].split(".")[0])
            else:
                iteration = checkpoint.get('iteration', 0)

            # Extract training history if available
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']

                data_point = {
                    'file': checkpoint_path.name,
                    'iteration': iteration,
                    'best_grokking_score': checkpoint.get('best_grokking_score', 0),
                    'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                    'last_modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
                }

                # Add latest metrics from history
                if history.get('iteration'):
                    data_point['last_iteration'] = history['iteration'][-1] if history['iteration'] else 0
                    data_point['last_clean_acc'] = history['clean_accuracy'][-1] if history.get('clean_accuracy') else 0
                    data_point['last_noisy_acc'] = history['noisy_accuracy'][-1] if history.get('noisy_accuracy') else 0
                    data_point['last_loss'] = history['loss'][-1] if history.get('loss') else 0
                    data_point['last_grok_score'] = history['grokking_score'][-1] if history.get('grokking_score') else 0
                    data_point['last_gap'] = history['generalization_gap'][-1] if history.get('generalization_gap') else 0

                    # Store full history for detailed plotting
                    data_point['full_history'] = history

                checkpoint_data.append(data_point)

        except Exception as e:
            print(f"Error loading {checkpoint_path.name}: {e}")

    # Load standalone history file if exists
    standalone_history = None
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                standalone_history = json.load(f)
                print(f"Loaded standalone history from {history_file.name}")
        except Exception as e:
            print(f"Error loading history file: {e}")

    return checkpoint_data, standalone_history

def plot_training_progress(checkpoint_data, standalone_history=None):
    """Create comprehensive training progress plots"""

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Enhanced HRM 32D Training Progress Analysis', fontsize=16, fontweight='bold')

    # Initialize all axes first
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    ax5 = axes[2, 0]
    ax6 = axes[2, 1]

    # Prepare data for plotting
    iterations = []
    clean_accs = []
    noisy_accs = []
    losses = []
    grok_scores = []
    gaps = []

    if checkpoint_data:
        # Sort by iteration
        checkpoint_data = sorted(checkpoint_data, key=lambda x: x.get('iteration', 0))

        # Collect all data points from all checkpoints
        for cp in checkpoint_data:
            if 'full_history' in cp:
                hist = cp['full_history']
                if hist.get('iteration'):
                    iterations.extend(hist['iteration'])
                    clean_accs.extend(hist.get('clean_accuracy', []))
                    noisy_accs.extend(hist.get('noisy_accuracy', []))
                    losses.extend(hist.get('loss', []))
                    grok_scores.extend(hist.get('grokking_score', []))
                    gaps.extend(hist.get('generalization_gap', []))

        # Plot 1: Accuracy over iterations
        if iterations and clean_accs:
            ax1.plot(iterations, clean_accs, 'b-', label='Clean Accuracy', alpha=0.7)
            if noisy_accs:
                ax1.plot(iterations, noisy_accs, 'r-', label='Noisy Accuracy', alpha=0.7)
            ax1.axhline(y=0.85, color='g', linestyle='--', label='Target (85%)', alpha=0.5)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Training Accuracy Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])

        # Plot 2: Loss over iterations
        if iterations and losses:
            ax2.plot(iterations, losses, 'purple', alpha=0.7)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Grokking Score
        if iterations and grok_scores:
            ax3.plot(iterations, grok_scores, 'orange', alpha=0.7)
            ax3.axhline(y=0.90, color='g', linestyle='--', label='Target (0.90)', alpha=0.5)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Grokking Score')
            ax3.set_title('Grokking Score Progress')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Generalization Gap
        if iterations and gaps:
            ax4.plot(iterations, gaps, 'green', alpha=0.7)
            ax4.axhline(y=0.05, color='r', linestyle='--', label='Target (<0.05)', alpha=0.5)
            ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Gap (Clean - Noisy)')
            ax4.set_title('Generalization Gap')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: Checkpoint Summary
        if checkpoint_data:
            cp_iters = [cp.get('iteration', 0) for cp in checkpoint_data]
            cp_clean = [cp.get('last_clean_acc', 0) for cp in checkpoint_data]
            cp_grok = [cp.get('last_grok_score', 0) for cp in checkpoint_data]

            if cp_iters and cp_clean:
                ax5.bar(cp_iters, cp_clean, width=2000, alpha=0.6, label='Clean Acc')
                ax5_twin = ax5.twinx()
                ax5_twin.plot(cp_iters, cp_grok, 'ro-', label='Grok Score', markersize=8)
                ax5.set_xlabel('Checkpoint Iteration')
                ax5.set_ylabel('Clean Accuracy', color='b')
                ax5_twin.set_ylabel('Grokking Score', color='r')
                ax5.set_title('Checkpoint Summary')
                ax5.grid(True, alpha=0.3)

        # Plot 6: Training Statistics
        ax6.axis('off')

        # Calculate statistics
        stats_text = "Training Statistics:\n\n"
        if clean_accs:
            stats_text += f"Best Clean Accuracy: {max(clean_accs):.3f}\n"
            stats_text += f"Latest Clean Accuracy: {clean_accs[-1]:.3f}\n\n"
        if grok_scores:
            stats_text += f"Best Grokking Score: {max(grok_scores):.3f}\n"
            stats_text += f"Latest Grokking Score: {grok_scores[-1]:.3f}\n\n"
        if gaps:
            positive_gaps = [g for g in gaps if g > 0]
            if positive_gaps:
                stats_text += f"Best Generalization Gap: {max(positive_gaps):.3f}\n"
            stats_text += f"Latest Gap: {gaps[-1]:.3f}\n\n"
        if iterations:
            stats_text += f"Total Iterations: {max(iterations)}\n"
            stats_text += f"Checkpoints Saved: {len(checkpoint_data)}\n\n"

        # Add target status
        stats_text += "Target Status:\n"
        if clean_accs and max(clean_accs) >= 0.85:
            stats_text += "[ACHIEVED] Clean Acc >= 85%\n"
        else:
            stats_text += "[PENDING] Clean Acc >= 85%\n"

        if grok_scores and max(grok_scores) >= 0.90:
            stats_text += "[ACHIEVED] Grok Score >= 0.90\n"
        else:
            stats_text += "[PENDING] Grok Score >= 0.90\n"

        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')

    # Add standalone history if available
    if standalone_history:
        if 'iterations' in standalone_history and len(standalone_history['iterations']) > 0:
            # This is from the fast training
            ax1.plot(standalone_history['iterations'], standalone_history['accuracies'],
                    'g--', label='Fast Training', alpha=0.5)
            ax1.legend()

    # Add message if no data
    if not iterations and not (standalone_history and 'iterations' in standalone_history):
        ax1.text(0.5, 0.5, 'No training data available yet\nWaiting for first checkpoint at 5000 iterations',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    plt.tight_layout()

    # Save the plot
    output_path = Path("models/training_progress.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphs saved to: {output_path}")

    # Also save as PDF for better quality
    pdf_path = Path("models/training_progress.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")

    plt.show()

    return fig

def generate_summary_report(checkpoint_data, standalone_history=None):
    """Generate a text summary report of training progress"""

    report = []
    report.append("="*80)
    report.append("ENHANCED HRM 32D TRAINING CHECKPOINT ANALYSIS")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    if checkpoint_data:
        report.append(f"Total Checkpoints Found: {len(checkpoint_data)}")
        report.append("")
        report.append("Checkpoint Details:")
        report.append("-"*40)

        for cp in sorted(checkpoint_data, key=lambda x: x.get('iteration', 0)):
            report.append(f"\nCheckpoint: {cp['file']}")
            report.append(f"  Iteration: {cp.get('iteration', 'N/A')}")
            report.append(f"  File Size: {cp.get('file_size_mb', 0):.2f} MB")
            report.append(f"  Modified: {cp.get('last_modified', 'N/A')}")

            if 'last_clean_acc' in cp:
                report.append(f"  Clean Accuracy: {cp['last_clean_acc']:.3f}")
                report.append(f"  Noisy Accuracy: {cp.get('last_noisy_acc', 0):.3f}")
                report.append(f"  Grokking Score: {cp.get('last_grok_score', 0):.3f}")
                report.append(f"  Generalization Gap: {cp.get('last_gap', 0):.3f}")
                report.append(f"  Loss: {cp.get('last_loss', 0):.4f}")
    else:
        report.append("No checkpoint files found!")

    report.append("")
    report.append("="*80)

    # Save report
    report_path = Path("models/checkpoint_analysis.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nText report saved to: {report_path}")
    print('\n'.join(report))

def main():
    """Main analysis function"""
    print("Loading checkpoint data...")
    checkpoint_data, standalone_history = load_checkpoint_data()

    if checkpoint_data or standalone_history:
        print(f"\nFound {len(checkpoint_data)} checkpoint files")
        print("\nGenerating visualizations...")
        plot_training_progress(checkpoint_data, standalone_history)

        print("\nGenerating summary report...")
        generate_summary_report(checkpoint_data, standalone_history)
    else:
        print("No checkpoint data found!")
        print("Make sure training has saved checkpoints (at 5k, 10k, 15k iterations, etc.)")

if __name__ == "__main__":
    main()