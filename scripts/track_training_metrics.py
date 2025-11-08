"""
Real-time training metrics tracker for Enhanced HRM 32D
Continuously monitors training output and saves metrics for graphing
"""

import json
import re
from datetime import datetime
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class TrainingMetricsTracker:
    def __init__(self, metrics_file="models/training_metrics_live.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics = {
            'iterations': [],
            'losses': [],
            'clean_accuracies': [],
            'noisy_accuracies': [],
            'generalization_gaps': [],
            'grokking_scores': [],
            'timestamps': [],
            'training_hours': []
        }

        # Load existing metrics if available
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                print(f"Loaded {len(self.metrics['iterations'])} existing data points")
            except:
                print("Starting fresh metrics tracking")

    def parse_training_line(self, line):
        """Parse a training output line for metrics"""
        # Pattern: Iter  1000: Loss=1.3730, Clean=0.375, Noisy=0.500, Gap=-0.125, Grok=1.333, Time=0.26h
        pattern = r'Iter\s+(\d+):\s+Loss=([\d.]+),\s+Clean=([\d.]+),\s+Noisy=([\d.]+),\s+Gap=([-\d.]+),\s+Grok=([\d.]+),\s+Time=([\d.]+)h'
        match = re.search(pattern, line)

        if match:
            return {
                'iteration': int(match.group(1)),
                'loss': float(match.group(2)),
                'clean_accuracy': float(match.group(3)),
                'noisy_accuracy': float(match.group(4)),
                'generalization_gap': float(match.group(5)),
                'grokking_score': float(match.group(6)),
                'training_hours': float(match.group(7)),
                'timestamp': datetime.now().isoformat()
            }
        return None

    def add_metrics(self, metrics_dict):
        """Add parsed metrics to tracking"""
        if metrics_dict:
            # Check if iteration already exists
            if metrics_dict['iteration'] not in self.metrics['iterations']:
                self.metrics['iterations'].append(metrics_dict['iteration'])
                self.metrics['losses'].append(metrics_dict['loss'])
                self.metrics['clean_accuracies'].append(metrics_dict['clean_accuracy'])
                self.metrics['noisy_accuracies'].append(metrics_dict['noisy_accuracy'])
                self.metrics['generalization_gaps'].append(metrics_dict['generalization_gap'])
                self.metrics['grokking_scores'].append(metrics_dict['grokking_score'])
                self.metrics['training_hours'].append(metrics_dict['training_hours'])
                self.metrics['timestamps'].append(metrics_dict['timestamp'])

                print(f"[TRACKED] Iter {metrics_dict['iteration']}: "
                      f"Loss={metrics_dict['loss']:.4f}, "
                      f"Clean={metrics_dict['clean_accuracy']:.3f}, "
                      f"Noisy={metrics_dict['noisy_accuracy']:.3f}, "
                      f"Gap={metrics_dict['generalization_gap']:.3f}, "
                      f"Grok={metrics_dict['grokking_score']:.3f}")

                # Save immediately
                self.save_metrics()
                return True
        return False

    def save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def export_to_csv(self):
        """Export metrics to CSV for analysis"""
        if self.metrics['iterations']:
            df = pd.DataFrame(self.metrics)
            csv_file = self.metrics_file.with_suffix('.csv')
            df.to_csv(csv_file, index=False)
            print(f"Exported to {csv_file}")
            return df
        return None

    def plot_live_metrics(self, save_path="models/training_metrics_live.png"):
        """Create live training metrics plots"""
        if not self.metrics['iterations']:
            print("No metrics to plot yet")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Enhanced HRM 32D Training Metrics - Live Tracking\n'
                    f'Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    fontsize=14, fontweight='bold')

        iters = self.metrics['iterations']

        # Plot 1: Loss
        ax1 = axes[0, 0]
        ax1.plot(iters, self.metrics['losses'], 'purple', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Plot 2: Accuracies
        ax2 = axes[0, 1]
        ax2.plot(iters, self.metrics['clean_accuracies'], 'b-', label='Clean', linewidth=2)
        ax2.plot(iters, self.metrics['noisy_accuracies'], 'r-', label='Noisy', linewidth=2)
        ax2.axhline(y=0.85, color='g', linestyle='--', label='Target (85%)', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Clean vs Noisy Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Plot 3: Generalization Gap
        ax3 = axes[1, 0]
        colors = ['red' if g < 0 else 'green' for g in self.metrics['generalization_gaps']]
        ax3.bar(iters, self.metrics['generalization_gaps'], color=colors, alpha=0.6)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax3.axhline(y=0.05, color='b', linestyle='--', label='Target (<0.05)', alpha=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gap (Clean - Noisy)')
        ax3.set_title('Generalization Gap')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Grokking Score
        ax4 = axes[1, 1]
        ax4.plot(iters, self.metrics['grokking_scores'], 'orange', linewidth=2, marker='o', markersize=4)
        ax4.axhline(y=0.90, color='g', linestyle='--', label='Target (0.90)', alpha=0.5)
        ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Grokking Score')
        ax4.set_title('Grokking Score Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(bottom=0)

        # Plot 5: Combined Progress
        ax5 = axes[2, 0]
        ax5_twin = ax5.twinx()
        ax5.plot(iters, self.metrics['clean_accuracies'], 'b-', label='Clean Acc', linewidth=2)
        ax5_twin.plot(iters, self.metrics['grokking_scores'], 'r-', label='Grok Score', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Clean Accuracy', color='b')
        ax5_twin.set_ylabel('Grokking Score', color='r')
        ax5.set_title('Combined Progress')
        ax5.tick_params(axis='y', labelcolor='b')
        ax5_twin.tick_params(axis='y', labelcolor='r')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Statistics
        ax6 = axes[2, 1]
        ax6.axis('off')

        # Calculate statistics
        stats_text = "Current Training Statistics:\n" + "="*30 + "\n\n"

        if self.metrics['iterations']:
            current_iter = self.metrics['iterations'][-1]
            current_loss = self.metrics['losses'][-1]
            current_clean = self.metrics['clean_accuracies'][-1]
            current_noisy = self.metrics['noisy_accuracies'][-1]
            current_gap = self.metrics['generalization_gaps'][-1]
            current_grok = self.metrics['grokking_scores'][-1]

            stats_text += f"Current Iteration: {current_iter:,}\n"
            stats_text += f"Training Time: {self.metrics['training_hours'][-1]:.2f} hours\n\n"

            stats_text += f"Current Loss: {current_loss:.4f}\n"
            stats_text += f"Current Clean Acc: {current_clean:.3f}\n"
            stats_text += f"Current Noisy Acc: {current_noisy:.3f}\n"
            stats_text += f"Current Gap: {current_gap:+.3f}\n"
            stats_text += f"Current Grok Score: {current_grok:.3f}\n\n"

            stats_text += f"Best Clean Acc: {max(self.metrics['clean_accuracies']):.3f}\n"
            stats_text += f"Best Grok Score: {max(self.metrics['grokking_scores']):.3f}\n"
            stats_text += f"Min Loss: {min(self.metrics['losses']):.4f}\n\n"

            # Check targets
            stats_text += "Target Status:\n"
            if max(self.metrics['clean_accuracies']) >= 0.85:
                stats_text += "[ACHIEVED] Clean >= 85%\n"
            else:
                progress = max(self.metrics['clean_accuracies']) / 0.85 * 100
                stats_text += f"[{progress:.1f}%] Clean >= 85%\n"

            if max(self.metrics['grokking_scores']) >= 0.90:
                stats_text += "[ACHIEVED] Grok >= 0.90\n"
            else:
                progress = max(self.metrics['grokking_scores']) / 0.90 * 100
                stats_text += f"[{progress:.1f}%] Grok >= 0.90\n"

        ax6.text(0.05, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Live metrics plot saved to {save_path}")

        # Also save as PDF
        pdf_path = Path(save_path).with_suffix('.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

        plt.close()  # Close instead of show to avoid hanging

        return fig

    def get_summary(self):
        """Get current training summary"""
        if not self.metrics['iterations']:
            return "No training data available yet"

        summary = []
        summary.append(f"Total iterations tracked: {len(self.metrics['iterations'])}")
        summary.append(f"Latest iteration: {self.metrics['iterations'][-1]}")
        summary.append(f"Training time: {self.metrics['training_hours'][-1]:.2f} hours")
        summary.append(f"Current metrics:")
        summary.append(f"  Loss: {self.metrics['losses'][-1]:.4f}")
        summary.append(f"  Clean Accuracy: {self.metrics['clean_accuracies'][-1]:.3f}")
        summary.append(f"  Noisy Accuracy: {self.metrics['noisy_accuracies'][-1]:.3f}")
        summary.append(f"  Generalization Gap: {self.metrics['generalization_gaps'][-1]:+.3f}")
        summary.append(f"  Grokking Score: {self.metrics['grokking_scores'][-1]:.3f}")
        summary.append(f"Best performance:")
        summary.append(f"  Max Clean Acc: {max(self.metrics['clean_accuracies']):.3f}")
        summary.append(f"  Max Grok Score: {max(self.metrics['grokking_scores']):.3f}")

        return "\n".join(summary)


def monitor_training_output(log_file=None):
    """Monitor training output and extract metrics"""
    tracker = TrainingMetricsTracker()

    # If log file provided, parse it
    if log_file and Path(log_file).exists():
        print(f"Parsing log file: {log_file}")
        with open(log_file, 'r') as f:
            for line in f:
                metrics = tracker.parse_training_line(line)
                if metrics:
                    tracker.add_metrics(metrics)

    # Display summary
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(tracker.get_summary())

    # Export to CSV
    df = tracker.export_to_csv()

    # Generate plots
    tracker.plot_live_metrics()

    return tracker


def extract_current_metrics():
    """Extract metrics from current training output"""
    tracker = TrainingMetricsTracker()

    # Parse the known training output
    training_output = """
Iter     0: Loss=2.0827, Clean=0.062, Noisy=0.125, Gap=-0.062, Grok=2.000, Time=0.00h
Iter  1000: Loss=1.3730, Clean=0.375, Noisy=0.500, Gap=-0.125, Grok=1.333, Time=0.26h
Iter  2000: Loss=1.0519, Clean=0.625, Noisy=0.688, Gap=-0.062, Grok=1.100, Time=0.47h
Iter  3000: Loss=1.0185, Clean=0.375, Noisy=0.562, Gap=-0.188, Grok=1.500, Time=0.69h
    """

    for line in training_output.strip().split('\n'):
        metrics = tracker.parse_training_line(line)
        if metrics:
            tracker.add_metrics(metrics)

    return tracker


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Monitor specific log file
        tracker = monitor_training_output(sys.argv[1])
    else:
        # Extract current known metrics
        tracker = extract_current_metrics()

        print("\n" + "="*60)
        print("CURRENT TRAINING METRICS")
        print("="*60)
        print(tracker.get_summary())

        # Generate plots
        tracker.plot_live_metrics()