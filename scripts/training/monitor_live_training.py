"""
Real-time training monitor with live updates
Continuously polls for new training output and updates metrics
"""

import time
import json
from datetime import datetime
from pathlib import Path
import re
import sys

class LiveTrainingMonitor:
    def __init__(self):
        self.metrics_file = Path('models/training_metrics_live.json')
        self.last_iteration = 0
        self.metrics = self.load_existing_metrics()

    def load_existing_metrics(self):
        """Load existing metrics if available"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if metrics['iterations']:
                        self.last_iteration = metrics['iterations'][-1]
                    return metrics
            except:
                pass

        return {
            'iterations': [],
            'losses': [],
            'clean_accuracies': [],
            'noisy_accuracies': [],
            'generalization_gaps': [],
            'grokking_scores': [],
            'timestamps': [],
            'training_hours': []
        }

    def parse_line(self, line):
        """Parse training output line"""
        pattern = r'Iter\s+(\d+):\s+Loss=([\d.]+),\s+Clean=([\d.]+),\s+Noisy=([\d.]+),\s+Gap=([-\d.]+),\s+Grok=([\d.]+),\s+Time=([\d.]+)h'
        match = re.search(pattern, line)

        if match:
            iteration = int(match.group(1))
            if iteration > self.last_iteration:
                return {
                    'iteration': iteration,
                    'loss': float(match.group(2)),
                    'clean_accuracy': float(match.group(3)),
                    'noisy_accuracy': float(match.group(4)),
                    'generalization_gap': float(match.group(5)),
                    'grokking_score': float(match.group(6)),
                    'training_hours': float(match.group(7))
                }
        return None

    def update_metrics(self, parsed):
        """Add new metrics"""
        if parsed:
            self.metrics['iterations'].append(parsed['iteration'])
            self.metrics['losses'].append(parsed['loss'])
            self.metrics['clean_accuracies'].append(parsed['clean_accuracy'])
            self.metrics['noisy_accuracies'].append(parsed['noisy_accuracy'])
            self.metrics['generalization_gaps'].append(parsed['generalization_gap'])
            self.metrics['grokking_scores'].append(parsed['grokking_score'])
            self.metrics['training_hours'].append(parsed['training_hours'])
            self.metrics['timestamps'].append(datetime.now().isoformat())

            self.last_iteration = parsed['iteration']
            self.save_metrics()
            return True
        return False

    def save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def display_status(self):
        """Display current training status"""
        if not self.metrics['iterations']:
            print("No training data yet")
            return

        # Clear screen (Windows)
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

        print("="*60)
        print("ENHANCED HRM 32D - LIVE TRAINING MONITOR")
        print("="*60)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Current metrics
        i = self.metrics['iterations'][-1]
        print(f"Iteration:    {i:,} / 100,000 ({i/1000:.1f}%)")
        print(f"Time:         {self.metrics['training_hours'][-1]:.2f} hours")
        print()

        print("Current Metrics:")
        print(f"  Loss:       {self.metrics['losses'][-1]:.4f}")
        print(f"  Clean Acc:  {self.metrics['clean_accuracies'][-1]:.3f}")
        print(f"  Noisy Acc:  {self.metrics['noisy_accuracies'][-1]:.3f}")
        print(f"  Gen. Gap:   {self.metrics['generalization_gaps'][-1]:+.3f}")
        print(f"  Grok Score: {self.metrics['grokking_scores'][-1]:.3f}")
        print()

        # Best metrics
        print("Best Performance:")
        print(f"  Max Clean:  {max(self.metrics['clean_accuracies']):.3f}")
        print(f"  Max Grok:   {max(self.metrics['grokking_scores']):.3f}")
        print(f"  Min Loss:   {min(self.metrics['losses']):.4f}")
        print()

        # Checkpoint status
        next_checkpoint = ((i // 5000) + 1) * 5000
        print(f"Next Checkpoint: {next_checkpoint:,} (in {next_checkpoint - i:,} iterations)")

        # Target status
        print()
        print("Target Progress:")
        clean_progress = max(self.metrics['clean_accuracies']) / 0.85 * 100
        grok_progress = max(self.metrics['grokking_scores']) / 0.90 * 100

        if clean_progress >= 100:
            print(f"  [ACHIEVED] Clean >= 85%")
        else:
            print(f"  [{clean_progress:.1f}%] Clean >= 85%")

        if grok_progress >= 100:
            print(f"  [ACHIEVED] Grok >= 0.90")
        else:
            print(f"  [{grok_progress:.1f}%] Grok >= 0.90")

        # Estimated time remaining
        if len(self.metrics['iterations']) > 1:
            hours_per_1k = self.metrics['training_hours'][-1] / (i / 1000)
            remaining_iterations = 100000 - i
            est_hours = (remaining_iterations / 1000) * hours_per_1k
            print()
            print(f"Estimated Time Remaining: {est_hours:.1f} hours")

        print("="*60)
        print("Press Ctrl+C to stop monitoring")

def monitor_training_log(log_file='logs/training_enhanced_hrm_32d.log'):
    """Monitor training log file for updates"""
    monitor = LiveTrainingMonitor()
    log_path = Path(log_file)

    print("Starting live training monitor...")
    print(f"Monitoring: {log_path}")
    print(f"Current iteration: {monitor.last_iteration}")
    print()

    # Create log file if it doesn't exist
    log_path.parent.mkdir(exist_ok=True)
    if not log_path.exists():
        log_path.touch()

    last_position = 0

    try:
        while True:
            # Check if log file has grown
            if log_path.exists():
                with open(log_path, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                # Process new lines
                updated = False
                for line in new_lines:
                    parsed = monitor.parse_line(line)
                    if monitor.update_metrics(parsed):
                        updated = True
                        print(f"[NEW] Iteration {parsed['iteration']}: "
                              f"Loss={parsed['loss']:.4f}, "
                              f"Clean={parsed['clean_accuracy']:.3f}, "
                              f"Grok={parsed['grokking_score']:.3f}")

                if updated:
                    monitor.display_status()

                    # Also run the plotting script
                    import subprocess
                    subprocess.run([sys.executable, 'scripts/track_training_metrics.py'],
                                 capture_output=True)
                    print("Graphs updated: models/training_metrics_live.png")

            # Check every 10 seconds
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Last iteration tracked: {monitor.last_iteration}")

        # Final update of plots
        import subprocess
        subprocess.run([sys.executable, 'scripts/track_training_metrics.py'])
        print("Final graphs saved to models/training_metrics_live.png")

if __name__ == "__main__":
    # Check for direct console output monitoring
    print("Monitoring training output...")

    # For now, just display current status since we're reading from background process
    monitor = LiveTrainingMonitor()
    monitor.display_status()

    # Note: To truly monitor the background process, we'd need to capture its output
    # or have it write to a log file that we can tail