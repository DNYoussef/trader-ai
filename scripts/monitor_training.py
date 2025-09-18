"""
Monitor Black Swan Training Progress
Real-time monitoring of the strategy selection training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import sqlite3
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress"""

    print("=" * 60)
    print("BLACK SWAN TRAINING MONITOR")
    print("=" * 60)

    log_file = Path("logs/training_output.log")
    db_path = Path("data/black_swan_training.db")

    while True:
        # Check log file
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Get last 10 lines
            recent_lines = lines[-10:] if len(lines) >= 10 else lines

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Recent Activity:")
            for line in recent_lines:
                if "Processing date" in line or "ERROR" in line or "Complete" in line:
                    print(f"  {line.strip()}")

        # Check database for saved weights
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()

                    # Check if weights table exists
                    cursor.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='strategy_weights'
                    """)

                    if cursor.fetchone():
                        cursor.execute("""
                            SELECT COUNT(*) FROM strategy_weights
                        """)
                        count = cursor.fetchone()[0]

                        if count > 0:
                            print(f"\n✅ Strategy weights saved: {count} strategies")

                            cursor.execute("""
                                SELECT strategy_name, weight
                                FROM strategy_weights
                                ORDER BY weight DESC
                                LIMIT 5
                            """)

                            print("\nTop Strategies by Weight:")
                            for strategy, weight in cursor.fetchall():
                                print(f"  {strategy}: {weight:.2%}")
            except Exception as e:
                pass

        # Check for completion markers
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()

            if "Training Complete!" in content:
                print("\n" + "=" * 60)
                print("🎉 TRAINING COMPLETE!")
                print("=" * 60)
                break
            elif "ERROR" in content or "Fatal error" in content:
                print("\n⚠️ Error detected in training!")
                # Continue monitoring

        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    monitor_training()