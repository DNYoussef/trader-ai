import sys
import os
from pathlib import Path

# Add parent directory to path to find src
sys.path.insert(0, str(Path(__file__).parent))

from src.dashboard.run_server_simple import SimpleDashboardServer

if __name__ == "__main__":
    print("Starting Trader AI Dashboard on port 8001...")
    server = SimpleDashboardServer()
    server.run(port=8001)
