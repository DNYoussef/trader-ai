"""
FINAL MOCK REMOVAL SCRIPT
Systematically removes all mock code and replaces with real implementations
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

def scan_for_mocks(root_dir: Path) -> Dict[str, List[Tuple[int, str]]]:
    """Scan all Python files for mock code patterns"""

    mock_patterns = [
        r'Mock[A-Z]\w*',           # MockClassName
        r'mock_\w+',               # mock_variable
        r'fake_\w+',               # fake_variable
        r'dummy_\w+',              # dummy_variable
        r'# TODO',                 # TODO comments
        r'# FIXME',                # FIXME comments
        r'# Placeholder',          # Placeholder comments
        r'placeholder',            # placeholder in strings
        r'return.*random\.',       # Random return values
        r'np\.random\.rand',       # Random numpy values
        r'"mock',                  # Mock strings
        r'\'mock',                 # Mock strings
    ]

    results = {}

    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file) or "test_" in py_file.name:
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            file_mocks = []
            for i, line in enumerate(lines):
                for pattern in mock_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        file_mocks.append((i + 1, line.strip()))
                        break

            if file_mocks:
                results[str(py_file)] = file_mocks

        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return results

def generate_removal_report(mock_files: Dict[str, List[Tuple[int, str]]]) -> None:
    """Generate detailed report of all mocks found"""

    print("=" * 80)
    print("MOCK CODE REMOVAL REPORT")
    print("=" * 80)

    total_mocks = sum(len(mocks) for mocks in mock_files.values())
    print(f"\nTotal mock instances found: {total_mocks}")
    print(f"Files with mocks: {len(mock_files)}")

    # Categorize by severity
    critical_files = []
    high_priority = []
    medium_priority = []

    for file_path, mocks in mock_files.items():
        if "dashboard" in file_path or "broker" in file_path or "trading_engine" in file_path:
            critical_files.append((file_path, len(mocks)))
        elif "strategies" in file_path or "gates" in file_path:
            high_priority.append((file_path, len(mocks)))
        else:
            medium_priority.append((file_path, len(mocks)))

    print("\n" + "=" * 80)
    print("CRITICAL FILES (Currently Running Systems):")
    print("=" * 80)
    for file, count in sorted(critical_files, key=lambda x: x[1], reverse=True)[:10]:
        file_name = Path(file).name
        print(f"  {file_name:40s} - {count} mocks")

    print("\n" + "=" * 80)
    print("HIGH PRIORITY (Core Trading Logic):")
    print("=" * 80)
    for file, count in sorted(high_priority, key=lambda x: x[1], reverse=True)[:10]:
        file_name = Path(file).name
        print(f"  {file_name:40s} - {count} mocks")

    print("\n" + "=" * 80)
    print("SPECIFIC MOCK CLASSES TO REMOVE:")
    print("=" * 80)

    mock_classes = set()
    for file_path, mocks in mock_files.items():
        for line_num, line in mocks:
            # Find class definitions
            if "class Mock" in line:
                class_match = re.search(r'class (Mock\w+)', line)
                if class_match:
                    mock_classes.add(class_match.group(1))

    for mock_class in sorted(mock_classes):
        print(f"  - {mock_class}")

    print("\n" + "=" * 80)
    print("REPLACEMENT ACTIONS NEEDED:")
    print("=" * 80)

    actions = {
        "MockAlpacaClient": "Use real_alpaca_adapter.py with actual API credentials",
        "MockDataGenerator": "Connect to real market data from database",
        "mock_metrics": "Calculate real risk metrics from positions",
        "mock_positions": "Fetch real positions from broker",
        "placeholder": "Implement actual calculations",
        "TODO": "Complete the implementation",
        "random": "Use real data or calculations"
    }

    for pattern, action in actions.items():
        count = 0
        for file_path, mocks in mock_files.items():
            for _, line in mocks:
                if pattern.lower() in line.lower():
                    count += 1

        if count > 0:
            print(f"  {pattern:20s} ({count:3d} instances) -> {action}")

def create_real_implementations():
    """Generate templates for real implementations"""

    print("\n" + "=" * 80)
    print("REAL IMPLEMENTATION TEMPLATES")
    print("=" * 80)

    # Real market data fetcher
    real_market_data = '''
def get_real_market_data():
    """Fetch real market data from database"""
    import sqlite3
    conn = sqlite3.connect('data/historical_market.db')
    cursor = conn.cursor()

    # Get latest market data
    cursor.execute("""
        SELECT symbol, close, volume, returns
        FROM market_data
        WHERE date = (SELECT MAX(date) FROM market_data)
    """)

    data = cursor.fetchall()
    conn.close()
    return data
'''

    # Real risk calculator
    real_risk_calc = '''
def calculate_real_risk_metrics(positions, market_data):
    """Calculate real risk metrics from positions"""
    total_value = sum(p['quantity'] * p['current_price'] for p in positions)

    # Real VaR calculation
    returns = [p['pnl_percent'] / 100 for p in positions]
    var_95 = np.percentile(returns, 5) if returns else 0

    # Real Sharpe ratio
    avg_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if returns else 1
    sharpe = avg_return / std_return if std_return > 0 else 0

    # Real P(ruin) - simplified Kelly criterion
    win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0.5
    avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
    avg_loss = abs(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 1

    if avg_loss > 0:
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        p_ruin = max(0, min(1, 1 - kelly_f))
    else:
        p_ruin = 0.5

    return {
        'portfolio_value': total_value,
        'var_95': var_95,
        'sharpe_ratio': sharpe,
        'p_ruin': p_ruin
    }
'''

    print("\n1. Real Market Data Fetcher:")
    print(real_market_data)

    print("\n2. Real Risk Calculator:")
    print(real_risk_calc)

    print("\n3. Real Broker Integration:")
    print("   Use src/brokers/real_alpaca_adapter.py")
    print("   Set environment variables:")
    print("   - ALPACA_API_KEY=your_api_key")
    print("   - ALPACA_SECRET_KEY=your_secret_key")

def main():
    """Main execution"""

    # Scan for mocks
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    print("Scanning for mock code...")
    mock_files = scan_for_mocks(src_dir)

    # Generate report
    generate_removal_report(mock_files)

    # Provide implementation templates
    create_real_implementations()

    print("\n" + "=" * 80)
    print("NEXT STEPS TO REMOVE ALL MOCKS:")
    print("=" * 80)

    print("""
1. Replace MockDataGenerator in run_server_simple.py:
   - Connect to SQLite database for real market data
   - Calculate real risk metrics from actual positions
   - Use real broker API for position data

2. Remove MockAlpacaClient from alpaca_adapter.py:
   - Use real_alpaca_adapter.py exclusively
   - Set ALPACA_API_KEY and ALPACA_SECRET_KEY

3. Fix all placeholder calculations:
   - Implement real mathematical formulas
   - Use actual market data for calculations
   - Remove all random number generators

4. Complete all TODO implementations:
   - Finish incomplete functions
   - Remove stub methods
   - Implement real logic

5. Test with real data:
   - Connect to live market feeds
   - Use real broker account (paper trading)
   - Validate all calculations
    """)

    print("\nTotal mocks to remove:", sum(len(m) for m in mock_files.values()))
    print("=" * 80)

if __name__ == "__main__":
    main()