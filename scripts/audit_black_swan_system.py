"""
Comprehensive Audit of Black Swan Hunting System
Verifies dataset quality, labeling accuracy, and system readiness
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlackSwanSystemAuditor:
    """Comprehensive auditor for the Black Swan Hunting System"""

    def __init__(self):
        self.audit_results = {
            'database': {},
            'data_quality': {},
            'labeling': {},
            'code_quality': {},
            'system_readiness': {}
        }

    def audit_database(self):
        """Audit database structure and contents"""
        logger.info("=" * 60)
        logger.info("DATABASE AUDIT")
        logger.info("=" * 60)

        db_paths = [
            Path("data/historical_market.db"),
            Path("data/black_swan_training.db")
        ]

        for db_path in db_paths:
            if not db_path.exists():
                logger.error(f"Database not found: {db_path}")
                self.audit_results['database'][str(db_path)] = "NOT FOUND"
                continue

            logger.info(f"\nAuditing {db_path}...")

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                logger.info(f"Tables: {[t[0] for t in tables]}")

                # Audit each table
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    logger.info(f"  {table_name}: {count:,} records")

                    # Get columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    col_names = [col[1] for col in columns]
                    logger.info(f"    Columns: {col_names[:5]}..." if len(col_names) > 5 else f"    Columns: {col_names}")

                    self.audit_results['database'][f"{db_path.name}/{table_name}"] = {
                        'records': count,
                        'columns': len(col_names)
                    }

        return self.audit_results['database']

    def audit_data_quality(self):
        """Audit the quality of market data"""
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY AUDIT")
        logger.info("=" * 60)

        db_path = Path("data/historical_market.db")
        if not db_path.exists():
            logger.error("Market database not found")
            return

        with sqlite3.connect(db_path) as conn:
            # Load sample data
            query = """
            SELECT * FROM market_data
            ORDER BY RANDOM()
            LIMIT 10000
            """
            df = pd.read_sql_query(query, conn)

            if df.empty:
                logger.error("No data to audit")
                return

            # Date coverage
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(date), MAX(date) FROM market_data")
            date_range = cursor.fetchone()
            logger.info(f"Date range: {date_range[0]} to {date_range[1]}")

            # Calculate years of data
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            years_of_data = (end_date - start_date).days / 365.25
            logger.info(f"Years of data: {years_of_data:.1f}")

            self.audit_results['data_quality']['years_of_data'] = years_of_data

            # Symbol coverage
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
            unique_symbols = cursor.fetchone()[0]
            logger.info(f"Unique symbols: {unique_symbols}")

            # Get symbol distribution
            cursor.execute("""
                SELECT symbol, COUNT(*) as count
                FROM market_data
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
            """)
            top_symbols = cursor.fetchall()
            logger.info("\nTop symbols by data points:")
            for symbol, count in top_symbols:
                logger.info(f"  {symbol}: {count:,}")

            # Data completeness
            logger.info("\nData completeness:")
            null_counts = df.isnull().sum()
            for col, nulls in null_counts.items():
                if nulls > 0:
                    pct = (nulls / len(df)) * 100
                    logger.info(f"  {col}: {pct:.2f}% missing")

            # Returns distribution
            if 'returns' in df.columns:
                returns = df['returns'].dropna()
                logger.info("\nReturns distribution:")
                logger.info(f"  Mean: {returns.mean():.6f}")
                logger.info(f"  Std Dev: {returns.std():.4f}")
                logger.info(f"  Skewness: {returns.skew():.2f}")
                logger.info(f"  Kurtosis: {returns.kurtosis():.2f}")

                # Extreme events
                extreme_negative = (returns < -0.05).sum()
                extreme_positive = (returns > 0.05).sum()
                total = len(returns)
                logger.info(f"\nExtreme moves (>5%):")
                logger.info(f"  Negative: {extreme_negative} ({extreme_negative/total*100:.2f}%)")
                logger.info(f"  Positive: {extreme_positive} ({extreme_positive/total*100:.2f}%)")

                self.audit_results['data_quality']['extreme_events'] = {
                    'negative': extreme_negative,
                    'positive': extreme_positive
                }

    def audit_labeling(self):
        """Audit black swan labeling"""
        logger.info("\n" + "=" * 60)
        logger.info("BLACK SWAN LABELING AUDIT")
        logger.info("=" * 60)

        # Check for labeled data
        db_path = Path("data/black_swan_training.db")
        if not db_path.exists():
            logger.warning("Black swan database not found")
            return

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check for labeled market data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='labeled_market_data'")
            if not cursor.fetchone():
                logger.warning("No labeled market data found - running labeling now...")
                self._run_labeling()
                return

            # Analyze labeled data
            query = "SELECT * FROM labeled_market_data WHERE is_black_swan = 1"
            black_swans = pd.read_sql_query(query, conn)

            if not black_swans.empty:
                logger.info(f"Black swan events found: {len(black_swans)}")

                # Group by year
                black_swans['year'] = pd.to_datetime(black_swans['date']).dt.year
                by_year = black_swans.groupby('year').size()
                logger.info("\nBlack swan events by year:")
                for year, count in by_year.items():
                    logger.info(f"  {year}: {count} events")

                # Top affected symbols
                by_symbol = black_swans.groupby('symbol').size().sort_values(ascending=False).head(10)
                logger.info("\nMost affected symbols:")
                for symbol, count in by_symbol.items():
                    logger.info(f"  {symbol}: {count} events")

                # Severity analysis
                if 'returns' in black_swans.columns:
                    returns = black_swans['returns']
                    logger.info("\nBlack swan severity:")
                    logger.info(f"  Worst crash: {returns.min():.2%}")
                    logger.info(f"  Biggest rally: {returns.max():.2%}")
                    logger.info(f"  Mean magnitude: {returns.abs().mean():.2%}")

                self.audit_results['labeling']['black_swan_count'] = len(black_swans)
            else:
                logger.warning("No black swan events labeled")

    def _run_labeling(self):
        """Run labeling on existing data"""
        from src.data.historical_data_manager import HistoricalDataManager
        from src.data.black_swan_labeler import BlackSwanLabeler

        logger.info("Running black swan labeling...")

        manager = HistoricalDataManager()
        labeler = BlackSwanLabeler()

        # Get all data
        df = manager.get_training_data(
            start_date="1995-01-01",
            end_date="2025-12-31",
            symbols=None  # All symbols
        )

        if df.empty:
            logger.error("No data to label")
            return

        # Label by symbol
        labeled_data = []
        symbols = df['symbol'].unique()

        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            if len(symbol_df) > 20:
                try:
                    labeled_df = labeler.label_tail_events(symbol_df)
                    labeled_data.append(labeled_df)
                except Exception as e:
                    logger.warning(f"Error labeling {symbol}: {e}")

        if labeled_data:
            combined = pd.concat(labeled_data, ignore_index=True)

            # Save to database
            db_path = Path("data/black_swan_training.db")
            with sqlite3.connect(db_path) as conn:
                combined.to_sql('labeled_market_data', conn, if_exists='replace', index=False)
                logger.info(f"Labeled {len(combined)} records")

    def audit_code_quality(self):
        """Audit code for TODOs, mock functions, and production readiness"""
        logger.info("\n" + "=" * 60)
        logger.info("CODE QUALITY AUDIT")
        logger.info("=" * 60)

        # Files to audit
        audit_files = [
            "src/data/historical_data_manager.py",
            "src/data/black_swan_labeler.py",
            "src/strategies/black_swan_strategies.py",
            "src/strategies/convex_reward_function.py",
            "src/intelligence/local_llm_orchestrator.py",
            "scripts/launch_black_swan_trading.py"
        ]

        issues_found = []

        for file_path in audit_files:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            with open(path, 'r') as f:
                content = f.read()
                lines = content.splitlines()

            logger.info(f"\nAuditing {file_path}...")

            # Check for TODOs
            todos = []
            for i, line in enumerate(lines, 1):
                if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
                    todos.append((i, line.strip()))

            if todos:
                logger.warning(f"  Found {len(todos)} TODO/FIXME markers:")
                for line_num, line in todos[:3]:  # Show first 3
                    logger.warning(f"    Line {line_num}: {line}")
                issues_found.append(f"{file_path}: {len(todos)} TODOs")

            # Check for mock/placeholder code
            mock_patterns = ['mock', 'placeholder', 'dummy', 'fake', 'test_only']
            mocks = []
            for i, line in enumerate(lines, 1):
                line_lower = line.lower()
                for pattern in mock_patterns:
                    if pattern in line_lower and 'unittest.mock' not in line:
                        mocks.append((i, line.strip()))
                        break

            if mocks:
                logger.warning(f"  Found {len(mocks)} potential mock/placeholder code:")
                for line_num, line in mocks[:3]:
                    logger.warning(f"    Line {line_num}: {line}")
                issues_found.append(f"{file_path}: {len(mocks)} mocks")

            # Check for hardcoded values that should be configurable
            hardcoded = []
            patterns = [
                (r'localhost:\d+', 'hardcoded localhost URL'),
                (r'["\']http://.*["\']', 'hardcoded URL'),
                (r'["\']api_key["\'].*=.*["\'].*["\']', 'hardcoded API key'),
            ]

            import re
            for i, line in enumerate(lines, 1):
                for pattern, desc in patterns:
                    if re.search(pattern, line):
                        hardcoded.append((i, desc))

            if hardcoded:
                logger.warning(f"  Found {len(hardcoded)} hardcoded values")

            logger.info(f"  ✓ File audited: {len(lines)} lines")

        self.audit_results['code_quality']['issues'] = issues_found

        if not issues_found:
            logger.info("\n✅ Code quality audit passed - no TODOs or mock code found")
        else:
            logger.warning(f"\n⚠️ Code quality issues found: {len(issues_found)} files with issues")

    def audit_system_readiness(self):
        """Check if system is ready for production"""
        logger.info("\n" + "=" * 60)
        logger.info("SYSTEM READINESS AUDIT")
        logger.info("=" * 60)

        readiness_checks = {
            'Database exists': Path("data/historical_market.db").exists(),
            'Black swan DB exists': Path("data/black_swan_training.db").exists(),
            'Has market data': False,
            'Has labeled data': False,
            'Strategies implemented': False,
            'Reward function ready': False,
            'LLM orchestrator ready': False,
            'Ollama installed': False,
            'Model downloaded': False
        }

        # Check for market data
        try:
            db_path = Path("data/historical_market.db")
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM market_data")
                    count = cursor.fetchone()[0]
                    readiness_checks['Has market data'] = count > 1000
                    logger.info(f"Market data records: {count:,}")
        except Exception as e:
            logger.error(f"Error checking market data: {e}")

        # Check for labeled data
        try:
            db_path = Path("data/black_swan_training.db")
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM black_swan_events")
                    count = cursor.fetchone()[0]
                    readiness_checks['Has labeled data'] = count > 0
                    logger.info(f"Black swan events: {count}")
        except Exception as e:
            logger.error(f"Error checking labeled data: {e}")

        # Check code components
        readiness_checks['Strategies implemented'] = Path("src/strategies/black_swan_strategies.py").exists()
        readiness_checks['Reward function ready'] = Path("src/strategies/convex_reward_function.py").exists()
        readiness_checks['LLM orchestrator ready'] = Path("src/intelligence/local_llm_orchestrator.py").exists()

        # Check Ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            readiness_checks['Ollama installed'] = result.returncode == 0
            readiness_checks['Model downloaded'] = 'mistral' in result.stdout.lower()
        except:
            pass

        # Print results
        logger.info("\nSystem Readiness Checklist:")
        ready_count = 0
        for check, status in readiness_checks.items():
            emoji = "✅" if status else "❌"
            logger.info(f"  {emoji} {check}")
            if status:
                ready_count += 1

        self.audit_results['system_readiness'] = readiness_checks

        # Overall assessment
        total_checks = len(readiness_checks)
        readiness_pct = (ready_count / total_checks) * 100

        logger.info(f"\nOverall Readiness: {ready_count}/{total_checks} ({readiness_pct:.0f}%)")

        if readiness_pct >= 80:
            logger.info("✅ System is READY for black swan hunting!")
        elif readiness_pct >= 60:
            logger.info("⚠️ System is PARTIALLY ready - some components missing")
        else:
            logger.info("❌ System is NOT ready - critical components missing")

        return readiness_pct

    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        logger.info("\n" + "=" * 60)
        logger.info("AUDIT REPORT SUMMARY")
        logger.info("=" * 60)

        # Database summary
        logger.info("\n📊 DATABASE STATUS:")
        for db, status in self.audit_results['database'].items():
            if isinstance(status, dict):
                logger.info(f"  {db}: {status.get('records', 0):,} records")

        # Data quality summary
        if self.audit_results['data_quality']:
            logger.info("\n📈 DATA QUALITY:")
            years = self.audit_results['data_quality'].get('years_of_data', 0)
            logger.info(f"  Years of data: {years:.1f}")
            extremes = self.audit_results['data_quality'].get('extreme_events', {})
            if extremes:
                logger.info(f"  Extreme events: {extremes.get('negative', 0) + extremes.get('positive', 0)}")

        # Labeling summary
        if self.audit_results['labeling']:
            logger.info("\n🏷️ LABELING STATUS:")
            swan_count = self.audit_results['labeling'].get('black_swan_count', 0)
            logger.info(f"  Black swan events labeled: {swan_count}")

        # Code quality summary
        logger.info("\n💻 CODE QUALITY:")
        issues = self.audit_results['code_quality'].get('issues', [])
        if not issues:
            logger.info("  ✅ No issues found")
        else:
            logger.info(f"  ⚠️ {len(issues)} files with issues")

        # System readiness
        logger.info("\n🚀 SYSTEM READINESS:")
        readiness = self.audit_results['system_readiness']
        if readiness:
            ready = sum(1 for v in readiness.values() if v)
            total = len(readiness)
            logger.info(f"  {ready}/{total} checks passed")

        return self.audit_results

def main():
    """Main audit execution"""
    auditor = BlackSwanSystemAuditor()

    try:
        # Run all audits
        auditor.audit_database()
        auditor.audit_data_quality()
        auditor.audit_labeling()
        auditor.audit_code_quality()
        readiness_pct = auditor.audit_system_readiness()

        # Generate report
        report = auditor.generate_audit_report()

        # Save report
        import json
        report_path = Path("data/audit_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n📄 Audit report saved to: {report_path}")

        return readiness_pct >= 60  # Return True if at least 60% ready

    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)