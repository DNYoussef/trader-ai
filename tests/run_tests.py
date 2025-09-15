#!/usr/bin/env python3
"""
Test runner script for Foundation phase comprehensive test suite.
Validates coverage targets and generates test reports.
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse


class TestRunner:
    """Comprehensive test runner for Foundation phase"""
    
    def __init__(self, target_coverage: float = 80.0):
        self.target_coverage = target_coverage
        self.test_root = Path(__file__).parent
        self.project_root = self.test_root.parent
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "coverage_percentage": 0.0,
            "duration": 0.0,
            "test_files": [],
            "coverage_by_file": {},
            "missing_coverage": []
        }
        
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage"""
        print("🧪 Running unit tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_root),
            "--cov=foundation",
            "--cov-report=term-missing",
            "--cov-report=html:coverage/html",
            "--cov-report=json:coverage/coverage.json",
            f"--cov-fail-under={self.target_coverage}",
            "--tb=short",
            "-v",
            "-x",  # Stop on first failure for quick feedback
            "--disable-warnings"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running unit tests: {e}")
            return False
            
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("🔧 Running integration tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_root / "foundation" / "test_integration.py"),
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-m", "integration"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            print("Integration test output:", result.stdout)
            if result.stderr:
                print("Integration test errors:", result.stderr)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running integration tests: {e}")
            return False
            
    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests"""
        print("💨 Running smoke tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_root),
            "-v",
            "--tb=line",
            "--disable-warnings",
            "-m", "not slow",
            "--maxfail=5"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            print("Smoke test results:", result.stdout[-1000:])  # Last 1000 chars
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running smoke tests: {e}")
            return False
            
    def validate_coverage(self) -> bool:
        """Validate test coverage meets targets"""
        print(f"📊 Validating coverage target: {self.target_coverage}%...")
        
        coverage_file = self.project_root / "coverage" / "coverage.json"
        
        if not coverage_file.exists():
            print("❌ Coverage file not found. Run tests with coverage first.")
            return False
            
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
                
            # Extract overall coverage percentage
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
            self.results["coverage_percentage"] = total_coverage
            
            print(f"📈 Overall coverage: {total_coverage:.1f}%")
            
            # Check if target is met
            if total_coverage >= self.target_coverage:
                print(f"✅ Coverage target met: {total_coverage:.1f}% >= {self.target_coverage}%")
                return True
            else:
                print(f"❌ Coverage target not met: {total_coverage:.1f}% < {self.target_coverage}%")
                self._report_missing_coverage(coverage_data)
                return False
                
        except Exception as e:
            print(f"❌ Error validating coverage: {e}")
            return False
            
    def _report_missing_coverage(self, coverage_data: Dict[str, Any]):
        """Report files with missing coverage"""
        print("\n📋 Files with low coverage:")
        
        files = coverage_data.get("files", {})
        for filename, file_data in files.items():
            file_coverage = file_data.get("summary", {}).get("percent_covered", 0.0)
            if file_coverage < self.target_coverage:
                missing_lines = file_data.get("missing_lines", [])
                print(f"  📄 {filename}: {file_coverage:.1f}% (missing lines: {len(missing_lines)})")
                
                self.results["missing_coverage"].append({
                    "file": filename,
                    "coverage": file_coverage,
                    "missing_lines": missing_lines
                })
                
    def run_component_tests(self, component: str) -> bool:
        """Run tests for specific component"""
        print(f"🔍 Running {component} component tests...")
        
        test_patterns = {
            "broker": "test_broker_integration.py",
            "gates": "test_gate_manager.py", 
            "cycle": "test_weekly_cycle.py",
            "integration": "test_integration.py"
        }
        
        pattern = test_patterns.get(component)
        if not pattern:
            print(f"❌ Unknown component: {component}")
            return False
            
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_root / "foundation" / pattern),
            "-v",
            "--tb=short",
            "--disable-warnings"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            print(f"{component.capitalize()} test output:", result.stdout[-500:])
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running {component} tests: {e}")
            return False
            
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        print("📋 Generating test summary...")
        
        # Count test files
        test_files = list(self.test_root.glob("**/test_*.py"))
        self.results["test_files"] = [str(f.relative_to(self.test_root)) for f in test_files]
        
        # Calculate test metrics from coverage data
        coverage_file = self.project_root / "coverage" / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    
                # Extract file-level coverage
                files = coverage_data.get("files", {})
                for filename, file_data in files.items():
                    self.results["coverage_by_file"][filename] = {
                        "coverage": file_data.get("summary", {}).get("percent_covered", 0.0),
                        "statements": file_data.get("summary", {}).get("num_statements", 0),
                        "missing": file_data.get("summary", {}).get("missing_lines", 0)
                    }
                    
            except Exception as e:
                print(f"Warning: Could not parse coverage data: {e}")
                
        return self.results
        
    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print("🚀 Running complete Foundation phase test suite...\n")
        
        success = True
        
        # 1. Run smoke tests first (quick validation)
        if not self.run_smoke_tests():
            print("❌ Smoke tests failed")
            success = False
            
        # 2. Run unit tests with coverage
        if not self.run_unit_tests():
            print("❌ Unit tests failed")
            success = False
            
        # 3. Validate coverage
        if not self.validate_coverage():
            print("❌ Coverage validation failed") 
            success = False
            
        # 4. Run integration tests
        if not self.run_integration_tests():
            print("❌ Integration tests failed")
            success = False
            
        # 5. Generate summary
        summary = self.generate_test_summary()
        
        print(f"\n{'='*60}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Coverage: {summary['coverage_percentage']:.1f}%")
        print(f"Target: {self.target_coverage}%")
        print(f"Test Files: {len(summary['test_files'])}")
        print(f"Files with Coverage: {len(summary['coverage_by_file'])}")
        print(f"Low Coverage Files: {len(summary['missing_coverage'])}")
        
        if success:
            print("\n✅ All tests passed! Foundation phase test suite is ready.")
        else:
            print(f"\n❌ Some tests failed. Check output above for details.")
            
        return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Foundation phase test runner")
    parser.add_argument("--coverage-target", type=float, default=80.0,
                       help="Coverage target percentage (default: 80.0)")
    parser.add_argument("--component", choices=["broker", "gates", "cycle", "integration"],
                       help="Run tests for specific component only")
    parser.add_argument("--smoke-only", action="store_true",
                       help="Run only smoke tests")
    parser.add_argument("--no-coverage", action="store_true",
                       help="Skip coverage validation")
    
    args = parser.parse_args()
    
    runner = TestRunner(target_coverage=args.coverage_target)
    
    if args.component:
        success = runner.run_component_tests(args.component)
    elif args.smoke_only:
        success = runner.run_smoke_tests()
    else:
        success = runner.run_all_tests()
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()