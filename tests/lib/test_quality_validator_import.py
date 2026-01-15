"""
Test import verification for quality_validator library component.

This test ensures the quality_validator component is properly installed
and can be imported from the lib.validation package.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_import_quality_validator():
    """Test that QualityValidator can be imported"""
    from lib.validation import QualityValidator
    assert QualityValidator is not None


def test_import_all_exports():
    """Test that all exported classes can be imported"""
    from lib.validation import (
        QualityValidator,
        QualityClaim,
        QualityValidationResult,
        ValidationResult,
        Violation,
        AnalysisResult,
        Severity,
        EvidenceQuality,
        RiskLevel,
    )

    assert QualityValidator is not None
    assert QualityClaim is not None
    assert QualityValidationResult is not None
    assert ValidationResult is not None
    assert Violation is not None
    assert AnalysisResult is not None
    assert Severity is not None
    assert EvidenceQuality is not None
    assert RiskLevel is not None


def test_basic_validator_usage():
    """Test basic QualityValidator functionality"""
    from lib.validation import QualityValidator, Violation

    validator = QualityValidator()

    # Add a test violation
    violation = validator.add_violation(
        rule_id="TEST-001",
        message="Test violation",
        file="test.py",
        line=10,
        severity="medium",
    )

    assert isinstance(violation, Violation)
    assert violation.rule_id == "TEST-001"
    assert len(validator.violations) == 1

    # Check score calculation
    score = validator.calculate_score()
    assert score == 98.0  # Base 100 - 2 for medium

    # Check gate passes (threshold not exceeded)
    assert validator.check_gate(fail_on="high") is True


def test_severity_validation():
    """Test that invalid severity raises ValueError"""
    from lib.validation import QualityValidator
    import pytest

    validator = QualityValidator()

    # This should raise ValueError for invalid severity
    try:
        validator.add_violation(
            rule_id="TEST-002",
            message="Invalid severity test",
            file="test.py",
            line=20,
            severity="invalid_severity",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid severity" in str(e)


def test_analysis_result():
    """Test AnalysisResult generation"""
    from lib.validation import QualityValidator

    validator = QualityValidator()
    validator.add_violation(
        rule_id="GATE-001",
        message="Gate test",
        file="gate.py",
        line=5,
        severity="low",
    )

    result = validator.analyze(fail_on="high", project_path="test-project")

    assert result.quality_gate_passed is True
    assert result.overall_score == 99.0  # Base 100 - 1 for low
    assert len(result.violations) == 1
    assert "test-project" in result.metadata["project_path"]


if __name__ == "__main__":
    # Run basic tests
    test_import_quality_validator()
    print("test_import_quality_validator: PASSED")

    test_import_all_exports()
    print("test_import_all_exports: PASSED")

    test_basic_validator_usage()
    print("test_basic_validator_usage: PASSED")

    test_severity_validation()
    print("test_severity_validation: PASSED")

    test_analysis_result()
    print("test_analysis_result: PASSED")

    print("\nAll import verification tests passed!")
