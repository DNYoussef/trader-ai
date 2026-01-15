"""
Validation library components.

Contains quality validation, spec validation, and gate logic components.
"""

from .quality_validator import (
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

from .spec_validation import (
    SpecValidator,
    SpecValidationResult,
    ValidationSchema,
    BaseValidator,
    PrereqsValidator,
    JSONFileValidator,
    ContextValidator,
    MarkdownDocumentValidator,
    SpecDocumentValidator,
    ImplementationPlanValidator,
    validate_spec_directory,
    create_validator_from_config,
)

__all__ = [
    # Quality validation
    "QualityValidator",
    "QualityClaim",
    "QualityValidationResult",
    "ValidationResult",
    "Violation",
    "AnalysisResult",
    "Severity",
    "EvidenceQuality",
    "RiskLevel",
    # Spec validation
    "SpecValidator",
    "SpecValidationResult",
    "ValidationSchema",
    "BaseValidator",
    "PrereqsValidator",
    "JSONFileValidator",
    "ContextValidator",
    "MarkdownDocumentValidator",
    "SpecDocumentValidator",
    "ImplementationPlanValidator",
    "validate_spec_directory",
    "create_validator_from_config",
]
