# Connascence Scan Summary

- Project: `trader-ai`
- Path: `D:\Projects\trader-ai`
- Git branch: `main`
- Git commit: `06c5fddd25ef7795f84504f9ea1a5e06df223401`
- Dirty before scan: `True`
- Scan succeeded: `True`
- Python files staged: `654`

## Commands Run
- `C:\Python312\python.exe -m analyzer C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\trader-ai\mirror --format json --output C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\trader-ai\connascence.raw.json --no-duplication --compliance-threshold 0 --max-god-objects 999999` (exit 0)
- `connascence_portfolio_runner.py generate-sarif-from-json D:\Projects\trader-ai\docs\connascence\scan-2026-06-06\connascence.json` (exit 0)
- `C:\Python312\python.exe -m analyzer.ast_engine --path C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\trader-ai\mirror --analyzer god_object --output C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\trader-ai\god-object.raw.json` (exit 0)

## Counts By Severity

- low: 44034
- medium: 4467
- critical: 245
- high: 154

## Counts By Type

- connascence_of_meaning: 39415
- CoV: 5235
- connascence_of_convention: 1289
- connascence_of_type: 844
- CoP: 707
- connascence_of_execution: 595
- connascence_of_algorithm: 335
- god_object: 236
- connascence_of_timing: 164
- CoA: 80

## Top Files

- `D:\Projects\trader-ai\src\theater-detection\theater-detector.py`: 696
- `D:\Projects\trader-ai\src\dashboard\run_server_simple.py`: 677
- `D:\Projects\trader-ai\src\data\enhanced_feature_extractor.py`: 603
- `D:\Projects\trader-ai\scripts\data\generate_110_feature_labels.py`: 547
- `D:\Projects\trader-ai\src\security\defense_industry_evidence_generator.py`: 542
- `D:\Projects\trader-ai\src\security\dfars_compliance_certification.py`: 530
- `D:\Projects\trader-ai\src\performance\reporting\PerformanceReporter.py`: 507
- `D:\Projects\trader-ai\src\theater-detection\reality-validator.py`: 485
- `D:\Projects\trader-ai\scripts\hrm_grokfast_trader.py`: 468
- `D:\Projects\trader-ai\src\security\continuous_risk_assessment.py`: 418

## Top 10 Actionable Findings

1. `D:\Projects\trader-ai\src\utils\money.py:37` - Class 'Money' is a God Object: 32 methods, ~224 lines
2. `D:\Projects\trader-ai\src\user_experience\value_screens.py:40` - Class 'ValueScreenGenerator' is a God Object (config context): Very low cohesion (0.14)
3. `D:\Projects\trader-ai\src\user_experience\reinforcement_integration.py:32` - Class 'ReinforcementOrchestrator' is a God Object: 22 methods, ~475 lines
4. `D:\Projects\trader-ai\src\user_experience\psychological_reinforcement.py:97` - Class 'PsychologicalReinforcementEngine' is a God Object (config context): Very low cohesion (0.14)
5. `D:\Projects\trader-ai\src\user_experience\onboarding_flow.py:79` - Class 'TradingOnboardingFlow' is a God Object (config context): Very low cohesion (0.12)
6. `D:\Projects\trader-ai\src\user_experience\causal_education.py:77` - Class 'CausalEducationEngine' is a God Object (config context): Very low cohesion (0.10)
7. `D:\Projects\trader-ai\src\training\trm_trainer.py:156` - Class 'TRMTrainer' is a God Object (config context): Very low cohesion (0.18)
8. `D:\Projects\trader-ai\src\training\trm_loss_functions.py:180` - Class 'TRMLoss' is a God Object (unknown context): Very low cohesion (0.25)
9. `D:\Projects\trader-ai\src\training\trm_data_loader.py:28` - Class 'TRMDataset' is a God Object (data_model context): Very low cohesion (0.23)
10. `D:\Projects\trader-ai\src\training\trm_data_loader.py:143` - Class 'TRMDataModule' is a God Object (config context): Very low cohesion (0.12)

## Tool Limitations

- Connascence currently analyzes Python files only; non-Python coupling is not covered.
- Source-bearing fields and literal values were stripped or redacted before writing artifacts.
- Excluded directories and sensitive data patterns were not staged into the scan mirror.

## Next Cleanup Recommendations

### 1. Quick Wins
- Add type annotations at public function boundaries with the highest CoT counts.
- Replace repeated or magic literals with named constants or configuration keys.

### 2. Medium Refactors
- Convert high-parameter functions to keyword-only APIs or parameter objects.
- Split complex functions and consolidate duplicated algorithmic branches.
- Start with the top files by violation count and keep each change behavior-preserving.

### 3. Large Architectural Work
- Split god objects into cohesive classes around stable domain responsibilities.
- Use module or service boundaries to isolate recurring high-count hotspots.
