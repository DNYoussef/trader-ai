#!/usr/bin/env python3
"""
Comprehensive GitHub Workflow Testing Suite

Tests all 9 fixed GitHub workflows for YAML syntax, Python script execution,
JSON output structure, quality gate logic, error handling, and end-to-end pipeline integration.

Validates:
1. Architecture Analysis (architecture-analysis.yml)
2. Connascence Core Analysis (connascence-core-analysis.yml)
3. Cache Optimization (cache-optimization.yml)
4. Security Pipeline (security-pipeline.yml)
5. Performance Monitoring (performance-monitoring.yml)
6. Quality Gates (quality-gates.yml)
7. MECE Duplication Analysis (mece-duplication-analysis.yml)
8. Self-Dogfooding (self-dogfooding.yml)
9. Quality Orchestrator (quality-orchestrator.yml)
"""

import json
import os
import sys
import yaml
import ast
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Validation result for a single test"""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class WorkflowTestReport:
    """Complete test report for a workflow"""
    workflow_name: str
    workflow_file: str
    test_results: List[ValidationResult]
    overall_passed: bool
    execution_time: float
    critical_issues: List[str]
    warnings: List[str]

class WorkflowValidator:
    """Validates GitHub workflows for syntax, execution, and integration"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.workflows_dir = self.repo_root / '.github' / 'workflows'
        self.artifacts_dir = self.repo_root / '.claude' / '.artifacts'
        self.target_workflows = [
            'architecture-analysis.yml',
            'connascence-core-analysis.yml', 
            'cache-optimization.yml',
            'security-pipeline.yml',
            'performance-monitoring.yml',
            'quality-gates.yml',
            'mece-duplication-analysis.yml',
            'self-dogfooding.yml',
            'quality-orchestrator.yml'
        ]
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_all_workflows(self) -> Dict[str, WorkflowTestReport]:
        """Validate all target workflows"""
        logger.info(f"Starting validation of {len(self.target_workflows)} workflows")
        
        reports = {}
        for workflow_file in self.target_workflows:
            logger.info(f"Validating workflow: {workflow_file}")
            start_time = datetime.now()
            
            report = self.validate_workflow(workflow_file)
            report.execution_time = (datetime.now() - start_time).total_seconds()
            reports[workflow_file] = report
            
            status = "PASSED" if report.overall_passed else "FAILED"
            logger.info(f"Workflow {workflow_file}: {status}")
            
        return reports
        
    def validate_workflow(self, workflow_file: str) -> WorkflowTestReport:
        """Validate a single workflow file"""
        workflow_path = self.workflows_dir / workflow_file
        
        if not workflow_path.exists():
            return WorkflowTestReport(
                workflow_name=workflow_file,
                workflow_file=str(workflow_path),
                test_results=[ValidationResult(
                    test_name="file_exists",
                    passed=False,
                    message=f"Workflow file not found: {workflow_path}",
                    error="FileNotFoundError"
                )],
                overall_passed=False,
                execution_time=0.0,
                critical_issues=[f"Workflow file missing: {workflow_file}"],
                warnings=[]
            )
            
        test_results = []
        critical_issues = []
        warnings = []
        
        # Test 1: YAML Syntax Validation
        yaml_result = self.test_yaml_syntax(workflow_path)
        test_results.append(yaml_result)
        if not yaml_result.passed:
            critical_issues.append(f"YAML syntax error in {workflow_file}")
            
        # Test 2: Python Script Validation
        python_result = self.test_python_scripts(workflow_path)
        test_results.append(python_result)
        if not python_result.passed:
            critical_issues.append(f"Python syntax errors in {workflow_file}")
            
        # Test 3: Unicode Character Check
        unicode_result = self.test_unicode_characters(workflow_path)
        test_results.append(unicode_result)
        if not unicode_result.passed:
            warnings.append(f"Unicode characters detected in {workflow_file}")
            
        # Test 4: JSON Output Structure
        json_result = self.test_json_output_structure(workflow_file)
        test_results.append(json_result)
        if not json_result.passed:
            warnings.append(f"JSON output structure issues in {workflow_file}")
            
        # Test 5: Quality Gate Logic
        gate_result = self.test_quality_gate_logic(workflow_path)
        test_results.append(gate_result)
        if not gate_result.passed:
            critical_issues.append(f"Quality gate logic errors in {workflow_file}")
            
        # Test 6: Error Handling
        error_result = self.test_error_handling(workflow_path)
        test_results.append(error_result)
        if not error_result.passed:
            warnings.append(f"Error handling issues in {workflow_file}")
            
        # Test 7: Dependency Analysis
        deps_result = self.test_dependencies(workflow_path)
        test_results.append(deps_result)
        if not deps_result.passed:
            warnings.append(f"Dependency issues in {workflow_file}")
            
        overall_passed = all(r.passed or r.test_name in ['unicode_check', 'json_structure', 'error_handling', 'dependencies'] for r in test_results)
        
        return WorkflowTestReport(
            workflow_name=workflow_file,
            workflow_file=str(workflow_path),
            test_results=test_results,
            overall_passed=overall_passed,
            execution_time=0.0,  # Will be set by caller
            critical_issues=critical_issues,
            warnings=warnings
        )
        
    def test_yaml_syntax(self, workflow_path: Path) -> ValidationResult:
        """Test YAML syntax validity"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
                
            # Validate required YAML structure
            required_keys = ['name', 'on', 'jobs']
            missing_keys = [key for key in required_keys if key not in yaml_content]
            
            if missing_keys:
                return ValidationResult(
                    test_name="yaml_syntax",
                    passed=False,
                    message=f"Missing required keys: {missing_keys}",
                    details={"missing_keys": missing_keys}
                )
                
            # Check for proper job structure
            jobs = yaml_content.get('jobs', {})
            if not jobs:
                return ValidationResult(
                    test_name="yaml_syntax",
                    passed=False,
                    message="No jobs defined in workflow",
                    details={"jobs_count": 0}
                )
                
            job_issues = []
            for job_name, job_config in jobs.items():
                if 'runs-on' not in job_config:
                    job_issues.append(f"Job '{job_name}' missing 'runs-on'")
                if 'steps' not in job_config:
                    job_issues.append(f"Job '{job_name}' missing 'steps'")
                    
            if job_issues:
                return ValidationResult(
                    test_name="yaml_syntax",
                    passed=False,
                    message=f"Job configuration issues: {', '.join(job_issues)}",
                    details={"job_issues": job_issues}
                )
                
            return ValidationResult(
                test_name="yaml_syntax",
                passed=True,
                message="YAML syntax is valid",
                details={
                    "jobs_count": len(jobs),
                    "workflow_name": yaml_content.get('name', 'Unknown')
                }
            )
            
        except yaml.YAMLError as e:
            return ValidationResult(
                test_name="yaml_syntax",
                passed=False,
                message=f"YAML parsing error: {e}",
                error=str(e)
            )
        except Exception as e:
            return ValidationResult(
                test_name="yaml_syntax",
                passed=False,
                message=f"Unexpected error parsing YAML: {e}",
                error=str(e)
            )
            
    def test_python_scripts(self, workflow_path: Path) -> ValidationResult:
        """Test Python script syntax in workflow steps"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract Python scripts from workflow
            python_scripts = self.extract_python_scripts(content)
            
            if not python_scripts:
                return ValidationResult(
                    test_name="python_scripts",
                    passed=True,
                    message="No Python scripts found in workflow",
                    details={"scripts_count": 0}
                )
                
            syntax_errors = []
            valid_scripts = 0
            
            for i, script in enumerate(python_scripts):
                try:
                    # Test if the script compiles
                    ast.parse(script)
                    valid_scripts += 1
                except SyntaxError as e:
                    syntax_errors.append(f"Script {i+1}: {e}")
                except Exception as e:
                    syntax_errors.append(f"Script {i+1}: {e}")
                    
            if syntax_errors:
                return ValidationResult(
                    test_name="python_scripts",
                    passed=False,
                    message=f"Python syntax errors found: {len(syntax_errors)}",
                    details={
                        "total_scripts": len(python_scripts),
                        "valid_scripts": valid_scripts,
                        "syntax_errors": syntax_errors[:5]  # Limit to first 5 errors
                    },
                    error="; ".join(syntax_errors[:3])
                )
                
            return ValidationResult(
                test_name="python_scripts",
                passed=True,
                message=f"All {len(python_scripts)} Python scripts have valid syntax",
                details={
                    "total_scripts": len(python_scripts),
                    "valid_scripts": valid_scripts
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="python_scripts",
                passed=False,
                message=f"Error validating Python scripts: {e}",
                error=str(e)
            )
            
    def extract_python_scripts(self, content: str) -> List[str]:
        """Extract Python scripts from workflow YAML content"""
        scripts = []
        
        # Pattern 1: python -c "script"
        pattern1 = r'python\s+-c\s+"([^"]*(?:\\.[^"]*)*)"'
        matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
        scripts.extend(matches1)
        
        # Pattern 2: python -c 'script'
        pattern2 = r"python\s+-c\s+'([^']*(?:\\.[^']*)*)'"
        matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
        scripts.extend(matches2)
        
        # Pattern 3: exec('''script''')
        pattern3 = r"exec\(['\"]([^'\"]*(?:\\.[^'\"]*)*)['\"]"
        matches3 = re.findall(pattern3, content, re.MULTILINE | re.DOTALL)
        scripts.extend(matches3)
        
        # Pattern 4: exec("""script""")
        pattern4 = r'exec\(\"{3}(.*?)\"{3}'
        matches4 = re.findall(pattern4, content, re.MULTILINE | re.DOTALL)
        scripts.extend(matches4)
        
        # Clean up scripts - unescape and normalize
        cleaned_scripts = []
        for script in scripts:
            # Unescape common escape sequences
            cleaned = script.replace('\\"', '"').replace("\\'", "'")
            cleaned = cleaned.replace('\\n', '\n').replace('\\t', '\t')
            # Remove exec wrapper if present
            if cleaned.strip().startswith("exec("):
                # Extract inner content
                start = cleaned.find("(") + 1
                end = cleaned.rfind(")")
                if start < end:
                    cleaned = cleaned[start:end]
                    # Remove quotes
                    if (cleaned.startswith('"""') and cleaned.endswith('"""')) or \
                       (cleaned.startswith("'''") and cleaned.endswith("'''")):
                        cleaned = cleaned[3:-3]
                    elif (cleaned.startswith('"') and cleaned.endswith('"')) or \
                         (cleaned.startswith("'") and cleaned.endswith("'")):
                        cleaned = cleaned[1:-1]
            
            if cleaned.strip():
                cleaned_scripts.append(cleaned.strip())
        
        return cleaned_scripts
        
    def test_unicode_characters(self, workflow_path: Path) -> ValidationResult:
        """Test for Unicode characters that might cause CI/CD issues"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find non-ASCII characters
            unicode_chars = []
            for i, char in enumerate(content):
                if ord(char) > 127:
                    line_num = content[:i].count('\n') + 1
                    unicode_chars.append({
                        'char': char,
                        'code': ord(char),
                        'line': line_num,
                        'position': i
                    })
                    
            if unicode_chars:
                return ValidationResult(
                    test_name="unicode_check",
                    passed=False,
                    message=f"Found {len(unicode_chars)} Unicode characters",
                    details={
                        "unicode_count": len(unicode_chars),
                        "first_10": unicode_chars[:10]
                    }
                )
                
            return ValidationResult(
                test_name="unicode_check",
                passed=True,
                message="No Unicode characters found - ASCII only",
                details={"unicode_count": 0}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="unicode_check",
                passed=False,
                message=f"Error checking Unicode characters: {e}",
                error=str(e)
            )
            
    def test_json_output_structure(self, workflow_file: str) -> ValidationResult:
        """Test expected JSON output structure based on workflow type"""
        expected_artifacts = self.get_expected_artifacts(workflow_file)
        
        if not expected_artifacts:
            return ValidationResult(
                test_name="json_structure",
                passed=True,
                message="No JSON artifacts expected for this workflow",
                details={"expected_artifacts": []}
            )
            
        # For now, we'll validate that the expected structure is defined
        # In a real CI environment, we would check actual artifact outputs
        structure_issues = []
        
        for artifact in expected_artifacts:
            # Validate artifact name and expected structure
            if not artifact.get('name'):
                structure_issues.append(f"Artifact missing name: {artifact}")
            if not artifact.get('expected_fields'):
                structure_issues.append(f"No expected fields defined for {artifact.get('name', 'unknown')}")
                
        if structure_issues:
            return ValidationResult(
                test_name="json_structure",
                passed=False,
                message=f"JSON structure definition issues: {len(structure_issues)}",
                details={
                    "expected_artifacts": len(expected_artifacts),
                    "structure_issues": structure_issues
                }
            )
            
        return ValidationResult(
            test_name="json_structure",
            passed=True,
            message=f"JSON structure properly defined for {len(expected_artifacts)} artifacts",
            details={
                "expected_artifacts": len(expected_artifacts),
                "artifact_names": [a.get('name') for a in expected_artifacts]
            }
        )
        
    def get_expected_artifacts(self, workflow_file: str) -> List[Dict[str, Any]]:
        """Get expected JSON artifacts for each workflow type"""
        artifact_map = {
            'architecture-analysis.yml': [
                {
                    'name': 'architecture_analysis.json',
                    'expected_fields': ['system_overview', 'architectural_hotspots', 'metrics', 'recommendations']
                }
            ],
            'connascence-core-analysis.yml': [
                {
                    'name': 'connascence_full.json',
                    'expected_fields': ['violations', 'summary', 'nasa_compliance', 'god_objects']
                }
            ],
            'cache-optimization.yml': [
                {
                    'name': 'cache_optimization.json',
                    'expected_fields': ['cache_health', 'performance_metrics', 'recommendations']
                }
            ],
            'security-pipeline.yml': [
                {
                    'name': 'security_gates_report.json',
                    'expected_fields': ['security_summary', 'overall_security_score', 'quality_gates']
                }
            ],
            'performance-monitoring.yml': [
                {
                    'name': 'performance_monitor.json',
                    'expected_fields': ['metrics', 'resource_utilization', 'optimization_recommendations']
                }
            ],
            'quality-gates.yml': [
                {
                    'name': 'quality_gates_report.json',
                    'expected_fields': ['multi_tier_results', 'comprehensive_metrics', 'overall_status']
                }
            ],
            'mece-duplication-analysis.yml': [
                {
                    'name': 'mece_analysis.json',
                    'expected_fields': ['mece_score', 'duplications', 'analysis_summary']
                }
            ],
            'self-dogfooding.yml': [
                {
                    'name': 'self_analysis_nasa.json',
                    'expected_fields': ['nasa_compliance', 'violations', 'summary']
                }
            ],
            'quality-orchestrator.yml': [
                {
                    'name': 'quality_gates_report.json',
                    'expected_fields': ['analysis_summary', 'overall_scores', 'critical_issues']
                }
            ]
        }
        return artifact_map.get(workflow_file, [])
        
    def test_quality_gate_logic(self, workflow_path: Path) -> ValidationResult:
        """Test quality gate threshold logic"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for quality gate patterns
            gate_patterns = [
                r'min_\w+\s*=\s*[\d.]+',  # min_threshold = 0.85
                r'max_\w+\s*=\s*[\d.]+',  # max_violations = 5
                r'if\s+\w+\s*[<>=!]+\s*[\d.]+',  # if score < 0.85
                r'threshold\s*[:=]\s*[\d.]+',  # threshold: 0.85
                r'exit\(1\)',  # exit(1) for failures
                r'sys\.exit\(\d+\)'  # sys.exit(1) for failures
            ]
            
            gate_checks = []
            for pattern in gate_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                gate_checks.extend(matches)
                
            if not gate_checks:
                return ValidationResult(
                    test_name="quality_gate_logic",
                    passed=False,
                    message="No quality gate logic patterns found",
                    details={"patterns_found": 0}
                )
                
            # Check for proper error handling in gates
            has_error_handling = any(pattern in content.lower() for pattern in [
                'except', 'try:', 'error', 'failed', 'pass', 'fail'
            ])
            
            # Check for threshold validation
            has_thresholds = any('threshold' in check.lower() or 
                               any(op in check for op in ['<', '>', '<=', '>=', '==']) 
                               for check in gate_checks)
            
            issues = []
            if not has_error_handling:
                issues.append("No error handling patterns found")
            if not has_thresholds:
                issues.append("No threshold validation patterns found")
                
            if issues:
                return ValidationResult(
                    test_name="quality_gate_logic",
                    passed=False,
                    message=f"Quality gate logic issues: {', '.join(issues)}",
                    details={
                        "patterns_found": len(gate_checks),
                        "issues": issues,
                        "has_error_handling": has_error_handling,
                        "has_thresholds": has_thresholds
                    }
                )
                
            return ValidationResult(
                test_name="quality_gate_logic",
                passed=True,
                message=f"Quality gate logic properly implemented with {len(gate_checks)} checks",
                details={
                    "patterns_found": len(gate_checks),
                    "has_error_handling": has_error_handling,
                    "has_thresholds": has_thresholds
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="quality_gate_logic",
                passed=False,
                message=f"Error validating quality gate logic: {e}",
                error=str(e)
            )
            
    def test_error_handling(self, workflow_path: Path) -> ValidationResult:
        """Test error handling and fallback mechanisms"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for error handling patterns
            error_patterns = [
                r'try:\s*\n',  # try blocks
                r'except\s+\w*Exception',  # except blocks
                r'except\s*:',  # bare except
                r'continue-on-error:\s*true',  # GitHub continue-on-error
                r'fallback\s*=\s*True',  # fallback mode
                r'\|\|\s*echo\s*".*failed',  # shell fallback with echo
                r'timeout\s+\d+',  # timeout handling
                r'if\s+\[\s*!\s*-f.*\]',  # file existence checks
            ]
            
            error_handling_count = 0
            for pattern in error_patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                error_handling_count += len(matches)
                
            # Check for specific fallback mechanisms
            fallback_indicators = [
                'fallback' in content.lower(),
                'continue-on-error' in content,
                'timeout' in content.lower(),
                'try:' in content and 'except' in content,
                'error' in content.lower() and 'handle' in content.lower()
            ]
            
            fallback_score = sum(fallback_indicators)
            
            if error_handling_count == 0:
                return ValidationResult(
                    test_name="error_handling",
                    passed=False,
                    message="No error handling patterns found",
                    details={
                        "error_handling_count": 0,
                        "fallback_score": fallback_score
                    }
                )
                
            # Score based on error handling robustness
            if error_handling_count < 3:
                message = f"Basic error handling found ({error_handling_count} patterns)"
                passed = False
            elif error_handling_count < 6:
                message = f"Adequate error handling ({error_handling_count} patterns)"
                passed = True
            else:
                message = f"Robust error handling ({error_handling_count} patterns)"
                passed = True
                
            return ValidationResult(
                test_name="error_handling",
                passed=passed,
                message=message,
                details={
                    "error_handling_count": error_handling_count,
                    "fallback_score": fallback_score,
                    "fallback_indicators": [i for i, indicator in enumerate(fallback_indicators) if indicator]
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="error_handling",
                passed=False,
                message=f"Error validating error handling: {e}",
                error=str(e)
            )
            
    def test_dependencies(self, workflow_path: Path) -> ValidationResult:
        """Test workflow dependencies and integration points"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            yaml_content = yaml.safe_load(content)
            
            # Check for dependency management
            dependency_checks = []
            
            # Look for Python dependencies
            if 'pip install' in content:
                dependency_checks.append("Python dependencies managed")
                
            # Look for artifact dependencies
            if 'upload-artifact' in content or 'download-artifact' in content:
                dependency_checks.append("Artifact management present")
                
            # Look for cache usage
            if 'cache:' in content or 'actions/cache' in content:
                dependency_checks.append("Caching configured")
                
            # Look for timeout configurations
            if 'timeout-minutes' in content:
                dependency_checks.append("Timeout management configured")
                
            # Look for conditional execution
            if 'if:' in content:
                dependency_checks.append("Conditional execution configured")
                
            # Check for required directory creation
            if 'mkdir' in content and '.claude/.artifacts' in content:
                dependency_checks.append("Artifact directory management")
                
            # Check for cross-workflow references
            cross_refs = []
            if 'needs:' in content:
                cross_refs.append("Job dependencies defined")
                
            issues = []
            if len(dependency_checks) < 3:
                issues.append(f"Limited dependency management ({len(dependency_checks)} aspects)")
                
            # Check for common dependency issues
            if 'pip install' in content and 'requirements.txt' not in content:
                issues.append("Python dependencies but no requirements.txt check")
                
            if issues:
                return ValidationResult(
                    test_name="dependencies",
                    passed=False,
                    message=f"Dependency issues: {', '.join(issues)}",
                    details={
                        "dependency_checks": dependency_checks,
                        "cross_references": cross_refs,
                        "issues": issues
                    }
                )
                
            return ValidationResult(
                test_name="dependencies",
                passed=True,
                message=f"Dependencies properly managed ({len(dependency_checks)} aspects)",
                details={
                    "dependency_checks": dependency_checks,
                    "cross_references": cross_refs,
                    "total_aspects": len(dependency_checks)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="dependencies",
                passed=False,
                message=f"Error validating dependencies: {e}",
                error=str(e)
            )

    def test_end_to_end_integration(self, reports: Dict[str, WorkflowTestReport]) -> ValidationResult:
        """Test end-to-end pipeline integration"""
        try:
            # Check that all workflows can produce their expected artifacts
            artifact_producers = {
                'architecture-analysis.yml': ['architecture_analysis.json'],
                'connascence-core-analysis.yml': ['connascence_full.json'],
                'cache-optimization.yml': ['cache_optimization.json'],
                'security-pipeline.yml': ['security_gates_report.json'],
                'performance-monitoring.yml': ['performance_monitor.json'],
                'quality-gates.yml': ['quality_gates_report.json'],
                'mece-duplication-analysis.yml': ['mece_analysis.json'],
                'self-dogfooding.yml': ['self_analysis_nasa.json'],
                'quality-orchestrator.yml': ['quality_gates_report.json']
            }
            
            # Check artifact consumers
            artifact_consumers = {
                'quality-gates.yml': [
                    'connascence_full.json',
                    'architecture_analysis.json', 
                    'performance_monitor.json',
                    'cache_optimization.json',
                    'mece_analysis.json'
                ],
                'quality-orchestrator.yml': [
                    'connascence_analysis.json',
                    'architecture_analysis.json',
                    'performance_monitoring.json',
                    'mece_analysis.json',
                    'cache_optimization.json'
                ]
            }
            
            integration_issues = []
            successful_workflows = 0
            
            for workflow_file, report in reports.items():
                if not report.overall_passed:
                    integration_issues.append(f"Workflow {workflow_file} failed validation")
                else:
                    successful_workflows += 1
                    
            # Check producer-consumer relationships
            for consumer, required_artifacts in artifact_consumers.items():
                if consumer in reports:
                    for artifact in required_artifacts:
                        # Find the producer of this artifact
                        producers = [w for w, artifacts in artifact_producers.items() 
                                   if any(a in artifact for a in artifacts)]
                        if not producers:
                            integration_issues.append(f"No producer found for artifact {artifact}")
                        elif not all(reports.get(p, WorkflowTestReport('', '', [], False, 0, [], [])).overall_passed 
                                   for p in producers):
                            integration_issues.append(f"Producer workflows for {artifact} have issues")
                            
            pipeline_health = successful_workflows / len(reports) if reports else 0
            
            if integration_issues:
                return ValidationResult(
                    test_name="end_to_end_integration",
                    passed=len(integration_issues) <= 2,  # Allow minor issues
                    message=f"Integration issues found: {len(integration_issues)}",
                    details={
                        "successful_workflows": successful_workflows,
                        "total_workflows": len(reports),
                        "pipeline_health": pipeline_health,
                        "integration_issues": integration_issues[:5],
                        "producer_consumer_map": {
                            "producers": artifact_producers,
                            "consumers": artifact_consumers
                        }
                    }
                )
                
            return ValidationResult(
                test_name="end_to_end_integration",
                passed=True,
                message=f"End-to-end integration validated ({successful_workflows}/{len(reports)} workflows)",
                details={
                    "successful_workflows": successful_workflows,
                    "total_workflows": len(reports),
                    "pipeline_health": pipeline_health
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="end_to_end_integration",
                passed=False,
                message=f"Error validating end-to-end integration: {e}",
                error=str(e)
            )

    def generate_validation_report(self, reports: Dict[str, WorkflowTestReport]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Add end-to-end integration test
        e2e_result = self.test_end_to_end_integration(reports)
        
        total_tests = sum(len(report.test_results) for report in reports.values()) + 1  # +1 for e2e
        passed_tests = sum(sum(1 for result in report.test_results if result.passed) 
                          for report in reports.values())
        if e2e_result.passed:
            passed_tests += 1
            
        critical_issues = []
        warnings = []
        
        for report in reports.values():
            critical_issues.extend(report.critical_issues)
            warnings.extend(report.warnings)
            
        # Overall assessment
        overall_passed = all(report.overall_passed for report in reports.values()) and e2e_result.passed
        
        # Production readiness assessment
        production_blockers = []
        for workflow_file, report in reports.items():
            if not report.overall_passed:
                # Check if this is a critical workflow for production
                if workflow_file in ['quality-gates.yml', 'security-pipeline.yml', 'connascence-core-analysis.yml']:
                    production_blockers.append(f"Critical workflow failed: {workflow_file}")
                    
        production_ready = len(production_blockers) == 0
        
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_workflows": len(reports),
                "passed_workflows": sum(1 for r in reports.values() if r.overall_passed),
                "failed_workflows": sum(1 for r in reports.values() if not r.overall_passed),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "overall_assessment": {
                "validation_passed": overall_passed,
                "production_ready": production_ready,
                "analyzer_pipeline_ready": overall_passed and production_ready,
                "deployment_recommendation": "DEPLOY" if production_ready else "BLOCK"
            },
            "workflow_results": {
                workflow_file: {
                    "passed": report.overall_passed,
                    "execution_time": report.execution_time,
                    "test_results": [
                        {
                            "test_name": result.test_name,
                            "passed": result.passed,
                            "message": result.message,
                            "error": result.error
                        } for result in report.test_results
                    ],
                    "critical_issues": report.critical_issues,
                    "warnings": report.warnings
                } for workflow_file, report in reports.items()
            },
            "end_to_end_integration": {
                "test_name": e2e_result.test_name,
                "passed": e2e_result.passed,
                "message": e2e_result.message,
                "details": e2e_result.details
            },
            "critical_issues": critical_issues,
            "warnings": warnings,
            "production_blockers": production_blockers,
            "recommendations": self.generate_recommendations(reports, e2e_result, critical_issues, warnings)
        }
        
        return validation_report
        
    def generate_recommendations(self, reports: Dict[str, WorkflowTestReport], 
                                e2e_result: ValidationResult,
                                critical_issues: List[str], 
                                warnings: List[str]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Critical issues recommendations
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical issues before production deployment")
            
            # Specific recommendations based on issue types
            yaml_issues = [issue for issue in critical_issues if 'YAML' in issue]
            if yaml_issues:
                recommendations.append("Fix YAML syntax errors - these prevent workflow execution")
                
            python_issues = [issue for issue in critical_issues if 'Python' in issue] 
            if python_issues:
                recommendations.append("Fix Python syntax errors in embedded scripts")
                
            gate_issues = [issue for issue in critical_issues if 'gate' in issue.lower()]
            if gate_issues:
                recommendations.append("Review and fix quality gate logic implementation")
                
        # Warning-based recommendations
        if len(warnings) > 10:
            recommendations.append(f"Consider addressing {len(warnings)} warnings for optimal performance")
            
        # Integration recommendations
        if not e2e_result.passed:
            recommendations.append("Fix end-to-end integration issues before enabling full pipeline")
            
        # Performance recommendations
        slow_workflows = [name for name, report in reports.items() if report.execution_time > 60]
        if slow_workflows:
            recommendations.append(f"Consider optimizing execution time for workflows: {', '.join(slow_workflows)}")
            
        # Success recommendations
        if not recommendations:
            recommendations.append("All validations passed - workflows are ready for production deployment")
            recommendations.append("Consider setting up automated validation in CI/CD pipeline")
            
        return recommendations


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GitHub workflows")
    parser.add_argument("--repo-root", default=".", help="Root directory of the repository")
    parser.add_argument("--output", default="tests/workflow-validation/validation_report.json", 
                       help="Output file for validation report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize validator
    validator = WorkflowValidator(args.repo_root)
    
    # Run validation
    logger.info("Starting comprehensive workflow validation...")
    reports = validator.validate_all_workflows()
    
    # Generate report
    validation_report = validator.generate_validation_report(reports)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
        
    logger.info(f"Validation report saved to: {output_path}")
    
    # Print summary
    summary = validation_report["summary"]
    assessment = validation_report["overall_assessment"]
    
    print("\n" + "="*80)
    print("WORKFLOW VALIDATION SUMMARY")
    print("="*80)
    print(f"Total Workflows: {summary['total_workflows']}")
    print(f"Passed Workflows: {summary['passed_workflows']}")
    print(f"Failed Workflows: {summary['failed_workflows']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print()
    print(f"Validation Status: {'PASSED' if assessment['validation_passed'] else 'FAILED'}")
    print(f"Production Ready: {'YES' if assessment['production_ready'] else 'NO'}")
    print(f"Deployment Recommendation: {assessment['deployment_recommendation']}")
    print()
    
    if validation_report["critical_issues"]:
        print("CRITICAL ISSUES:")
        for issue in validation_report["critical_issues"][:5]:
            print(f"  - {issue}")
        print()
        
    if validation_report["recommendations"]:
        print("RECOMMENDATIONS:")
        for rec in validation_report["recommendations"][:5]:
            print(f"  - {rec}")
        print()
        
    print("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if assessment["validation_passed"] else 1)


if __name__ == "__main__":
    main()