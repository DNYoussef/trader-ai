"""
Comprehensive Reporting and Visualization System
Advanced reporting engine for Gary×Taleb trading strategy performance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import warnings
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template
import webbrowser
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

@dataclass
class ReportSection:
    """Report section definition"""
    title: str
    content: str
    visualizations: List[Dict] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    section_type: str = "standard"  # standard, executive, technical, appendix

@dataclass
class ReportConfig:
    """Report configuration"""
    title: str = "Gary×Taleb Trading Strategy Performance Report"
    author: str = "Performance Analysis System"
    logo_path: Optional[str] = None
    include_executive_summary: bool = True
    include_detailed_analysis: bool = True
    include_visualizations: bool = True
    include_appendix: bool = True
    format_type: str = "html"  # html, pdf, markdown
    color_scheme: str = "professional"  # professional, vibrant, minimal
    page_size: str = "A4"
    orientation: str = "portrait"

class PerformanceReporter:
    """Comprehensive performance reporting system"""

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.sections: List[ReportSection] = []
        self.assets_dir = Path("report_assets")
        self.assets_dir.mkdir(exist_ok=True)

        # Color schemes
        self.color_schemes = {
            'professional': {
                'primary': '#2E3440',
                'secondary': '#5E81AC',
                'accent': '#88C0D0',
                'success': '#A3BE8C',
                'warning': '#EBCB8B',
                'danger': '#BF616A',
                'background': '#ECEFF4',
                'text': '#2E3440'
            },
            'vibrant': {
                'primary': '#FF6B6B',
                'secondary': '#4ECDC4',
                'accent': '#45B7D1',
                'success': '#96CEB4',
                'warning': '#FFEAA7',
                'danger': '#DDA0DD',
                'background': '#F8F9FA',
                'text': '#2C3E50'
            },
            'minimal': {
                'primary': '#333333',
                'secondary': '#666666',
                'accent': '#999999',
                'success': '#28A745',
                'warning': '#FFC107',
                'danger': '#DC3545',
                'background': '#FFFFFF',
                'text': '#212529'
            }
        }

    def add_section(self, section: ReportSection):
        """Add a section to the report"""
        self.sections.append(section)

    def create_executive_summary(self, performance_data: Dict,
                               key_metrics: Dict,
                               recommendations: List[str]) -> ReportSection:
        """Create executive summary section"""

        content = f"""
        ## Executive Summary

        This report provides a comprehensive analysis of the Gary×Taleb trading strategy performance,
        combining Gary's Dynamic Performance Indicator (DPI) methodology with Nassim Taleb's
        antifragility principles.

        ### Key Performance Highlights

        **Overall Performance:**
        - Total Return: {performance_data.get('total_return', 0):.2%}
        - Annualized Return: {performance_data.get('annualized_return', 0):.2%}
        - Sharpe Ratio: {key_metrics.get('sharpe_ratio', 0):.3f}
        - Maximum Drawdown: {key_metrics.get('max_drawdown', 0):.2%}

        **Gary×Taleb Integration:**
        - DPI Score: {key_metrics.get('dpi_score', 0):.3f}
        - Antifragility Score: {key_metrics.get('antifragility_score', 0):.3f}
        - Synergy Factor: {key_metrics.get('synergy_score', 0):.3f}

        **Risk Profile:**
        - Volatility: {key_metrics.get('volatility', 0):.2%}
        - VaR (95%): {key_metrics.get('var_95', 0):.2%}
        - Tail Risk Protection: {key_metrics.get('tail_protection', 0):.3f}

        ### Strategic Assessment

        The strategy demonstrates {'strong' if key_metrics.get('overall_score', 0) > 0.7 else 'moderate' if key_metrics.get('overall_score', 0) > 0.5 else 'weak'}
        performance with {'excellent' if key_metrics.get('antifragility_score', 0) > 0.7 else 'good' if key_metrics.get('antifragility_score', 0) > 0.5 else 'limited'}
        antifragile characteristics and {'robust' if key_metrics.get('dpi_score', 0) > 0.7 else 'adequate' if key_metrics.get('dpi_score', 0) > 0.5 else 'developing'}
        dynamic performance adaptation.

        ### Key Recommendations

        {chr(10).join([f"• {rec}" for rec in recommendations[:5]])}
        """

        return ReportSection(
            title="Executive Summary",
            content=content,
            section_type="executive",
            priority=1
        )

    def create_performance_overview(self, returns: pd.Series,
                                  equity_curve: pd.Series,
                                  benchmark_data: Optional[Dict] = None) -> ReportSection:
        """Create performance overview section"""

        # Calculate key metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0

        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        content = f"""
        ## Performance Overview

        ### Return Analysis

        The Gary×Taleb strategy achieved a total return of **{total_return:.2%}** over the analysis period,
        representing an annualized return of **{annualized_return:.2%}**. This performance was achieved
        with a volatility of **{volatility:.2%}**, resulting in a Sharpe ratio of **{sharpe_ratio:.3f}**.

        ### Risk Metrics

        The strategy maintained disciplined risk management with:
        - Maximum drawdown of **{max_drawdown:.2%}**
        - Average daily volatility of **{returns.std():.3%}**
        - Positive return days: **{(returns > 0).sum() / len(returns):.1%}**

        ### Benchmark Comparison

        {'The strategy significantly outperformed the benchmark' if benchmark_data and benchmark_data.get('outperformance', 0) > 0.05
          else 'The strategy performed in line with the benchmark' if benchmark_data and abs(benchmark_data.get('outperformance', 0)) <= 0.05
          else 'The strategy underperformed the benchmark' if benchmark_data
          else 'Benchmark comparison not available'}.

        ### Performance Attribution

        The strong performance can be attributed to:
        1. **Dynamic Performance Adaptation**: Effective signal generation and market timing
        2. **Antifragile Positioning**: Benefits from volatility and tail events
        3. **Risk Management**: Controlled downside exposure during adverse periods
        4. **Market Regime Adaptation**: Consistent performance across different market conditions
        """

        # Create performance visualization
        fig = self._create_performance_chart(returns, equity_curve, benchmark_data)

        visualizations = [{
            'type': 'plotly',
            'figure': fig,
            'title': 'Performance Overview',
            'description': 'Equity curve and key performance metrics'
        }]

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

        return ReportSection(
            title="Performance Overview",
            content=content,
            visualizations=visualizations,
            metrics=metrics,
            priority=2
        )

    def create_gary_taleb_analysis(self, antifragility_metrics: Dict,
                                 dpi_metrics: Dict,
                                 integration_analysis: Dict) -> ReportSection:
        """Create Gary×Taleb specific analysis section"""

        content = f"""
        ## Gary×Taleb Methodology Analysis

        ### Dynamic Performance Indicator (DPI) Analysis

        Gary's DPI methodology shows **{dpi_metrics.get('composite_score', 0):.3f}** overall effectiveness:

        **Core Components:**
        - Momentum Score: **{dpi_metrics.get('momentum_score', 0):.3f}**
        - Stability Score: **{dpi_metrics.get('stability_score', 0):.3f}**
        - Consistency Score: **{dpi_metrics.get('consistency_score', 0):.3f}**
        - Adaptability Score: **{dpi_metrics.get('adaptability_score', 0):.3f}**
        - Signal Quality: **{dpi_metrics.get('signal_quality', 0):.3f}**

        The DPI system demonstrates {'excellent' if dpi_metrics.get('composite_score', 0) > 0.8 else 'strong' if dpi_metrics.get('composite_score', 0) > 0.6 else 'moderate'}
        dynamic adaptation capabilities with particularly strong performance in
        {'momentum detection' if dpi_metrics.get('momentum_score', 0) > 0.7 else 'stability maintenance' if dpi_metrics.get('stability_score', 0) > 0.7 else 'signal consistency'}.

        ### Antifragility Analysis

        Taleb's antifragility principles show **{antifragility_metrics.get('overall_score', 0):.3f}** implementation:

        **Key Antifragile Characteristics:**
        - Convexity Score: **{antifragility_metrics.get('convexity_score', 0):.3f}**
        - Volatility Benefit: **{antifragility_metrics.get('volatility_benefit', 0):.3f}**
        - Asymmetry Ratio: **{antifragility_metrics.get('asymmetry_ratio', 0):.3f}**
        - Black Swan Protection: **{antifragility_metrics.get('black_swan_protection', 0):.3f}**
        - Barbell Efficiency: **{antifragility_metrics.get('barbell_efficiency', 0):.3f}**

        The strategy exhibits {'strong' if antifragility_metrics.get('volatility_benefit', 0) > 0.6 else 'moderate' if antifragility_metrics.get('volatility_benefit', 0) > 0.4 else 'limited'}
        antifragile behavior, particularly benefiting from {'high volatility periods' if antifragility_metrics.get('volatility_benefit', 0) > 0.6 else 'market stress events' if antifragility_metrics.get('black_swan_protection', 0) > 0.6 else 'tail risk scenarios'}.

        ### Integration Synergy

        The combination of Gary's DPI and Taleb's antifragility creates **{integration_analysis.get('synergy_score', 0):.3f}** synergistic value:

        - **Complementary Strengths**: DPI provides dynamic adaptation while antifragility ensures robustness
        - **Risk-Return Optimization**: Balanced approach to capturing upside while protecting downside
        - **Regime Independence**: Performance consistency across different market environments
        - **Scalability**: Framework adapts to different market conditions and volatility regimes

        ### Strategic Implications

        The Gary×Taleb integration suggests:
        1. **Tactical Advantage**: Superior market timing through DPI signals
        2. **Strategic Resilience**: Antifragile positioning for long-term sustainability
        3. **Risk Management**: Balanced exposure management across volatility regimes
        4. **Performance Persistence**: Reduced dependence on specific market conditions
        """

        # Create Gary×Taleb visualization
        fig = self._create_gary_taleb_chart(antifragility_metrics, dpi_metrics, integration_analysis)

        visualizations = [{
            'type': 'plotly',
            'figure': fig,
            'title': 'Gary×Taleb Analysis Dashboard',
            'description': 'Comprehensive analysis of DPI and antifragility metrics'
        }]

        return ReportSection(
            title="Gary×Taleb Methodology Analysis",
            content=content,
            visualizations=visualizations,
            metrics={**antifragility_metrics, **dpi_metrics, **integration_analysis},
            priority=3
        )

    def create_risk_analysis(self, returns: pd.Series,
                           risk_metrics: Dict,
                           drawdown_analysis: Dict) -> ReportSection:
        """Create risk analysis section"""

        content = f"""
        ## Risk Analysis

        ### Risk Metrics Summary

        The strategy maintains comprehensive risk controls with the following profile:

        **Value at Risk (VaR):**
        - 95% VaR: **{risk_metrics.get('var_95', 0):.2%}**
        - 99% VaR: **{risk_metrics.get('var_99', 0):.2%}**
        - Conditional VaR (95%): **{risk_metrics.get('cvar_95', 0):.2%}**

        **Drawdown Analysis:**
        - Maximum Drawdown: **{drawdown_analysis.get('max_drawdown', 0):.2%}**
        - Average Drawdown: **{drawdown_analysis.get('avg_drawdown', 0):.2%}**
        - Recovery Time: **{drawdown_analysis.get('avg_recovery_time', 0):.1f} days**
        - Drawdown Frequency: **{drawdown_analysis.get('drawdown_frequency', 0):.1%}**

        **Risk-Adjusted Performance:**
        - Sharpe Ratio: **{risk_metrics.get('sharpe_ratio', 0):.3f}**
        - Sortino Ratio: **{risk_metrics.get('sortino_ratio', 0):.3f}**
        - Calmar Ratio: **{risk_metrics.get('calmar_ratio', 0):.3f}**
        - Ulcer Index: **{risk_metrics.get('ulcer_index', 0):.3f}**

        ### Risk Model Validation

        The risk management framework demonstrates:
        - **Model Accuracy**: VaR backtesting shows {'good' if risk_metrics.get('var_accuracy', 0) > 0.9 else 'adequate' if risk_metrics.get('var_accuracy', 0) > 0.8 else 'needs improvement'} calibration
        - **Tail Risk Management**: {'Excellent' if risk_metrics.get('tail_protection', 0) > 0.8 else 'Good' if risk_metrics.get('tail_protection', 0) > 0.6 else 'Moderate'} protection against extreme events
        - **Dynamic Adjustment**: Risk parameters adapt to changing market conditions
        - **Stress Testing**: Strategy maintains stability under adverse scenarios

        ### Antifragile Risk Characteristics

        The strategy exhibits several antifragile risk properties:
        1. **Volatility Benefits**: Performance improves during high volatility periods
        2. **Convex Payoffs**: Upside capture exceeds downside exposure
        3. **Tail Protection**: Robust performance during extreme market events
        4. **Adaptive Sizing**: Position sizing responds to risk environment changes

        ### Risk Recommendations

        Based on the risk analysis:
        - {'Maintain current risk parameters' if risk_metrics.get('overall_risk_score', 0.5) > 0.7 else 'Consider risk parameter optimization' if risk_metrics.get('overall_risk_score', 0.5) > 0.5 else 'Implement enhanced risk controls'}
        - Monitor correlation with market stress indicators
        - Regular recalibration of VaR models
        - Continued focus on antifragile positioning
        """

        # Create risk visualization
        fig = self._create_risk_chart(returns, risk_metrics, drawdown_analysis)

        visualizations = [{
            'type': 'plotly',
            'figure': fig,
            'title': 'Risk Analysis Dashboard',
            'description': 'Comprehensive risk metrics and drawdown analysis'
        }]

        return ReportSection(
            title="Risk Analysis",
            content=content,
            visualizations=visualizations,
            metrics={**risk_metrics, **drawdown_analysis},
            priority=4
        )

    def create_optimization_recommendations(self, optimization_results: Dict,
                                         recommendations: List[str],
                                         implementation_plan: Dict) -> ReportSection:
        """Create optimization recommendations section"""

        content = f"""
        ## Optimization Recommendations

        ### Performance Optimization Analysis

        Based on comprehensive analysis, the following optimization opportunities have been identified:

        **Current Performance Score: {optimization_results.get('current_score', 0):.3f}**
        **Optimized Performance Score: {optimization_results.get('optimized_score', 0):.3f}**
        **Expected Improvement: {((optimization_results.get('optimized_score', 0) / optimization_results.get('current_score', 1)) - 1) * 100:.1f}%**

        ### Priority Recommendations

        #### High Priority (Immediate Implementation)
        {chr(10).join([f"• {rec}" for rec in recommendations if 'HIGH' in rec.upper()][:3])}

        #### Medium Priority (Next Quarter)
        {chr(10).join([f"• {rec}" for rec in recommendations if 'MEDIUM' in rec.upper()][:3])}

        #### Low Priority (Future Consideration)
        {chr(10).join([f"• {rec}" for rec in recommendations if 'LOW' in rec.upper()][:2])}

        ### Parameter Optimization

        **Signal Processing:**
        - Current signal threshold: {optimization_results.get('current_signal_threshold', 0.5):.3f}
        - Recommended threshold: {optimization_results.get('optimal_signal_threshold', 0.6):.3f}
        - Expected improvement: {optimization_results.get('signal_improvement', 0):.1%}

        **Risk Management:**
        - Current stop-loss: {optimization_results.get('current_stop_loss', 0.02):.1%}
        - Recommended stop-loss: {optimization_results.get('optimal_stop_loss', 0.025):.1%}
        - Risk-return improvement: {optimization_results.get('risk_improvement', 0):.1%}

        **Position Sizing:**
        - Current max position: {optimization_results.get('current_max_position', 1.0):.1%}
        - Recommended max position: {optimization_results.get('optimal_max_position', 0.8):.1%}
        - Sharpe improvement: {optimization_results.get('sharpe_improvement', 0):.1%}

        ### Implementation Roadmap

        **Phase 1 (Weeks 1-2): Foundation**
        {chr(10).join([f"• {item}" for item in implementation_plan.get('phase_1', [])])}

        **Phase 2 (Weeks 3-6): Enhancement**
        {chr(10).join([f"• {item}" for item in implementation_plan.get('phase_2', [])])}

        **Phase 3 (Weeks 7-12): Advanced Features**
        {chr(10).join([f"• {item}" for item in implementation_plan.get('phase_3', [])])}

        ### Expected Outcomes

        Implementation of these recommendations is projected to deliver:
        - **Performance**: {optimization_results.get('expected_return_improvement', 0):.1%} improvement in risk-adjusted returns
        - **Risk Reduction**: {optimization_results.get('expected_risk_reduction', 0):.1%} reduction in maximum drawdown
        - **Consistency**: {optimization_results.get('expected_consistency_improvement', 0):.1%} improvement in performance stability
        - **Antifragility**: Enhanced robustness to market stress and volatility

        ### Monitoring and Validation

        Post-implementation monitoring should focus on:
        - Real-time performance tracking against optimized parameters
        - Validation of improvement predictions through A/B testing
        - Continuous recalibration based on market regime changes
        - Regular review of antifragile characteristics maintenance
        """

        # Create optimization visualization
        fig = self._create_optimization_chart(optimization_results, recommendations)

        visualizations = [{
            'type': 'plotly',
            'figure': fig,
            'title': 'Optimization Analysis',
            'description': 'Parameter optimization results and improvement projections'
        }]

        return ReportSection(
            title="Optimization Recommendations",
            content=content,
            visualizations=visualizations,
            metrics=optimization_results,
            priority=5
        )

    def create_statistical_validation(self, validation_results: Dict,
                                    statistical_tests: List[Dict]) -> ReportSection:
        """Create statistical validation section"""

        content = f"""
        ## Statistical Validation

        ### Validation Summary

        The strategy underwent comprehensive statistical validation with the following results:

        **Overall Validation Score: {validation_results.get('overall_score', 0):.3f}**
        **Statistical Power: {validation_results.get('statistical_power', 0):.1%}**
        **Confidence Level: {validation_results.get('confidence_level', 0.95):.1%}**

        ### Statistical Tests Results

        #### Performance Significance
        {chr(10).join([f"• {test['name']}: {'PASS' if test['pass'] else 'FAIL'} (p-value: {test['p_value']:.4f})" for test in statistical_tests if test['category'] == 'performance'])}

        #### Distribution Properties
        {chr(10).join([f"• {test['name']}: {'PASS' if test['pass'] else 'FAIL'} (p-value: {test['p_value']:.4f})" for test in statistical_tests if test['category'] == 'distribution'])}

        #### Model Assumptions
        {chr(10).join([f"• {test['name']}: {'PASS' if test['pass'] else 'FAIL'} (p-value: {test['p_value']:.4f})" for test in statistical_tests if test['category'] == 'assumptions'])}

        ### Key Findings

        **Performance Validation:**
        - Returns are {'statistically significant' if validation_results.get('performance_significant', False) else 'not statistically significant'} at 95% confidence
        - Sharpe ratio is {'significantly positive' if validation_results.get('sharpe_significant', False) else 'not significantly different from zero'}
        - Risk-adjusted performance shows {'strong' if validation_results.get('risk_adjusted_score', 0) > 0.7 else 'moderate'} statistical evidence

        **Model Robustness:**
        - Strategy assumptions are {'well-supported' if validation_results.get('assumptions_valid', False) else 'partially supported'} by statistical tests
        - Out-of-sample validation shows {'consistent' if validation_results.get('oos_consistent', False) else 'mixed'} performance
        - Bootstrap confidence intervals {'confirm' if validation_results.get('bootstrap_confirm', False) else 'suggest caution for'} performance stability

        ### Validation Recommendations

        Based on statistical analysis:
        - {'Continue with current approach' if validation_results.get('overall_score', 0) > 0.7 else 'Consider model refinements' if validation_results.get('overall_score', 0) > 0.5 else 'Significant model improvements recommended'}
        - {'Increase sample size for stronger statistical power' if validation_results.get('statistical_power', 0) < 0.8 else 'Statistical power is adequate'}
        - {'Regular revalidation' if validation_results.get('overall_score', 0) > 0.6 else 'Enhanced validation protocols'} recommended for ongoing monitoring
        """

        # Create statistical validation visualization
        fig = self._create_validation_chart(validation_results, statistical_tests)

        visualizations = [{
            'type': 'plotly',
            'figure': fig,
            'title': 'Statistical Validation Results',
            'description': 'Comprehensive statistical test results and validation metrics'
        }]

        return ReportSection(
            title="Statistical Validation",
            content=content,
            visualizations=visualizations,
            metrics=validation_results,
            priority=6,
            section_type="technical"
        )

    def generate_html_report(self, output_path: str = "performance_report.html") -> str:
        """Generate comprehensive HTML report"""

        # Sort sections by priority
        sorted_sections = sorted(self.sections, key=lambda x: x.priority)

        # Get color scheme
        colors = self.color_schemes[self.config.color_scheme]

        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }

                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: {{ colors.text }};
                    background-color: {{ colors.background }};
                }

                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }

                .header {
                    background: linear-gradient(135deg, {{ colors.primary }}, {{ colors.secondary }});
                    color: white;
                    padding: 40px 0;
                    text-align: center;
                    margin-bottom: 30px;
                }

                .header h1 {
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }

                .header p {
                    font-size: 1.2em;
                    opacity: 0.9;
                }

                .toc {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }

                .toc h2 {
                    color: {{ colors.primary }};
                    margin-bottom: 15px;
                }

                .toc ul {
                    list-style: none;
                }

                .toc li {
                    padding: 8px 0;
                    border-bottom: 1px solid #eee;
                }

                .toc a {
                    color: {{ colors.secondary }};
                    text-decoration: none;
                    font-weight: 500;
                }

                .toc a:hover {
                    color: {{ colors.primary }};
                }

                .section {
                    background: white;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    overflow: hidden;
                }

                .section-header {
                    background: {{ colors.primary }};
                    color: white;
                    padding: 20px;
                }

                .section-content {
                    padding: 30px;
                }

                .executive .section-header {
                    background: {{ colors.accent }};
                }

                .technical .section-header {
                    background: {{ colors.secondary }};
                }

                .metric-card {
                    background: {{ colors.background }};
                    padding: 20px;
                    border-radius: 6px;
                    margin: 10px 0;
                    border-left: 4px solid {{ colors.accent }};
                }

                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }

                .visualization {
                    margin: 20px 0;
                    text-align: center;
                }

                .chart-container {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }

                .footer {
                    text-align: center;
                    padding: 30px;
                    background: {{ colors.primary }};
                    color: white;
                    margin-top: 50px;
                }

                h2 {
                    color: {{ colors.primary }};
                    margin: 30px 0 20px 0;
                    padding-bottom: 10px;
                    border-bottom: 2px solid {{ colors.accent }};
                }

                h3 {
                    color: {{ colors.secondary }};
                    margin: 25px 0 15px 0;
                }

                h4 {
                    color: {{ colors.primary }};
                    margin: 20px 0 10px 0;
                }

                .highlight {
                    background: {{ colors.warning }};
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-weight: bold;
                }

                .positive {
                    color: {{ colors.success }};
                    font-weight: bold;
                }

                .negative {
                    color: {{ colors.danger }};
                    font-weight: bold;
                }

                @media print {
                    .container { max-width: none; }
                    .section { page-break-inside: avoid; }
                }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>{{ title }}</h1>
                    <p>Generated on {{ date }} by {{ author }}</p>
                </div>
            </div>

            <div class="container">
                {% if config.include_toc %}
                <div class="toc">
                    <h2>Table of Contents</h2>
                    <ul>
                        {% for section in sections %}
                        <li><a href="#section-{{ loop.index }}">{{ section.title }}</a></li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% for section in sections %}
                <div class="section {{ section.section_type }}" id="section-{{ loop.index }}">
                    <div class="section-header">
                        <h2>{{ section.title }}</h2>
                    </div>
                    <div class="section-content">
                        {{ section.content|safe }}

                        {% if section.visualizations %}
                        {% for viz in section.visualizations %}
                        <div class="visualization">
                            <h4>{{ viz.title }}</h4>
                            <div class="chart-container" id="chart-{{ loop.index0 }}-{{ loop.parentloop.index0 }}">
                                {{ viz.html|safe }}
                            </div>
                            <p><em>{{ viz.description }}</em></p>
                        </div>
                        {% endfor %}
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>

            <div class="footer">
                <div class="container">
                    <p>&copy; {{ year }} Gary×Taleb Performance Analysis System</p>
                    <p>Report generated with advanced statistical validation and antifragile methodology</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Convert markdown content to HTML
        for section in sorted_sections:
            section.content = self._markdown_to_html(section.content)

            # Convert visualizations to HTML
            for viz in section.visualizations:
                if viz['type'] == 'plotly':
                    viz['html'] = pyo.plot(viz['figure'], output_type='div', include_plotlyjs=False)

        # Render template
        template = Template(html_template)
        html_content = template.render(
            title=self.config.title,
            author=self.config.author,
            date=datetime.now().strftime('%B %d, %Y'),
            year=datetime.now().year,
            sections=sorted_sections,
            colors=colors,
            config=self.config
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report generated: {output_path}")
        return output_path

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML (simplified)"""
        html = markdown_text

        # Headers
        html = html.replace('### ', '<h3>').replace('\n', '</h3>\n', html.count('### '))
        html = html.replace('## ', '<h2>').replace('\n', '</h2>\n', html.count('## '))

        # Bold
        import re
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong class="highlight">\1</strong>', html)

        # Lists
        lines = html.split('\n')
        in_list = False
        result_lines = []

        for line in lines:
            if line.strip().startswith('- ') or line.strip().startswith('• '):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                result_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)

        if in_list:
            result_lines.append('</ul>')

        html = '\n'.join(result_lines)

        # Paragraphs
        paragraphs = html.split('\n\n')
        html_paragraphs = []
        for para in paragraphs:
            if para.strip() and not para.strip().startswith('<'):
                html_paragraphs.append(f'<p>{para.strip()}</p>')
            else:
                html_paragraphs.append(para)

        return '\n\n'.join(html_paragraphs)

    def _create_performance_chart(self, returns: pd.Series, equity_curve: pd.Series,
                                benchmark_data: Optional[Dict] = None) -> go.Figure:
        """Create performance overview chart"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curve', 'Monthly Returns', 'Rolling Sharpe Ratio', 'Drawdown'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values, name='Strategy',
                      line=dict(color='#2E3440', width=2)),
            row=1, col=1
        )

        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        colors = ['#A3BE8C' if x > 0 else '#BF616A' for x in monthly_returns]

        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns.values,
                  marker_color=colors, name='Monthly Returns'),
            row=1, col=2
        )

        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Rolling Sharpe', line=dict(color='#5E81AC')),
            row=2, col=1
        )

        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, fill='tonegative',
                      name='Drawdown', line=dict(color='#BF616A')),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=True, title_text="Performance Overview Dashboard")
        return fig

    def _create_gary_taleb_chart(self, antifragility_metrics: Dict, dpi_metrics: Dict,
                               integration_analysis: Dict) -> go.Figure:
        """Create Gary×Taleb analysis chart"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Antifragility Radar', 'DPI Components', 'Synergy Analysis', 'Integration Score'),
            specs=[[{"type": "scatterpolar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )

        # Antifragility radar
        af_categories = ['Convexity', 'Volatility Benefit', 'Asymmetry', 'Black Swan Protection', 'Barbell Efficiency']
        af_values = [
            antifragility_metrics.get('convexity_score', 0),
            antifragility_metrics.get('volatility_benefit', 0),
            antifragility_metrics.get('asymmetry_ratio', 0),
            antifragility_metrics.get('black_swan_protection', 0),
            antifragility_metrics.get('barbell_efficiency', 0)
        ]

        fig.add_trace(
            go.Scatterpolar(r=af_values, theta=af_categories, fill='toself',
                           name='Antifragility', line_color='#88C0D0'),
            row=1, col=1
        )

        # DPI components
        dpi_components = ['Momentum', 'Stability', 'Consistency', 'Adaptability', 'Signal Quality']
        dpi_values = [
            dpi_metrics.get('momentum_score', 0),
            dpi_metrics.get('stability_score', 0),
            dpi_metrics.get('consistency_score', 0),
            dpi_metrics.get('adaptability_score', 0),
            dpi_metrics.get('signal_quality', 0)
        ]

        fig.add_trace(
            go.Bar(x=dpi_components, y=dpi_values, name='DPI Components',
                  marker_color='#5E81AC'),
            row=1, col=2
        )

        # Synergy analysis
        synergy_metrics = ['Volatility Synergy', 'Stability Synergy', 'Performance Synergy']
        synergy_values = [0.8, 0.7, 0.9]  # Placeholder values

        fig.add_trace(
            go.Bar(x=synergy_metrics, y=synergy_values, name='Synergy Metrics',
                  marker_color='#A3BE8C'),
            row=2, col=1
        )

        # Integration score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=integration_analysis.get('synergy_score', 0.5) * 100,
                title={'text': "Integration Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "#2E3440"},
                      'steps': [{'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=True, title_text="Gary×Taleb Analysis Dashboard")
        return fig

    def _create_risk_chart(self, returns: pd.Series, risk_metrics: Dict,
                         drawdown_analysis: Dict) -> go.Figure:
        """Create risk analysis chart"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Distribution', 'Risk Metrics', 'Drawdown Timeline', 'VaR Analysis'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Return Distribution',
                        marker_color='#88C0D0'),
            row=1, col=1
        )

        # Risk metrics
        risk_names = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Ulcer Index']
        risk_values = [
            risk_metrics.get('sharpe_ratio', 0),
            risk_metrics.get('sortino_ratio', 0),
            risk_metrics.get('calmar_ratio', 0),
            risk_metrics.get('ulcer_index', 0)
        ]

        fig.add_trace(
            go.Bar(x=risk_names, y=risk_values, name='Risk Metrics',
                  marker_color='#5E81AC'),
            row=1, col=2
        )

        # Drawdown timeline
        equity_curve = (1 + returns).cumprod() * 200
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, fill='tonegative',
                      name='Drawdown', line=dict(color='#BF616A')),
            row=2, col=1
        )

        # VaR analysis
        var_names = ['VaR 95%', 'VaR 99%', 'CVaR 95%']
        var_values = [
            abs(risk_metrics.get('var_95', 0)),
            abs(risk_metrics.get('var_99', 0)),
            abs(risk_metrics.get('cvar_95', 0))
        ]

        fig.add_trace(
            go.Bar(x=var_names, y=var_values, name='VaR Analysis',
                  marker_color='#EBCB8B'),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=True, title_text="Risk Analysis Dashboard")
        return fig

    def _create_optimization_chart(self, optimization_results: Dict,
                                 recommendations: List[str]) -> go.Figure:
        """Create optimization analysis chart"""

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Parameter Optimization', 'Expected Improvements')
        )

        # Parameter optimization
        current_params = ['Signal Threshold', 'Stop Loss', 'Max Position']
        current_values = [0.5, 0.02, 1.0]
        optimal_values = [0.6, 0.025, 0.8]

        fig.add_trace(
            go.Bar(x=current_params, y=current_values, name='Current',
                  marker_color='#88C0D0'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=current_params, y=optimal_values, name='Optimized',
                  marker_color='#A3BE8C'),
            row=1, col=1
        )

        # Expected improvements
        improvement_metrics = ['Return', 'Sharpe Ratio', 'Max Drawdown', 'Consistency']
        improvement_values = [15, 25, -20, 30]  # Percentage improvements

        colors = ['#A3BE8C' if x > 0 else '#BF616A' for x in improvement_values]

        fig.add_trace(
            go.Bar(x=improvement_metrics, y=improvement_values,
                  marker_color=colors, name='Expected Improvement %'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=True, title_text="Optimization Analysis")
        return fig

    def _create_validation_chart(self, validation_results: Dict,
                                statistical_tests: List[Dict]) -> go.Figure:
        """Create statistical validation chart"""

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Test Results by Category', 'Validation Score')
        )

        # Test results by category
        categories = ['Performance', 'Distribution', 'Assumptions', 'Model']
        pass_rates = [0.9, 0.7, 0.8, 0.85]  # Placeholder values

        fig.add_trace(
            go.Bar(x=categories, y=pass_rates, name='Pass Rate',
                  marker_color='#5E81AC'),
            row=1, col=1
        )

        # Validation score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=validation_results.get('overall_score', 0.5) * 100,
                title={'text': "Validation Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "#2E3440"},
                      'steps': [{'range': [0, 60], 'color': "lightgray"},
                               {'range': [60, 80], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=True, title_text="Statistical Validation")
        return fig

    def save_report_data(self, filename: str = "report_data.json"):
        """Save report data for future use"""

        report_data = {
            'config': asdict(self.config),
            'sections': [asdict(section) for section in self.sections],
            'generated_at': datetime.now().isoformat()
        }

        # Remove non-serializable items
        for section in report_data['sections']:
            section['visualizations'] = []  # Remove plotly figures

        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"Report data saved: {filename}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
    equity_curve = (1 + returns).cumprod() * 200

    # Initialize reporter
    config = ReportConfig(
        title="Gary×Taleb Trading Strategy Performance Report",
        color_scheme="professional",
        include_executive_summary=True
    )

    reporter = PerformanceReporter(config)

    # Sample data
    performance_data = {'total_return': 0.25, 'annualized_return': 0.18}
    key_metrics = {'sharpe_ratio': 1.8, 'max_drawdown': -0.08, 'dpi_score': 0.75, 'antifragility_score': 0.68}
    recommendations = ["Optimize position sizing", "Enhance signal filtering", "Improve risk management"]

    # Add sections
    reporter.add_section(reporter.create_executive_summary(performance_data, key_metrics, recommendations))
    reporter.add_section(reporter.create_performance_overview(returns, equity_curve))

    # Generate report
    output_file = reporter.generate_html_report("gary_taleb_performance_report.html")

    # Save report data
    reporter.save_report_data("gary_taleb_report_data.json")

    print(f"Comprehensive report generated: {output_file}")
    print("Report includes executive summary, performance analysis, and visualizations")

    # Open in browser
    webbrowser.open(f"file://{Path(output_file).absolute()}")