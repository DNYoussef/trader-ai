"""
Information Mycelium / Distributional Flow Ledger (DFL)

Implements Gary's vision of tracking wealth flows by income decile to understand
"who captures each marginal unit of cash/credit". This system maps landlord and
creditor capture patterns, monitors housing affordability, and tracks margin/markup trends.

Core Philosophy: "Follow the Flow - always map incidence (who keeps the cash/claims)"

Mathematical Foundation:
- Real-time flow tracking across income deciles (D1-D10)
- Capture rate analysis for different wealth holders
- Marginal flow tracking for every unit of new money/credit
- Integration with DPI to provide distributional context for trading decisions
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
from scipy import stats
from collections import defaultdict, deque
import sqlite3
import json

logger = logging.getLogger(__name__)


class IncomeDecile(Enum):
    """Income deciles from poorest (D1) to richest (D10)"""
    D1 = "decile_1"  # Bottom 10% (poorest)
    D2 = "decile_2"
    D3 = "decile_3"
    D4 = "decile_4"
    D5 = "decile_5"  # Median
    D6 = "decile_6"
    D7 = "decile_7"
    D8 = "decile_8"
    D9 = "decile_9"
    D10 = "decile_10"  # Top 10% (richest)


class FlowCaptor(Enum):
    """Types of wealth flow captors"""
    LANDLORDS = "landlords"
    CREDITORS = "creditors"
    EMPLOYERS = "employers"
    RETAILERS = "retailers"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    GOVERNMENT = "government"
    FINANCIAL_SERVICES = "financial_services"
    UTILITIES = "utilities"
    OTHER = "other"


@dataclass
class FlowEvent:
    """Individual wealth flow event"""
    timestamp: datetime
    amount: float
    source_decile: IncomeDecile
    captor_type: FlowCaptor
    captor_id: str
    flow_category: str  # 'rent', 'credit_payment', 'grocery', etc.
    urgency_score: float  # 0-1, how essential/urgent this flow is
    elasticity: float  # Price elasticity of this flow
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecileFlowProfile:
    """Flow profile for a specific income decile"""
    decile: IncomeDecile
    total_income: float
    discretionary_income: float
    essential_flows: Dict[FlowCaptor, float]
    discretionary_flows: Dict[FlowCaptor, float]
    capture_rates: Dict[FlowCaptor, float]  # % of income captured by each type
    vulnerability_score: float  # 0-1, how vulnerable to capture
    flow_velocity: float  # How quickly income flows out
    margin_sensitivity: float  # Sensitivity to price increases


@dataclass
class CaptorProfile:
    """Profile of a wealth flow captor"""
    captor_id: str
    captor_type: FlowCaptor
    total_captured: float
    decile_distribution: Dict[IncomeDecile, float]  # How much from each decile
    market_power: float  # 0-1, monopolistic/competitive position
    pricing_power: float  # 0-1, ability to raise prices
    essentiality_score: float  # 0-1, how essential their service is
    capture_efficiency: float  # % of customer income captured
    growth_rate: float  # Rate of capture growth


@dataclass
class MarginMarkupAnalysis:
    """Analysis of margin and markup trends"""
    sector: str
    average_margin: float
    margin_trend: float  # % change over time
    price_elasticity: float
    market_concentration: float  # Herfindahl index
    regulatory_capture: float  # 0-1, degree of regulatory protection
    barrier_height: float  # 0-1, barriers to entry
    consumer_surplus_extraction: float  # % of consumer surplus captured


class DistributionalFlowLedger:
    """
    Information Mycelium tracking wealth flows across income deciles

    This system implements Gary's vision of mapping who captures each marginal
    unit of cash/credit in the economy, with particular focus on landlord and
    creditor capture patterns.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Distributional Flow Ledger

        Args:
            db_path: Path to SQLite database for persistence
        """
        self.db_path = db_path or ":memory:"
        self.flow_history: deque = deque(maxlen=10000)  # Recent flow events
        self.decile_profiles: Dict[IncomeDecile, DecileFlowProfile] = {}
        self.captor_profiles: Dict[str, CaptorProfile] = {}
        self.margin_analyses: Dict[str, MarginMarkupAnalysis] = {}

        # Initialize database
        self._init_database()

        # Initialize decile profiles with realistic data
        self._initialize_decile_profiles()

        logger.info("Distributional Flow Ledger initialized")

    def _init_database(self):
        """Initialize SQLite database for flow tracking"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()

            # Create flow events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    amount REAL NOT NULL,
                    source_decile TEXT NOT NULL,
                    captor_type TEXT NOT NULL,
                    captor_id TEXT NOT NULL,
                    flow_category TEXT NOT NULL,
                    urgency_score REAL NOT NULL,
                    elasticity REAL NOT NULL,
                    metadata TEXT
                )
            """)

            # Create decile snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decile_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    decile TEXT NOT NULL,
                    total_income REAL NOT NULL,
                    discretionary_income REAL NOT NULL,
                    vulnerability_score REAL NOT NULL,
                    flow_velocity REAL NOT NULL,
                    capture_data TEXT
                )
            """)

            # Create captor profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS captor_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    captor_id TEXT NOT NULL,
                    captor_type TEXT NOT NULL,
                    total_captured REAL NOT NULL,
                    market_power REAL NOT NULL,
                    pricing_power REAL NOT NULL,
                    capture_efficiency REAL NOT NULL,
                    decile_data TEXT
                )
            """)

            self.conn.commit()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _initialize_decile_profiles(self):
        """Initialize realistic decile profiles based on US economic data"""
        # US income distribution approximations (2023 data)
        decile_incomes = {
            IncomeDecile.D1: 15000,   # Bottom 10%
            IncomeDecile.D2: 25000,   # 10-20%
            IncomeDecile.D3: 35000,   # 20-30%
            IncomeDecile.D4: 45000,   # 30-40%
            IncomeDecile.D5: 55000,   # 40-50% (Median)
            IncomeDecile.D6: 70000,   # 50-60%
            IncomeDecile.D7: 85000,   # 60-70%
            IncomeDecile.D8: 110000,  # 70-80%
            IncomeDecile.D9: 150000,  # 80-90%
            IncomeDecile.D10: 350000, # Top 10%
        }

        # Housing cost ratios by decile (% of income)
        housing_ratios = {
            IncomeDecile.D1: 0.60,  # 60% of income to housing (severe burden)
            IncomeDecile.D2: 0.50,  # 50% (cost burdened)
            IncomeDecile.D3: 0.40,  # 40%
            IncomeDecile.D4: 0.35,  # 35%
            IncomeDecile.D5: 0.30,  # 30%
            IncomeDecile.D6: 0.28,  # 28%
            IncomeDecile.D7: 0.25,  # 25%
            IncomeDecile.D8: 0.22,  # 22%
            IncomeDecile.D9: 0.20,  # 20%
            IncomeDecile.D10: 0.15, # 15%
        }

        for decile, income in decile_incomes.items():
            housing_cost = income * housing_ratios[decile]
            essential_flows = {
                FlowCaptor.LANDLORDS: housing_cost,
                FlowCaptor.UTILITIES: income * 0.08,  # 8% utilities
                FlowCaptor.HEALTHCARE: income * 0.12,  # 12% healthcare
                FlowCaptor.RETAILERS: income * 0.15,   # 15% groceries/essentials
                FlowCaptor.CREDITORS: income * max(0.05, 0.25 - (income / 500000))  # Higher for lower deciles
            }

            total_essential = sum(essential_flows.values())
            discretionary_income = max(0, income - total_essential)

            # Calculate capture rates
            capture_rates = {captor: amount / income for captor, amount in essential_flows.items()}

            # Vulnerability score (higher for lower deciles)
            vulnerability_score = max(0, 1.0 - (discretionary_income / income))

            # Flow velocity (how quickly income flows out)
            flow_velocity = min(1.0, total_essential / income)

            profile = DecileFlowProfile(
                decile=decile,
                total_income=income,
                discretionary_income=discretionary_income,
                essential_flows=essential_flows,
                discretionary_flows={},
                capture_rates=capture_rates,
                vulnerability_score=vulnerability_score,
                flow_velocity=flow_velocity,
                margin_sensitivity=vulnerability_score * 0.8  # Vulnerable deciles more sensitive
            )

            self.decile_profiles[decile] = profile

    def record_flow_event(self, flow_event: FlowEvent):
        """Record a new wealth flow event"""
        try:
            # Add to memory
            self.flow_history.append(flow_event)

            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO flow_events (
                    timestamp, amount, source_decile, captor_type, captor_id,
                    flow_category, urgency_score, elasticity, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flow_event.timestamp.isoformat(),
                flow_event.amount,
                flow_event.source_decile.value,
                flow_event.captor_type.value,
                flow_event.captor_id,
                flow_event.flow_category,
                flow_event.urgency_score,
                flow_event.elasticity,
                json.dumps(flow_event.metadata)
            ))
            self.conn.commit()

            # Update profiles
            self._update_profiles_from_flow(flow_event)

        except Exception as e:
            logger.error(f"Error recording flow event: {e}")
            raise

    def _update_profiles_from_flow(self, flow_event: FlowEvent):
        """Update decile and captor profiles based on new flow"""
        # Update captor profile
        if flow_event.captor_id not in self.captor_profiles:
            self.captor_profiles[flow_event.captor_id] = CaptorProfile(
                captor_id=flow_event.captor_id,
                captor_type=flow_event.captor_type,
                total_captured=0.0,
                decile_distribution={},
                market_power=0.5,  # Default
                pricing_power=0.5,  # Default
                essentiality_score=flow_event.urgency_score,
                capture_efficiency=0.0,
                growth_rate=0.0
            )

        captor = self.captor_profiles[flow_event.captor_id]
        captor.total_captured += flow_event.amount

        # Update decile distribution for captor
        if flow_event.source_decile not in captor.decile_distribution:
            captor.decile_distribution[flow_event.source_decile] = 0.0
        captor.decile_distribution[flow_event.source_decile] += flow_event.amount

    def track_marginal_flow(self, amount: float, policy_context: str = "") -> Dict[str, Any]:
        """
        Track who captures each marginal unit of new money/credit

        Args:
            amount: Amount of new money/credit entering system
            policy_context: Context of money creation (QE, stimulus, etc.)

        Returns:
            Analysis of who captures the marginal flows
        """
        logger.info(f"Tracking marginal flow of ${amount:,.2f} ({policy_context})")

        try:
            marginal_captures = {}
            total_captured = 0.0

            # Simulate marginal flow through deciles
            for decile, profile in self.decile_profiles.items():
                # Amount reaching this decile (based on income distribution)
                decile_share = self._calculate_decile_money_share(decile, policy_context)
                decile_amount = amount * decile_share

                # Calculate what each captor type captures from this decile
                for captor_type, capture_rate in profile.capture_rates.items():
                    captured = decile_amount * capture_rate

                    if captor_type not in marginal_captures:
                        marginal_captures[captor_type] = {
                            'total': 0.0,
                            'by_decile': {},
                            'capture_rate': 0.0
                        }

                    marginal_captures[captor_type]['total'] += captured
                    marginal_captures[captor_type]['by_decile'][decile.value] = captured
                    total_captured += captured

            # Calculate overall capture rates
            for captor_type in marginal_captures:
                marginal_captures[captor_type]['capture_rate'] = (
                    marginal_captures[captor_type]['total'] / amount
                )

            # Calculate what remains with original recipients
            retained = amount - total_captured

            analysis = {
                'timestamp': datetime.now(),
                'injected_amount': amount,
                'policy_context': policy_context,
                'marginal_captures': marginal_captures,
                'total_captured': total_captured,
                'retained_by_recipients': retained,
                'capture_efficiency': total_captured / amount,
                'flow_velocity': self._calculate_system_flow_velocity()
            }

            logger.info(f"Marginal flow analysis: {total_captured/amount:.1%} captured by captors")
            return analysis

        except Exception as e:
            logger.error(f"Error tracking marginal flow: {e}")
            raise

    def _calculate_decile_money_share(self, decile: IncomeDecile, policy_context: str) -> float:
        """Calculate what share of new money reaches each decile"""
        # Different policies have different distributional effects
        if "stimulus" in policy_context.lower():
            # Stimulus tends to be more bottom-heavy
            shares = {
                IncomeDecile.D1: 0.15, IncomeDecile.D2: 0.13, IncomeDecile.D3: 0.12,
                IncomeDecile.D4: 0.11, IncomeDecile.D5: 0.10, IncomeDecile.D6: 0.10,
                IncomeDecile.D7: 0.09, IncomeDecile.D8: 0.08, IncomeDecile.D9: 0.07,
                IncomeDecile.D10: 0.05
            }
        elif "qe" in policy_context.lower() or "asset" in policy_context.lower():
            # QE/asset purchases flow more to top
            shares = {
                IncomeDecile.D1: 0.02, IncomeDecile.D2: 0.03, IncomeDecile.D3: 0.04,
                IncomeDecile.D4: 0.05, IncomeDecile.D5: 0.06, IncomeDecile.D6: 0.08,
                IncomeDecile.D7: 0.10, IncomeDecile.D8: 0.15, IncomeDecile.D9: 0.20,
                IncomeDecile.D10: 0.27
            }
        else:
            # Default to income-proportional distribution
            total_income = sum(p.total_income for p in self.decile_profiles.values())
            shares = {d: p.total_income / total_income for d, p in self.decile_profiles.items()}

        return shares.get(decile, 0.1)  # Default 10% if not found

    def analyze_housing_affordability(self) -> Dict[str, Any]:
        """Analyze housing affordability across deciles"""
        affordability_analysis = {
            'timestamp': datetime.now(),
            'decile_analysis': {},
            'overall_metrics': {}
        }

        cost_burdened_count = 0  # >30% income to housing
        severely_burdened_count = 0  # >50% income to housing
        total_housing_capture = 0.0

        for decile, profile in self.decile_profiles.items():
            housing_cost = profile.essential_flows.get(FlowCaptor.LANDLORDS, 0)
            housing_ratio = housing_cost / profile.total_income

            is_cost_burdened = housing_ratio > 0.30
            is_severely_burdened = housing_ratio > 0.50

            if is_cost_burdened:
                cost_burdened_count += 1
            if is_severely_burdened:
                severely_burdened_count += 1

            total_housing_capture += housing_cost

            affordability_analysis['decile_analysis'][decile.value] = {
                'housing_cost': housing_cost,
                'housing_ratio': housing_ratio,
                'cost_burdened': is_cost_burdened,
                'severely_burdened': is_severely_burdened,
                'affordability_score': max(0, 1 - (housing_ratio - 0.20) / 0.40)  # 1 at 20%, 0 at 60%
            }

        # Overall metrics
        total_income = sum(p.total_income for p in self.decile_profiles.values())

        affordability_analysis['overall_metrics'] = {
            'cost_burdened_deciles': cost_burdened_count,
            'severely_burdened_deciles': severely_burdened_count,
            'total_housing_capture': total_housing_capture,
            'housing_capture_rate': total_housing_capture / total_income,
            'affordability_crisis_score': severely_burdened_count / len(self.decile_profiles)
        }

        return affordability_analysis

    def analyze_margin_markup_trends(self, sector: str, time_window_days: int = 90) -> MarginMarkupAnalysis:
        """Analyze margin and markup trends for a specific sector"""
        try:
            # Get recent flow events for this sector
            recent_flows = self._get_recent_flows_by_sector(sector, time_window_days)

            if not recent_flows:
                logger.warning(f"No recent flows found for sector {sector}")
                return self._create_default_margin_analysis(sector)

            # Calculate margin metrics
            total_flows = sum(flow.amount for flow in recent_flows)
            flow_counts = len(recent_flows)

            # Estimate margins based on flow patterns
            average_flow = total_flows / flow_counts if flow_counts > 0 else 0

            # Calculate market concentration (simplified Herfindahl)
            captor_flows = defaultdict(float)
            for flow in recent_flows:
                captor_flows[flow.captor_id] += flow.amount

            # Herfindahl index calculation
            captor_shares = [(flow / total_flows)**2 for flow in captor_flows.values()]
            herfindahl = sum(captor_shares) if captor_shares else 0

            # Estimate pricing power from elasticity
            elasticities = [flow.elasticity for flow in recent_flows if flow.elasticity > 0]
            avg_elasticity = statistics.mean(elasticities) if elasticities else -1.0
            pricing_power = max(0, 1 + avg_elasticity)  # Higher for less elastic goods

            # Calculate trend (simple linear)
            if len(recent_flows) >= 5:
                # Sort by timestamp and calculate trend
                sorted_flows = sorted(recent_flows, key=lambda x: x.timestamp)
                flow_amounts = [flow.amount for flow in sorted_flows]
                x = np.arange(len(flow_amounts))
                slope, _, _, _, _ = stats.linregress(x, flow_amounts)
                margin_trend = slope / average_flow if average_flow > 0 else 0
            else:
                margin_trend = 0.0

            analysis = MarginMarkupAnalysis(
                sector=sector,
                average_margin=0.25,  # Placeholder - would need cost data
                margin_trend=margin_trend,
                price_elasticity=avg_elasticity,
                market_concentration=herfindahl,
                regulatory_capture=min(1.0, herfindahl * 1.5),  # Higher concentration = more capture
                barrier_height=pricing_power,
                consumer_surplus_extraction=min(0.8, pricing_power * 0.8)
            )

            self.margin_analyses[sector] = analysis
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing margin trends for {sector}: {e}")
            return self._create_default_margin_analysis(sector)

    def _get_recent_flows_by_sector(self, sector: str, days: int) -> List[FlowEvent]:
        """Get recent flow events for a specific sector"""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter in-memory flows first (faster)
        recent_flows = [
            flow for flow in self.flow_history
            if flow.timestamp >= cutoff_date and
            (sector.lower() in flow.flow_category.lower() or
             sector.lower() in flow.captor_type.value.lower())
        ]

        # If we need more data, query database
        if len(recent_flows) < 10:
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    SELECT * FROM flow_events
                    WHERE timestamp >= ? AND
                          (flow_category LIKE ? OR captor_type LIKE ?)
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (
                    cutoff_date.isoformat(),
                    f"%{sector}%",
                    f"%{sector}%"
                ))

                db_flows = []
                for row in cursor.fetchall():
                    flow = FlowEvent(
                        timestamp=datetime.fromisoformat(row[1]),
                        amount=row[2],
                        source_decile=IncomeDecile(row[3]),
                        captor_type=FlowCaptor(row[4]),
                        captor_id=row[5],
                        flow_category=row[6],
                        urgency_score=row[7],
                        elasticity=row[8],
                        metadata=json.loads(row[9]) if row[9] else {}
                    )
                    db_flows.append(flow)

                recent_flows.extend(db_flows)

            except Exception as e:
                logger.error(f"Error querying database for sector flows: {e}")

        return recent_flows

    def _create_default_margin_analysis(self, sector: str) -> MarginMarkupAnalysis:
        """Create default margin analysis when no data available"""
        return MarginMarkupAnalysis(
            sector=sector,
            average_margin=0.20,  # 20% default margin
            margin_trend=0.0,
            price_elasticity=-1.0,  # Unit elastic default
            market_concentration=0.15,  # Moderate concentration
            regulatory_capture=0.3,  # Moderate regulatory capture
            barrier_height=0.4,  # Moderate barriers
            consumer_surplus_extraction=0.3  # Moderate extraction
        )

    def _calculate_system_flow_velocity(self) -> float:
        """Calculate overall system flow velocity"""
        velocities = [profile.flow_velocity for profile in self.decile_profiles.values()]
        return statistics.mean(velocities) if velocities else 0.5

    def get_landlord_capture_analysis(self) -> Dict[str, Any]:
        """Specific analysis of landlord wealth capture patterns"""
        analysis = {
            'timestamp': datetime.now(),
            'total_capture': 0.0,
            'decile_breakdown': {},
            'concentration_metrics': {},
            'affordability_impact': {}
        }

        # Calculate total landlord capture
        for decile, profile in self.decile_profiles.items():
            landlord_capture = profile.essential_flows.get(FlowCaptor.LANDLORDS, 0)
            analysis['total_capture'] += landlord_capture

            analysis['decile_breakdown'][decile.value] = {
                'absolute_capture': landlord_capture,
                'capture_rate': landlord_capture / profile.total_income,
                'burden_category': self._categorize_housing_burden(
                    landlord_capture / profile.total_income
                )
            }

        # Calculate concentration metrics
        landlord_captors = [
            captor for captor in self.captor_profiles.values()
            if captor.captor_type == FlowCaptor.LANDLORDS
        ]

        if landlord_captors:
            total_landlord_capture = sum(captor.total_captured for captor in landlord_captors)
            captor_shares = [
                (captor.total_captured / total_landlord_capture)**2
                for captor in landlord_captors
            ]
            herfindahl = sum(captor_shares)

            analysis['concentration_metrics'] = {
                'herfindahl_index': herfindahl,
                'market_concentration': 'High' if herfindahl > 0.25 else 'Moderate' if herfindahl > 0.15 else 'Low',
                'number_of_captors': len(landlord_captors)
            }

        return analysis

    def _categorize_housing_burden(self, housing_ratio: float) -> str:
        """Categorize housing burden level"""
        if housing_ratio > 0.50:
            return "Severely Burdened"
        elif housing_ratio > 0.30:
            return "Cost Burdened"
        elif housing_ratio > 0.20:
            return "Moderate Burden"
        else:
            return "Affordable"

    def get_creditor_capture_analysis(self) -> Dict[str, Any]:
        """Specific analysis of creditor wealth capture patterns"""
        analysis = {
            'timestamp': datetime.now(),
            'total_capture': 0.0,
            'decile_breakdown': {},
            'debt_burden_analysis': {},
            'predatory_indicators': {}
        }

        # Calculate creditor capture by decile
        for decile, profile in self.decile_profiles.items():
            creditor_capture = profile.essential_flows.get(FlowCaptor.CREDITORS, 0)
            analysis['total_capture'] += creditor_capture

            debt_burden_ratio = creditor_capture / profile.total_income

            analysis['decile_breakdown'][decile.value] = {
                'absolute_capture': creditor_capture,
                'debt_burden_ratio': debt_burden_ratio,
                'debt_stress_level': self._assess_debt_stress(debt_burden_ratio, profile.vulnerability_score)
            }

        # Analyze predatory lending indicators
        lower_decile_burden = sum(
            analysis['decile_breakdown'][d.value]['debt_burden_ratio']
            for d in [IncomeDecile.D1, IncomeDecile.D2, IncomeDecile.D3]
        ) / 3

        upper_decile_burden = sum(
            analysis['decile_breakdown'][d.value]['debt_burden_ratio']
            for d in [IncomeDecile.D8, IncomeDecile.D9, IncomeDecile.D10]
        ) / 3

        analysis['predatory_indicators'] = {
            'burden_inequality': lower_decile_burden / upper_decile_burden if upper_decile_burden > 0 else float('inf'),
            'predatory_score': min(1.0, lower_decile_burden / 0.30),  # Score based on 30% threshold
            'systemic_risk': lower_decile_burden > 0.25  # Risk if bottom deciles >25% burden
        }

        return analysis

    def _assess_debt_stress(self, debt_ratio: float, vulnerability: float) -> str:
        """Assess debt stress level"""
        combined_stress = debt_ratio + vulnerability * 0.5

        if combined_stress > 0.50:
            return "Critical"
        elif combined_stress > 0.35:
            return "High"
        elif combined_stress > 0.20:
            return "Moderate"
        else:
            return "Low"

    def generate_flow_intelligence_summary(self) -> Dict[str, Any]:
        """Generate comprehensive flow intelligence summary for trading decisions"""
        summary = {
            'timestamp': datetime.now(),
            'system_status': {
                'total_deciles_tracked': len(self.decile_profiles),
                'total_captors_tracked': len(self.captor_profiles),
                'flow_events_recorded': len(self.flow_history),
                'system_flow_velocity': self._calculate_system_flow_velocity()
            },
            'distributional_pressure': {},
            'capture_analysis': {},
            'trading_implications': {}
        }

        # Calculate distributional pressure metrics
        vulnerability_scores = [p.vulnerability_score for p in self.decile_profiles.values()]
        flow_velocities = [p.flow_velocity for p in self.decile_profiles.values()]

        summary['distributional_pressure'] = {
            'average_vulnerability': statistics.mean(vulnerability_scores),
            'vulnerability_inequality': max(vulnerability_scores) - min(vulnerability_scores),
            'system_flow_velocity': statistics.mean(flow_velocities),
            'pressure_score': statistics.mean(vulnerability_scores) * statistics.mean(flow_velocities)
        }

        # Capture analysis
        housing_analysis = self.analyze_housing_affordability()
        landlord_analysis = self.get_landlord_capture_analysis()
        creditor_analysis = self.get_creditor_capture_analysis()

        summary['capture_analysis'] = {
            'housing_affordability': housing_analysis['overall_metrics'],
            'landlord_capture': landlord_analysis.get('concentration_metrics', {}),
            'creditor_capture': creditor_analysis.get('predatory_indicators', {})
        }

        # Trading implications
        pressure_score = summary['distributional_pressure']['pressure_score']
        crisis_score = housing_analysis['overall_metrics'].get('affordability_crisis_score', 0)

        summary['trading_implications'] = {
            'defensive_positioning_recommended': pressure_score > 0.7 or crisis_score > 0.6,
            'consumer_discretionary_risk': pressure_score,
            'housing_sector_risk': crisis_score,
            'credit_risk_alert': creditor_analysis.get('predatory_indicators', {}).get('systemic_risk', False),
            'recommended_sectors': self._get_recommended_sectors(pressure_score, crisis_score)
        }

        return summary

    def _get_recommended_sectors(self, pressure_score: float, crisis_score: float) -> List[str]:
        """Get sector recommendations based on distributional analysis"""
        recommendations = []

        if pressure_score > 0.7:
            recommendations.extend(['utilities', 'consumer_staples', 'healthcare'])

        if crisis_score > 0.6:
            recommendations.extend(['rental_housing_reits', 'debt_collection', 'discount_retail'])

        if pressure_score < 0.3 and crisis_score < 0.3:
            recommendations.extend(['consumer_discretionary', 'luxury_goods', 'travel'])

        return list(set(recommendations))  # Remove duplicates

    def integrate_with_dpi(self, dpi_score: float, symbol: str) -> Dict[str, Any]:
        """
        Integrate distributional flow analysis with DPI calculations

        Args:
            dpi_score: Current DPI score for the symbol
            symbol: Trading symbol

        Returns:
            Enhanced analysis combining DPI with distributional flows
        """
        flow_summary = self.generate_flow_intelligence_summary()

        # Calculate distributional enhancement factor
        pressure_score = flow_summary['distributional_pressure']['pressure_score']
        vulnerability = flow_summary['distributional_pressure']['average_vulnerability']

        # Adjust DPI based on distributional context
        distributional_factor = 1.0

        if pressure_score > 0.7:  # High distributional pressure
            if dpi_score > 0:  # Bullish DPI
                distributional_factor = 0.8  # Reduce bullish signal
            else:  # Bearish DPI
                distributional_factor = 1.2  # Amplify bearish signal

        adjusted_dpi = dpi_score * distributional_factor

        integration_analysis = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'original_dpi': dpi_score,
            'adjusted_dpi': adjusted_dpi,
            'distributional_factor': distributional_factor,
            'flow_context': {
                'pressure_score': pressure_score,
                'vulnerability_score': vulnerability,
                'crisis_indicators': flow_summary['capture_analysis']
            },
            'trading_recommendation': {
                'position_size_adjustment': distributional_factor - 1.0,
                'risk_level': 'High' if pressure_score > 0.7 else 'Moderate' if pressure_score > 0.4 else 'Low',
                'sector_rotation_signal': flow_summary['trading_implications']['recommended_sectors']
            }
        }

        return integration_analysis

    def close(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()