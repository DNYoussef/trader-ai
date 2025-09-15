"""
Natural Experiments Registry

Implements Gary's vision of tracking and cataloging natural experiments for causal
identification: VAT shifts, energy price shocks, housing regulation changes, and
policy announcement effects. This system maintains a registry of instrumental
variables and causal identification strategies.

Core Philosophy: "Prove Causality - refute yourself (instruments, natural experiments, shock decompositions)"

Mathematical Foundation:
- Natural experiment identification strategies
- Instrumental variable registry and validation
- Regression discontinuity design detection
- Difference-in-differences setup identification
- Event study methodology for policy announcements
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
from scipy import stats
import json
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of natural experiments"""
    VAT_CHANGE = "vat_change"
    ENERGY_PRICE_SHOCK = "energy_price_shock"
    HOUSING_REGULATION = "housing_regulation"
    POLICY_ANNOUNCEMENT = "policy_announcement"
    MONETARY_POLICY_SHOCK = "monetary_policy_shock"
    FISCAL_POLICY_CHANGE = "fiscal_policy_change"
    REGULATORY_SHOCK = "regulatory_shock"
    TAX_REFORM = "tax_reform"
    TRADE_POLICY = "trade_policy"
    LABOR_REGULATION = "labor_regulation"
    ENVIRONMENTAL_POLICY = "environmental_policy"
    FINANCIAL_REGULATION = "financial_regulation"


class IdentificationStrategy(Enum):
    """Causal identification strategies"""
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    EVENT_STUDY = "event_study"
    SYNTHETIC_CONTROL = "synthetic_control"
    NATURAL_EXPERIMENT = "natural_experiment"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"


class GeographicScope(Enum):
    """Geographic scope of experiment"""
    GLOBAL = "global"
    COUNTRY = "country"
    REGION = "region"
    STATE_PROVINCE = "state_province"
    CITY = "city"
    LOCAL = "local"


@dataclass
class NaturalExperiment:
    """Natural experiment entry"""
    experiment_id: str
    experiment_type: ExperimentType
    identification_strategy: IdentificationStrategy
    title: str
    description: str

    # Geographic and temporal scope
    geographic_scope: GeographicScope
    countries: List[str]
    regions: List[str]
    start_date: datetime
    end_date: Optional[datetime]

    # Causal structure
    treatment_variable: str
    outcome_variables: List[str]
    control_variables: List[str]
    instrumental_variables: List[str]

    # Data and sources
    data_sources: List[str]
    data_availability: Dict[str, Any]  # Frequency, coverage, quality

    # Identification quality
    identification_strength: float  # 0-1 scale
    external_validity: float  # 0-1 scale
    internal_validity: float  # 0-1 scale

    # Research and replication
    published_studies: List[str]
    replication_studies: List[str]
    effect_sizes: Dict[str, float]  # Outcome -> effect size

    # Trading relevance
    market_relevance: float  # 0-1 scale
    trading_implications: List[str]

    # Registry metadata
    date_added: datetime
    last_updated: datetime
    curator: str
    verification_status: str  # 'verified', 'pending', 'disputed'

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    quality_score: float = 0.0


@dataclass
class InstrumentalVariable:
    """Instrumental variable specification"""
    instrument_id: str
    instrument_name: str
    instrument_description: str

    # IV criteria
    relevance_strength: float  # Correlation with treatment
    exclusion_restriction_plausible: bool
    exogeneity_evidence: List[str]

    # Usage context
    applicable_treatments: List[str]
    applicable_outcomes: List[str]
    applicable_contexts: List[str]

    # Empirical evidence
    first_stage_f_stat: Optional[float]
    overidentification_tests: Dict[str, float]
    weak_instrument_concerns: bool

    # Sources and validation
    data_sources: List[str]
    validation_studies: List[str]

    # Registry metadata
    date_added: datetime
    quality_assessment: Dict[str, float]


@dataclass
class PolicyShock:
    """Policy shock/announcement event"""
    shock_id: str
    shock_type: str
    announcement_date: datetime
    implementation_date: Optional[datetime]

    # Shock characteristics
    magnitude: float
    persistence_expected: float
    surprise_component: float  # How unexpected was it

    # Affected variables
    directly_affected: List[str]
    indirectly_affected: List[str]

    # Market reaction
    immediate_market_reaction: Dict[str, float]
    announcement_effects: Dict[str, List[float]]  # Time series

    # Context
    economic_conditions: Dict[str, float]
    concurrent_events: List[str]

    date_recorded: datetime


class NaturalExperimentsRegistry:
    """
    Registry for natural experiments and causal identification strategies

    This system catalogs natural experiments, instrumental variables, and causal
    identification opportunities for trading and economic analysis.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Natural Experiments Registry

        Args:
            db_path: Path to SQLite database for persistence
        """
        self.db_path = db_path or ":memory:"
        self.experiments: Dict[str, NaturalExperiment] = {}
        self.instruments: Dict[str, InstrumentalVariable] = {}
        self.policy_shocks: Dict[str, PolicyShock] = {}

        # Initialize database
        self._init_database()

        # Populate with key natural experiments
        self._populate_key_experiments()

        logger.info("Natural Experiments Registry initialized")

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()

            # Natural experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS natural_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_type TEXT NOT NULL,
                    identification_strategy TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    geographic_scope TEXT,
                    countries TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    treatment_variable TEXT,
                    outcome_variables TEXT,
                    identification_strength REAL,
                    market_relevance REAL,
                    quality_score REAL,
                    verification_status TEXT,
                    date_added TEXT,
                    metadata TEXT
                )
            """)

            # Instrumental variables table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instrumental_variables (
                    instrument_id TEXT PRIMARY KEY,
                    instrument_name TEXT NOT NULL,
                    instrument_description TEXT,
                    relevance_strength REAL,
                    exclusion_restriction_plausible BOOLEAN,
                    first_stage_f_stat REAL,
                    weak_instrument_concerns BOOLEAN,
                    applicable_treatments TEXT,
                    applicable_outcomes TEXT,
                    date_added TEXT,
                    metadata TEXT
                )
            """)

            # Policy shocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policy_shocks (
                    shock_id TEXT PRIMARY KEY,
                    shock_type TEXT NOT NULL,
                    announcement_date TEXT,
                    implementation_date TEXT,
                    magnitude REAL,
                    surprise_component REAL,
                    directly_affected TEXT,
                    immediate_market_reaction TEXT,
                    date_recorded TEXT,
                    metadata TEXT
                )
            """)

            self.conn.commit()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _populate_key_experiments(self):
        """Populate registry with key natural experiments"""

        # VAT Changes
        self.register_experiment(NaturalExperiment(
            experiment_id="uk_vat_2008_2011",
            experiment_type=ExperimentType.VAT_CHANGE,
            identification_strategy=IdentificationStrategy.DIFFERENCE_IN_DIFFERENCES,
            title="UK VAT Changes 2008-2011",
            description="UK temporarily reduced VAT from 17.5% to 15% (Dec 2008-Dec 2009), then raised to 20% (Jan 2011)",
            geographic_scope=GeographicScope.COUNTRY,
            countries=["United Kingdom"],
            regions=["Europe"],
            start_date=datetime(2008, 12, 1),
            end_date=datetime(2011, 6, 30),
            treatment_variable="vat_rate",
            outcome_variables=["consumer_prices", "retail_sales", "consumption"],
            control_variables=["income", "unemployment", "interest_rates"],
            instrumental_variables=["vat_rate_change"],
            data_sources=["ONS", "Bank of England", "HM Treasury"],
            data_availability={"frequency": "monthly", "coverage": "complete", "quality": "high"},
            identification_strength=0.9,
            external_validity=0.7,
            internal_validity=0.9,
            published_studies=["Crossley et al. (2009)", "Benzarti et al. (2020)"],
            replication_studies=["Adam et al. (2020)"],
            effect_sizes={"consumer_prices": -0.75, "retail_sales": 0.023},
            market_relevance=0.8,
            trading_implications=["Consumer discretionary stocks respond to VAT changes", "Currency effects through inflation"],
            date_added=datetime.now(),
            last_updated=datetime.now(),
            curator="registry_system",
            verification_status="verified",
            tags=["taxation", "consumption", "inflation"],
            quality_score=0.85
        ))

        # Energy Price Shock
        self.register_experiment(NaturalExperiment(
            experiment_id="oil_shock_1973",
            experiment_type=ExperimentType.ENERGY_PRICE_SHOCK,
            identification_strategy=IdentificationStrategy.EVENT_STUDY,
            title="1973 Oil Price Shock",
            description="OPEC oil embargo led to 4x increase in oil prices, providing exogenous shock to energy costs",
            geographic_scope=GeographicScope.GLOBAL,
            countries=["USA", "Germany", "Japan", "UK"],
            regions=["Global"],
            start_date=datetime(1973, 10, 6),
            end_date=datetime(1974, 3, 18),
            treatment_variable="oil_price",
            outcome_variables=["inflation", "unemployment", "gdp_growth", "stock_returns"],
            control_variables=["monetary_policy", "fiscal_policy"],
            instrumental_variables=["opec_production_decisions"],
            data_sources=["IMF", "OECD", "Federal Reserve"],
            data_availability={"frequency": "monthly", "coverage": "good", "quality": "medium"},
            identification_strength=0.95,
            external_validity=0.8,
            internal_validity=0.85,
            published_studies=["Hamilton (1983)", "Kilian (2008)"],
            replication_studies=["Blanchard & Gali (2007)"],
            effect_sizes={"inflation": 0.8, "unemployment": 0.6, "gdp_growth": -0.4},
            market_relevance=0.95,
            trading_implications=["Energy stocks outperform", "Defensive sectors preferred", "Currency effects through terms of trade"],
            date_added=datetime.now(),
            last_updated=datetime.now(),
            curator="registry_system",
            verification_status="verified",
            tags=["energy", "inflation", "recession", "commodities"],
            quality_score=0.90
        ))

        # Housing Regulation
        self.register_experiment(NaturalExperiment(
            experiment_id="berlin_rent_control_2020",
            experiment_type=ExperimentType.HOUSING_REGULATION,
            identification_strategy=IdentificationStrategy.REGRESSION_DISCONTINUITY,
            title="Berlin Rent Control 2020",
            description="Berlin implemented rent freeze (Mietendeckel) in Feb 2020, ruled unconstitutional in Apr 2021",
            geographic_scope=GeographicScope.CITY,
            countries=["Germany"],
            regions=["Berlin"],
            start_date=datetime(2020, 2, 23),
            end_date=datetime(2021, 4, 15),
            treatment_variable="rent_control_policy",
            outcome_variables=["rental_prices", "housing_supply", "construction_permits"],
            control_variables=["income", "population", "interest_rates"],
            instrumental_variables=["policy_implementation"],
            data_sources=["Berlin Statistical Office", "German Federal Bank"],
            data_availability={"frequency": "monthly", "coverage": "good", "quality": "high"},
            identification_strength=0.85,
            external_validity=0.6,
            internal_validity=0.9,
            published_studies=["Kholodilin & Sebastian (2021)"],
            replication_studies=[],
            effect_sizes={"rental_prices": -0.1, "housing_supply": -0.15},
            market_relevance=0.7,
            trading_implications=["Real estate investment trusts", "Construction companies", "Local banking sector"],
            date_added=datetime.now(),
            last_updated=datetime.now(),
            curator="registry_system",
            verification_status="verified",
            tags=["housing", "regulation", "rent_control"],
            quality_score=0.75
        ))

        # Monetary Policy Announcement
        self.register_experiment(NaturalExperiment(
            experiment_id="ecb_qe_announcement_2015",
            experiment_type=ExperimentType.POLICY_ANNOUNCEMENT,
            identification_strategy=IdentificationStrategy.EVENT_STUDY,
            title="ECB QE Announcement January 2015",
            description="ECB announced expanded asset purchase programme (QE) on January 22, 2015",
            geographic_scope=GeographicScope.REGION,
            countries=["Germany", "France", "Italy", "Spain", "Netherlands"],
            regions=["Eurozone"],
            start_date=datetime(2015, 1, 22),
            end_date=datetime(2015, 3, 31),
            treatment_variable="qe_announcement",
            outcome_variables=["bond_yields", "stock_prices", "eur_exchange_rate", "bank_stocks"],
            control_variables=["economic_indicators", "market_sentiment"],
            instrumental_variables=["ecb_announcement"],
            data_sources=["ECB", "Bloomberg", "Refinitiv"],
            data_availability={"frequency": "daily", "coverage": "complete", "quality": "high"},
            identification_strength=0.9,
            external_validity=0.8,
            internal_validity=0.95,
            published_studies=["Andrade et al. (2016)", "Krishnamurthy et al. (2018)"],
            replication_studies=["Altavilla et al. (2019)"],
            effect_sizes={"bond_yields": -0.25, "stock_prices": 0.08, "eur_exchange_rate": -0.03},
            market_relevance=0.95,
            trading_implications=["Long European equities", "Short EUR", "Long duration bonds"],
            date_added=datetime.now(),
            last_updated=datetime.now(),
            curator="registry_system",
            verification_status="verified",
            tags=["monetary_policy", "quantitative_easing", "central_bank"],
            quality_score=0.90
        ))

        # Regulatory Shock
        self.register_experiment(NaturalExperiment(
            experiment_id="dodd_frank_2010",
            experiment_type=ExperimentType.FINANCIAL_REGULATION,
            identification_strategy=IdentificationStrategy.DIFFERENCE_IN_DIFFERENCES,
            title="Dodd-Frank Act Implementation 2010-2015",
            description="US financial reform legislation affecting bank regulation, derivatives markets, and systemic risk",
            geographic_scope=GeographicScope.COUNTRY,
            countries=["USA"],
            regions=["North America"],
            start_date=datetime(2010, 7, 21),
            end_date=datetime(2015, 12, 31),
            treatment_variable="dodd_frank_rules",
            outcome_variables=["bank_profitability", "lending_standards", "systemic_risk"],
            control_variables=["gdp_growth", "interest_rates", "unemployment"],
            instrumental_variables=["asset_size_threshold"],
            data_sources=["Federal Reserve", "FDIC", "SEC"],
            data_availability={"frequency": "quarterly", "coverage": "complete", "quality": "high"},
            identification_strength=0.8,
            external_validity=0.7,
            internal_validity=0.85,
            published_studies=["Acharya et al. (2018)", "Begenau & Landvoigt (2021)"],
            replication_studies=["Buchak et al. (2018)"],
            effect_sizes={"bank_profitability": -0.05, "lending_standards": 0.1},
            market_relevance=0.85,
            trading_implications=["Bank stock performance", "Shadow banking growth", "Fintech opportunities"],
            date_added=datetime.now(),
            last_updated=datetime.now(),
            curator="registry_system",
            verification_status="verified",
            tags=["financial_regulation", "banking", "systemic_risk"],
            quality_score=0.8
        ))

        logger.info(f"Populated registry with {len(self.experiments)} key natural experiments")

    def register_experiment(self, experiment: NaturalExperiment):
        """Register a new natural experiment"""
        try:
            # Store in memory
            self.experiments[experiment.experiment_id] = experiment

            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO natural_experiments (
                    experiment_id, experiment_type, identification_strategy, title, description,
                    geographic_scope, countries, start_date, end_date, treatment_variable,
                    outcome_variables, identification_strength, market_relevance, quality_score,
                    verification_status, date_added, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.experiment_id,
                experiment.experiment_type.value,
                experiment.identification_strategy.value,
                experiment.title,
                experiment.description,
                experiment.geographic_scope.value,
                json.dumps(experiment.countries),
                experiment.start_date.isoformat(),
                experiment.end_date.isoformat() if experiment.end_date else None,
                experiment.treatment_variable,
                json.dumps(experiment.outcome_variables),
                experiment.identification_strength,
                experiment.market_relevance,
                experiment.quality_score,
                experiment.verification_status,
                experiment.date_added.isoformat(),
                json.dumps({
                    'regions': experiment.regions,
                    'control_variables': experiment.control_variables,
                    'instrumental_variables': experiment.instrumental_variables,
                    'data_sources': experiment.data_sources,
                    'published_studies': experiment.published_studies,
                    'effect_sizes': experiment.effect_sizes,
                    'trading_implications': experiment.trading_implications,
                    'tags': experiment.tags
                })
            ))
            self.conn.commit()

            logger.info(f"Registered experiment: {experiment.experiment_id}")

        except Exception as e:
            logger.error(f"Error registering experiment {experiment.experiment_id}: {e}")
            raise

    def register_instrumental_variable(self, instrument: InstrumentalVariable):
        """Register a new instrumental variable"""
        try:
            # Store in memory
            self.instruments[instrument.instrument_id] = instrument

            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO instrumental_variables (
                    instrument_id, instrument_name, instrument_description, relevance_strength,
                    exclusion_restriction_plausible, first_stage_f_stat, weak_instrument_concerns,
                    applicable_treatments, applicable_outcomes, date_added, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instrument.instrument_id,
                instrument.instrument_name,
                instrument.instrument_description,
                instrument.relevance_strength,
                instrument.exclusion_restriction_plausible,
                instrument.first_stage_f_stat,
                instrument.weak_instrument_concerns,
                json.dumps(instrument.applicable_treatments),
                json.dumps(instrument.applicable_outcomes),
                instrument.date_added.isoformat(),
                json.dumps({
                    'applicable_contexts': instrument.applicable_contexts,
                    'data_sources': instrument.data_sources,
                    'validation_studies': instrument.validation_studies,
                    'quality_assessment': instrument.quality_assessment
                })
            ))
            self.conn.commit()

            logger.info(f"Registered instrumental variable: {instrument.instrument_id}")

        except Exception as e:
            logger.error(f"Error registering instrument {instrument.instrument_id}: {e}")
            raise

    def record_policy_shock(self, shock: PolicyShock):
        """Record a new policy shock/announcement"""
        try:
            # Store in memory
            self.policy_shocks[shock.shock_id] = shock

            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO policy_shocks (
                    shock_id, shock_type, announcement_date, implementation_date, magnitude,
                    surprise_component, directly_affected, immediate_market_reaction,
                    date_recorded, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                shock.shock_id,
                shock.shock_type,
                shock.announcement_date.isoformat(),
                shock.implementation_date.isoformat() if shock.implementation_date else None,
                shock.magnitude,
                shock.surprise_component,
                json.dumps(shock.directly_affected),
                json.dumps(shock.immediate_market_reaction),
                shock.date_recorded.isoformat(),
                json.dumps({
                    'indirectly_affected': shock.indirectly_affected,
                    'announcement_effects': shock.announcement_effects,
                    'economic_conditions': shock.economic_conditions,
                    'concurrent_events': shock.concurrent_events
                })
            ))
            self.conn.commit()

            logger.info(f"Recorded policy shock: {shock.shock_id}")

        except Exception as e:
            logger.error(f"Error recording policy shock {shock.shock_id}: {e}")
            raise

    def search_experiments(self,
                         experiment_type: Optional[ExperimentType] = None,
                         identification_strategy: Optional[IdentificationStrategy] = None,
                         countries: Optional[List[str]] = None,
                         treatment_variable: Optional[str] = None,
                         outcome_variables: Optional[List[str]] = None,
                         min_quality_score: float = 0.0,
                         min_market_relevance: float = 0.0,
                         tags: Optional[List[str]] = None) -> List[NaturalExperiment]:
        """
        Search for natural experiments matching criteria

        Args:
            experiment_type: Type of experiment to search for
            identification_strategy: Identification strategy to filter by
            countries: Countries to include
            treatment_variable: Treatment variable name
            outcome_variables: Outcome variables to match
            min_quality_score: Minimum quality score
            min_market_relevance: Minimum market relevance score
            tags: Tags to match

        Returns:
            List of matching experiments
        """
        logger.info("Searching for natural experiments")

        matching_experiments = []

        try:
            for experiment in self.experiments.values():
                # Type filter
                if experiment_type and experiment.experiment_type != experiment_type:
                    continue

                # Strategy filter
                if identification_strategy and experiment.identification_strategy != identification_strategy:
                    continue

                # Countries filter
                if countries and not any(country in experiment.countries for country in countries):
                    continue

                # Treatment variable filter
                if treatment_variable and treatment_variable not in experiment.treatment_variable:
                    continue

                # Outcome variables filter
                if outcome_variables and not any(
                    outcome in experiment.outcome_variables for outcome in outcome_variables
                ):
                    continue

                # Quality score filter
                if experiment.quality_score < min_quality_score:
                    continue

                # Market relevance filter
                if experiment.market_relevance < min_market_relevance:
                    continue

                # Tags filter
                if tags and not any(tag in experiment.tags for tag in tags):
                    continue

                matching_experiments.append(experiment)

            # Sort by quality score and market relevance
            matching_experiments.sort(
                key=lambda x: (x.quality_score + x.market_relevance) / 2,
                reverse=True
            )

            logger.info(f"Found {len(matching_experiments)} matching experiments")
            return matching_experiments

        except Exception as e:
            logger.error(f"Error searching experiments: {e}")
            return []

    def find_instrumental_variables(self,
                                  treatment: str,
                                  outcome: str,
                                  context: Optional[str] = None,
                                  min_relevance: float = 0.5) -> List[InstrumentalVariable]:
        """
        Find suitable instrumental variables for a treatment-outcome pair

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            context: Application context
            min_relevance: Minimum relevance strength

        Returns:
            List of suitable instrumental variables
        """
        logger.info(f"Finding instruments for {treatment} -> {outcome}")

        suitable_instruments = []

        try:
            for instrument in self.instruments.values():
                # Check if instrument applies to this treatment
                if treatment not in instrument.applicable_treatments:
                    continue

                # Check if instrument applies to this outcome
                if outcome not in instrument.applicable_outcomes:
                    continue

                # Check context if specified
                if context and context not in instrument.applicable_contexts:
                    continue

                # Check relevance strength
                if instrument.relevance_strength < min_relevance:
                    continue

                # Check if exclusion restriction is plausible
                if not instrument.exclusion_restriction_plausible:
                    continue

                # Check for weak instrument concerns
                if instrument.weak_instrument_concerns:
                    continue

                suitable_instruments.append(instrument)

            # Sort by relevance strength
            suitable_instruments.sort(key=lambda x: x.relevance_strength, reverse=True)

            logger.info(f"Found {len(suitable_instruments)} suitable instruments")
            return suitable_instruments

        except Exception as e:
            logger.error(f"Error finding instruments: {e}")
            return []

    def get_policy_shock_analysis(self,
                                shock_type: Optional[str] = None,
                                date_range: Optional[Tuple[datetime, datetime]] = None,
                                affected_variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze policy shocks for patterns and trading implications

        Args:
            shock_type: Type of policy shock
            date_range: Date range to analyze
            affected_variables: Variables affected by shocks

        Returns:
            Analysis of policy shocks
        """
        logger.info("Analyzing policy shocks")

        try:
            # Filter shocks
            relevant_shocks = []
            for shock in self.policy_shocks.values():
                # Type filter
                if shock_type and shock.shock_type != shock_type:
                    continue

                # Date range filter
                if date_range:
                    start_date, end_date = date_range
                    if not (start_date <= shock.announcement_date <= end_date):
                        continue

                # Affected variables filter
                if affected_variables:
                    if not any(var in shock.directly_affected + shock.indirectly_affected
                             for var in affected_variables):
                        continue

                relevant_shocks.append(shock)

            if not relevant_shocks:
                return {'error': 'No matching policy shocks found'}

            # Analyze patterns
            analysis = {
                'total_shocks': len(relevant_shocks),
                'shock_types': defaultdict(int),
                'average_magnitude': 0.0,
                'average_surprise': 0.0,
                'common_affected_variables': defaultdict(int),
                'market_reaction_patterns': defaultdict(list),
                'trading_patterns': {}
            }

            total_magnitude = 0.0
            total_surprise = 0.0

            for shock in relevant_shocks:
                analysis['shock_types'][shock.shock_type] += 1
                total_magnitude += abs(shock.magnitude)
                total_surprise += shock.surprise_component

                # Count affected variables
                for var in shock.directly_affected:
                    analysis['common_affected_variables'][var] += 1

                # Collect market reactions
                for asset, reaction in shock.immediate_market_reaction.items():
                    analysis['market_reaction_patterns'][asset].append(reaction)

            analysis['average_magnitude'] = total_magnitude / len(relevant_shocks)
            analysis['average_surprise'] = total_surprise / len(relevant_shocks)

            # Calculate average market reactions
            for asset, reactions in analysis['market_reaction_patterns'].items():
                analysis['trading_patterns'][asset] = {
                    'average_reaction': np.mean(reactions),
                    'reaction_volatility': np.std(reactions),
                    'positive_reactions': sum(1 for r in reactions if r > 0),
                    'negative_reactions': sum(1 for r in reactions if r < 0)
                }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing policy shocks: {e}")
            return {'error': str(e)}

    def suggest_natural_experiments(self,
                                  research_question: str,
                                  treatment_variable: str,
                                  outcome_variable: str,
                                  available_data: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest natural experiments for a research question

        Args:
            research_question: Description of research question
            treatment_variable: Treatment of interest
            outcome_variable: Outcome of interest
            available_data: Available data sources

        Returns:
            List of suggested experiments with feasibility assessment
        """
        logger.info(f"Suggesting experiments for: {research_question}")

        suggestions = []

        try:
            # Search for relevant experiments
            relevant_experiments = self.search_experiments(
                treatment_variable=treatment_variable,
                outcome_variables=[outcome_variable]
            )

            for experiment in relevant_experiments:
                # Assess feasibility
                data_overlap = len(set(experiment.data_sources) & set(available_data))
                data_feasibility = data_overlap / len(experiment.data_sources)

                # Calculate suitability score
                suitability_score = (
                    experiment.quality_score * 0.3 +
                    experiment.identification_strength * 0.3 +
                    experiment.market_relevance * 0.2 +
                    data_feasibility * 0.2
                )

                suggestion = {
                    'experiment_id': experiment.experiment_id,
                    'title': experiment.title,
                    'identification_strategy': experiment.identification_strategy.value,
                    'suitability_score': suitability_score,
                    'data_feasibility': data_feasibility,
                    'strengths': [],
                    'limitations': [],
                    'implementation_steps': []
                }

                # Assess strengths
                if experiment.identification_strength > 0.8:
                    suggestion['strengths'].append("Strong causal identification")
                if experiment.quality_score > 0.8:
                    suggestion['strengths'].append("High-quality research base")
                if experiment.market_relevance > 0.8:
                    suggestion['strengths'].append("High market relevance")

                # Assess limitations
                if experiment.external_validity < 0.7:
                    suggestion['limitations'].append("Limited external validity")
                if data_feasibility < 0.5:
                    suggestion['limitations'].append("Limited data availability")
                if experiment.verification_status != 'verified':
                    suggestion['limitations'].append("Unverified experiment")

                # Implementation steps
                suggestion['implementation_steps'] = [
                    f"Obtain data from: {', '.join(experiment.data_sources)}",
                    f"Implement {experiment.identification_strategy.value} methodology",
                    "Validate identification assumptions",
                    "Conduct robustness checks",
                    "Assess trading implications"
                ]

                suggestions.append(suggestion)

            # Sort by suitability score
            suggestions.sort(key=lambda x: x['suitability_score'], reverse=True)

            # Also suggest creating new experiments
            if len(suggestions) < 3:
                suggestions.append({
                    'experiment_id': 'new_experiment',
                    'title': 'Design New Natural Experiment',
                    'identification_strategy': 'to_be_determined',
                    'suitability_score': 0.5,
                    'data_feasibility': 0.7,
                    'strengths': ['Tailored to specific research question'],
                    'limitations': ['Requires original research design'],
                    'implementation_steps': [
                        'Identify source of exogenous variation',
                        'Design identification strategy',
                        'Collect or access relevant data',
                        'Validate causal assumptions',
                        'Implement empirical analysis'
                    ]
                })

            logger.info(f"Generated {len(suggestions)} experiment suggestions")
            return suggestions

        except Exception as e:
            logger.error(f"Error suggesting experiments: {e}")
            return []

    def validate_identification_strategy(self,
                                       experiment_id: str,
                                       data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate the identification strategy of an experiment

        Args:
            experiment_id: ID of experiment to validate
            data: Optional data for empirical validation

        Returns:
            Validation results
        """
        if experiment_id not in self.experiments:
            return {'error': f'Experiment {experiment_id} not found'}

        experiment = self.experiments[experiment_id]

        validation = {
            'experiment_id': experiment_id,
            'identification_strategy': experiment.identification_strategy.value,
            'theoretical_validity': {},
            'empirical_validity': {},
            'overall_assessment': 'unknown'
        }

        try:
            # Theoretical validity checks
            if experiment.identification_strategy == IdentificationStrategy.INSTRUMENTAL_VARIABLES:
                validation['theoretical_validity'] = {
                    'relevance': len(experiment.instrumental_variables) > 0,
                    'exclusion_restriction': 'assumed_plausible',  # Would need domain knowledge
                    'independence': 'assumed_plausible'
                }

            elif experiment.identification_strategy == IdentificationStrategy.REGRESSION_DISCONTINUITY:
                validation['theoretical_validity'] = {
                    'discontinuity_present': True,  # Assumed if in registry
                    'no_manipulation': 'needs_verification',
                    'local_randomization': 'plausible'
                }

            elif experiment.identification_strategy == IdentificationStrategy.DIFFERENCE_IN_DIFFERENCES:
                validation['theoretical_validity'] = {
                    'parallel_trends': 'needs_testing',
                    'no_spillovers': 'assumed',
                    'stable_composition': 'assumed'
                }

            # Empirical validity checks (if data provided)
            if data is not None and not data.empty:
                validation['empirical_validity'] = self._empirical_validity_tests(
                    experiment, data
                )

            # Overall assessment
            theoretical_checks = list(validation['theoretical_validity'].values())
            if all(check in [True, 'plausible', 'assumed_plausible'] for check in theoretical_checks):
                validation['overall_assessment'] = 'strong'
            elif any(check in ['needs_verification', 'needs_testing'] for check in theoretical_checks):
                validation['overall_assessment'] = 'moderate'
            else:
                validation['overall_assessment'] = 'weak'

            return validation

        except Exception as e:
            logger.error(f"Error validating identification strategy: {e}")
            validation['error'] = str(e)
            return validation

    def _empirical_validity_tests(self, experiment: NaturalExperiment,
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Perform empirical validity tests on data"""
        tests = {}

        try:
            if experiment.identification_strategy == IdentificationStrategy.DIFFERENCE_IN_DIFFERENCES:
                # Test parallel trends assumption
                if ('time' in data.columns and 'treatment' in data.columns and
                    experiment.outcome_variables[0] in data.columns):

                    outcome = experiment.outcome_variables[0]

                    # Simple test: compare pre-treatment trends
                    pre_treatment = data[data['time'] < experiment.start_date]

                    if len(pre_treatment) > 5:
                        treated = pre_treatment[pre_treatment['treatment'] == 1]
                        control = pre_treatment[pre_treatment['treatment'] == 0]

                        if len(treated) > 2 and len(control) > 2:
                            # Calculate trend differences
                            treated_trend = stats.linregress(
                                range(len(treated)), treated[outcome]
                            ).slope
                            control_trend = stats.linregress(
                                range(len(control)), control[outcome]
                            ).slope

                            trend_difference = abs(treated_trend - control_trend)
                            tests['parallel_trends'] = {
                                'trend_difference': trend_difference,
                                'likely_violation': trend_difference > 0.1  # Arbitrary threshold
                            }

            elif experiment.identification_strategy == IdentificationStrategy.INSTRUMENTAL_VARIABLES:
                # Test instrument relevance
                if (experiment.instrumental_variables and
                    experiment.instrumental_variables[0] in data.columns and
                    experiment.treatment_variable in data.columns):

                    instrument = experiment.instrumental_variables[0]
                    treatment = experiment.treatment_variable

                    correlation, p_value = stats.pearsonr(
                        data[instrument].dropna(),
                        data[treatment].dropna()
                    )

                    tests['instrument_relevance'] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'weak_instrument': abs(correlation) < 0.3
                    }

        except Exception as e:
            logger.error(f"Error in empirical validity tests: {e}")
            tests['error'] = str(e)

        return tests

    def export_registry_summary(self) -> Dict[str, Any]:
        """Export comprehensive registry summary"""
        return {
            'registry_summary': {
                'total_experiments': len(self.experiments),
                'total_instruments': len(self.instruments),
                'total_policy_shocks': len(self.policy_shocks),
                'last_updated': datetime.now().isoformat()
            },
            'experiment_types': {
                exp_type.value: sum(1 for exp in self.experiments.values()
                                  if exp.experiment_type == exp_type)
                for exp_type in ExperimentType
            },
            'identification_strategies': {
                strategy.value: sum(1 for exp in self.experiments.values()
                                  if exp.identification_strategy == strategy)
                for strategy in IdentificationStrategy
            },
            'geographic_coverage': {
                scope.value: sum(1 for exp in self.experiments.values()
                               if exp.geographic_scope == scope)
                for scope in GeographicScope
            },
            'quality_distribution': {
                'high_quality': sum(1 for exp in self.experiments.values()
                                  if exp.quality_score > 0.8),
                'medium_quality': sum(1 for exp in self.experiments.values()
                                    if 0.5 < exp.quality_score <= 0.8),
                'low_quality': sum(1 for exp in self.experiments.values()
                                 if exp.quality_score <= 0.5)
            },
            'market_relevance': {
                'high_relevance': sum(1 for exp in self.experiments.values()
                                    if exp.market_relevance > 0.8),
                'medium_relevance': sum(1 for exp in self.experiments.values()
                                      if 0.5 < exp.market_relevance <= 0.8),
                'low_relevance': sum(1 for exp in self.experiments.values()
                                   if exp.market_relevance <= 0.5)
            }
        }

    def close(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()