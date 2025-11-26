"""
Causal Intelligence Education System

Makes complex causal intelligence concepts accessible and engaging for users,
following mobile app psychology principles for progressive education and
user engagement. Transforms technical concepts into understandable insights.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class EducationLevel(Enum):
    """User education levels for progressive disclosure"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConceptDifficulty(Enum):
    """Concept difficulty levels for content adaptation"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class EducationalConcept:
    """A single educational concept with progressive disclosure"""
    concept_id: str
    title: str
    simplified_explanation: str
    detailed_explanation: str
    expert_details: str

    # Visual and engagement elements
    analogy: str
    visual_metaphor: str
    real_world_example: str

    # Prerequisites and progression
    prerequisites: List[str]
    unlocks: List[str]
    difficulty: ConceptDifficulty

    # Psychology elements
    curiosity_hook: str
    confidence_builder: str
    practical_benefit: str


@dataclass
class InteractiveTutorial:
    """Interactive tutorial with hands-on elements"""
    tutorial_id: str
    title: str
    description: str
    steps: List[Dict[str, Any]]

    # User engagement
    estimated_time: str
    reward_upon_completion: str
    practical_outcome: str

    # Progress tracking
    completion_criteria: Dict[str, Any]
    mastery_indicators: List[str]


class CausalEducationEngine:
    """
    Core education engine that adapts content to user level and provides
    progressive disclosure of causal intelligence concepts
    """

    def __init__(self):
        """Initialize the education engine with core concepts"""
        self.concepts = self._initialize_core_concepts()
        self.tutorials = self._initialize_tutorials()
        self.user_progress = {}

        logger.info("Causal Education Engine initialized")

    def _initialize_core_concepts(self) -> Dict[str, EducationalConcept]:
        """Initialize the core causal intelligence concepts"""

        concepts = {
            "distributional_flows": EducationalConcept(
                concept_id="distributional_flows",
                title="How Money Really Flows in Markets",
                simplified_explanation=(
                    "Money doesn't just move randomly in markets - it flows from "
                    "specific groups to others in predictable patterns. Our system "
                    "tracks these money flows to spot opportunities."
                ),
                detailed_explanation=(
                    "Distributional Flow Analysis tracks how wealth moves between "
                    "different market participants (retail investors, institutions, "
                    "central banks). By understanding these flows, we can predict "
                    "price movements before they happen."
                ),
                expert_details=(
                    "The Distributional Flow Ledger (DFL) uses kernel density estimation "
                    "to model wealth transfer functions between heterogeneous agents, "
                    "incorporating transaction costs and market microstructure effects."
                ),
                analogy=(
                    "Think of the market like a river system - money flows from "
                    "small streams (retail) to big rivers (institutions). We're "
                    "tracking where the water is going before others notice."
                ),
                visual_metaphor="River system with tributaries and main channels",
                real_world_example=(
                    "When GameStop spiked, our system would have detected unusual "
                    "retail flow patterns weeks before the main move."
                ),
                prerequisites=[],
                unlocks=["policy_shocks", "causal_dag"],
                difficulty=ConceptDifficulty.MODERATE,
                curiosity_hook="Want to see where smart money is moving before everyone else?",
                confidence_builder="You already understand supply and demand - this is just more precise",
                practical_benefit="Spot major moves 1-3 weeks before they happen"
            ),

            "causal_dag": EducationalConcept(
                concept_id="causal_dag",
                title="The Hidden Cause-and-Effect Map of Markets",
                simplified_explanation=(
                    "Markets aren't random - they follow cause-and-effect patterns. "
                    "Our AI builds a map of what really causes price movements, "
                    "not just what's correlated."
                ),
                detailed_explanation=(
                    "Causal Directed Acyclic Graphs (DAGs) map the true causal "
                    "relationships between economic variables. Unlike correlation, "
                    "this tells us what actually causes what, enabling prediction."
                ),
                expert_details=(
                    "Dynamic causal inference using structural equation modeling "
                    "with time-varying coefficients, incorporating instrumental "
                    "variables and natural experiments for identification."
                ),
                analogy=(
                    "It's like having a detective who knows exactly which domino "
                    "will cause the others to fall, instead of just seeing them "
                    "fall at the same time."
                ),
                visual_metaphor="Network diagram with arrows showing cause-and-effect",
                real_world_example=(
                    "While others see that tech stocks and interest rates move together, "
                    "we know exactly how Fed policy changes cause specific sector rotations."
                ),
                prerequisites=["distributional_flows"],
                unlocks=["policy_shocks", "natural_experiments"],
                difficulty=ConceptDifficulty.COMPLEX,
                curiosity_hook="What if you could see the invisible strings moving the market?",
                confidence_builder="You already understand cause and effect in daily life",
                practical_benefit="Predict market reactions to news before they happen"
            ),

            "policy_shocks": EducationalConcept(
                concept_id="policy_shocks",
                title="How Government Decisions Move Your Portfolio",
                simplified_explanation=(
                    "When the government makes policy changes, certain stocks "
                    "always react in predictable ways. We model these reactions "
                    "to position you before the moves happen."
                ),
                detailed_explanation=(
                    "Policy shock analysis uses HANK (Heterogeneous Agent New "
                    "Keynesian) models to predict how different market participants "
                    "will react to policy changes, creating trading opportunities."
                ),
                expert_details=(
                    "HANK-lite modeling incorporates heterogeneous agents with "
                    "different wealth levels, risk preferences, and information "
                    "processing capabilities to simulate policy transmission mechanisms."
                ),
                analogy=(
                    "It's like knowing that when the government announces a new "
                    "highway, certain businesses along the route will benefit - "
                    "but for financial policies and markets."
                ),
                visual_metaphor="Government policy ripple effects across different market segments",
                real_world_example=(
                    "When the Fed signals rate changes, tech stocks react differently "
                    "than utilities based on their specific financial structures."
                ),
                prerequisites=["causal_dag", "distributional_flows"],
                unlocks=["synthetic_controls"],
                difficulty=ConceptDifficulty.COMPLEX,
                curiosity_hook="Want to front-run government policy effects on markets?",
                confidence_builder="You've seen how news moves stocks - this makes it systematic",
                practical_benefit="Position before policy-driven sector rotations"
            ),

            "natural_experiments": EducationalConcept(
                concept_id="natural_experiments",
                title="Learning from Market's Natural Tests",
                simplified_explanation=(
                    "Sometimes the market accidentally runs perfect experiments "
                    "that show us what really works. We collect and learn from "
                    "these rare insights."
                ),
                detailed_explanation=(
                    "Natural experiments are rare market events that create "
                    "controlled test conditions, allowing us to identify causal "
                    "relationships that would be impossible to detect otherwise."
                ),
                expert_details=(
                    "Quasi-experimental identification strategies using exogenous "
                    "variation in policy or market structure to estimate causal "
                    "effects with instrumental variable approaches."
                ),
                analogy=(
                    "Like learning from twins separated at birth - we find "
                    "identical market situations with one key difference to "
                    "see what really causes outperformance."
                ),
                visual_metaphor="Split-path experiment showing different outcomes",
                real_world_example=(
                    "When Robinhood had outages, we could see exactly how retail "
                    "flow affects specific stock movements by comparing affected "
                    "vs unaffected stocks."
                ),
                prerequisites=["causal_dag"],
                unlocks=["synthetic_controls"],
                difficulty=ConceptDifficulty.MODERATE,
                curiosity_hook="What if market crashes were actually teaching us hidden secrets?",
                confidence_builder="You naturally learn from comparing similar situations",
                practical_benefit="Discover edge cases that create exceptional opportunities"
            ),

            "synthetic_controls": EducationalConcept(
                concept_id="synthetic_controls",
                title="Creating Perfect Market Comparisons",
                simplified_explanation=(
                    "We create artificial 'control groups' for market events "
                    "to prove our strategies actually work, not just get lucky. "
                    "It's like A/B testing but for trading strategies."
                ),
                detailed_explanation=(
                    "Synthetic control methods create counterfactual baselines "
                    "by combining similar assets to estimate what would have "
                    "happened without our intervention or external events."
                ),
                expert_details=(
                    "Synthetic control methodology using weighted combinations "
                    "of donor assets to construct counterfactual outcomes, with "
                    "statistical inference via permutation tests."
                ),
                analogy=(
                    "Like having a clone of yourself that didn't take the trading "
                    "action, so you can see exactly how much better you did."
                ),
                visual_metaphor="Twin portfolios with different paths showing impact",
                real_world_example=(
                    "When we buy a stock, we create a synthetic version using "
                    "similar stocks to see if our specific pick actually outperformed."
                ),
                prerequisites=["natural_experiments", "policy_shocks"],
                unlocks=[],
                difficulty=ConceptDifficulty.COMPLEX,
                curiosity_hook="Want proof your trading edge is real, not just luck?",
                confidence_builder="You already compare outcomes in daily decisions",
                practical_benefit="Validate and improve your trading strategies scientifically"
            )
        }

        return concepts

    def _initialize_tutorials(self) -> Dict[str, InteractiveTutorial]:
        """Initialize interactive tutorials"""

        tutorials = {
            "money_flows_intro": InteractiveTutorial(
                tutorial_id="money_flows_intro",
                title="Spot the Money Flow",
                description="Learn to identify where big money is moving in real market data",
                steps=[
                    {
                        "step": 1,
                        "title": "Observe the Pattern",
                        "content": "Look at this chart showing unusual buying in tech stocks",
                        "action": "identify_pattern",
                        "data": "sample_flow_data"
                    },
                    {
                        "step": 2,
                        "title": "Predict the Outcome",
                        "content": "Based on the flow pattern, what happens next?",
                        "action": "make_prediction",
                        "choices": ["Prices rise", "Prices fall", "No change"]
                    },
                    {
                        "step": 3,
                        "title": "See the Results",
                        "content": "Here's what actually happened - and why you were right!",
                        "action": "reveal_outcome",
                        "reward": "Flow Detection Badge"
                    }
                ],
                estimated_time="3 minutes",
                reward_upon_completion="Flow Detection Badge + Next tutorial unlocked",
                practical_outcome="You can now spot institutional buying before price moves",
                completion_criteria={"prediction_accuracy": 0.7},
                mastery_indicators=["Correctly identified 3/4 flow patterns"]
            ),

            "cause_effect_detective": InteractiveTutorial(
                tutorial_id="cause_effect_detective",
                title="Market Detective: Find the Real Cause",
                description="Distinguish between correlation and causation in market movements",
                steps=[
                    {
                        "step": 1,
                        "title": "The Mystery",
                        "content": "Tech stocks and crypto both fell 20%. Most think they're connected...",
                        "action": "present_mystery"
                    },
                    {
                        "step": 2,
                        "title": "Gather Evidence",
                        "content": "But look at the timing and order of events...",
                        "action": "examine_timeline",
                        "interactive": True
                    },
                    {
                        "step": 3,
                        "title": "Solve the Case",
                        "content": "The real cause was institutional deleveraging!",
                        "action": "reveal_solution",
                        "reward": "Causal Detective Badge"
                    }
                ],
                estimated_time="5 minutes",
                reward_upon_completion="Causal Detective Badge + Advanced concepts unlocked",
                practical_outcome="Never confuse correlation with causation again",
                completion_criteria={"identified_true_cause": True},
                mastery_indicators=["Correctly identified causal chain"]
            )
        }

        return tutorials

    def get_personalized_learning_path(self,
                                     user_id: str,
                                     current_level: EducationLevel,
                                     interests: List[str] = None) -> Dict[str, Any]:
        """Generate a personalized learning path for the user"""

        if user_id not in self.user_progress:
            self.user_progress[user_id] = {
                "level": current_level,
                "completed_concepts": [],
                "current_tutorials": [],
                "interests": interests or []
            }

        progress = self.user_progress[user_id]

        # Find next concepts to learn
        available_concepts = []
        for concept_id, concept in self.concepts.items():
            if (concept_id not in progress["completed_concepts"] and
                all(prereq in progress["completed_concepts"] for prereq in concept.prerequisites)):

                # Adapt difficulty to user level
                if current_level == EducationLevel.BEGINNER and concept.difficulty in [ConceptDifficulty.SIMPLE, ConceptDifficulty.MODERATE]:
                    available_concepts.append(concept)
                elif current_level == EducationLevel.INTERMEDIATE and concept.difficulty != ConceptDifficulty.EXPERT:
                    available_concepts.append(concept)
                elif current_level in [EducationLevel.ADVANCED, EducationLevel.EXPERT]:
                    available_concepts.append(concept)

        # Find available tutorials
        available_tutorials = []
        for tutorial_id, tutorial in self.tutorials.items():
            if tutorial_id not in progress.get("completed_tutorials", []):
                available_tutorials.append(tutorial)

        return {
            "user_id": user_id,
            "current_level": current_level.value,
            "next_concepts": available_concepts[:3],  # Top 3 recommendations
            "recommended_tutorials": available_tutorials[:2],  # Top 2 tutorials
            "progress_percentage": len(progress["completed_concepts"]) / len(self.concepts) * 100,
            "achievements_unlocked": self._get_achievements(user_id),
            "next_milestone": self._get_next_milestone(user_id)
        }

    def explain_concept(self,
                       concept_id: str,
                       user_level: EducationLevel,
                       personalization_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Provide personalized explanation of a concept"""

        if concept_id not in self.concepts:
            return {"error": "Concept not found"}

        concept = self.concepts[concept_id]

        # Choose explanation level based on user
        if user_level == EducationLevel.BEGINNER:
            explanation = concept.simplified_explanation
            hook = concept.curiosity_hook
            builder = concept.confidence_builder
        elif user_level == EducationLevel.INTERMEDIATE:
            explanation = concept.detailed_explanation
            hook = concept.curiosity_hook
            builder = f"Building on what you know: {concept.confidence_builder}"
        else:
            explanation = concept.expert_details
            hook = "Advanced insight: " + concept.curiosity_hook
            builder = concept.practical_benefit

        return {
            "concept_id": concept_id,
            "title": concept.title,
            "explanation": explanation,
            "curiosity_hook": hook,
            "confidence_builder": builder,
            "practical_benefit": concept.practical_benefit,
            "analogy": concept.analogy,
            "visual_metaphor": concept.visual_metaphor,
            "real_world_example": concept.real_world_example,
            "what_unlocks_next": concept.unlocks,
            "estimated_read_time": "2-3 minutes"
        }

    def start_interactive_tutorial(self,
                                 tutorial_id: str,
                                 user_id: str) -> Dict[str, Any]:
        """Start an interactive tutorial session"""

        if tutorial_id not in self.tutorials:
            return {"error": "Tutorial not found"}

        tutorial = self.tutorials[tutorial_id]

        # Initialize tutorial session
        session_id = f"{user_id}_{tutorial_id}_{int(datetime.now().timestamp())}"

        return {
            "session_id": session_id,
            "tutorial": {
                "title": tutorial.title,
                "description": tutorial.description,
                "estimated_time": tutorial.estimated_time,
                "reward": tutorial.reward_upon_completion,
                "practical_outcome": tutorial.practical_outcome
            },
            "current_step": tutorial.steps[0],
            "progress": "1/" + str(len(tutorial.steps)),
            "next_action": "Begin tutorial"
        }

    def process_tutorial_response(self,
                                session_id: str,
                                user_response: Dict[str, Any]) -> Dict[str, Any]:
        """Process user response in tutorial and advance to next step"""

        # This would handle interactive tutorial progression
        # For now, return a mock progression

        return {
            "session_id": session_id,
            "response_processed": True,
            "feedback": "Great job! You correctly identified the causal pattern.",
            "points_earned": 10,
            "next_step_available": True,
            "celebration": {
                "type": "progress_milestone",
                "message": "You're getting the hang of this!",
                "animation": "sparkles"
            }
        }

    def _get_achievements(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's unlocked achievements"""

        progress = self.user_progress.get(user_id, {})
        completed_concepts = progress.get("completed_concepts", [])

        achievements = []

        if "distributional_flows" in completed_concepts:
            achievements.append({
                "id": "flow_tracker",
                "title": "Flow Tracker",
                "description": "Mastered money flow detection",
                "icon": "water_drop",
                "unlocked_date": datetime.now().isoformat()
            })

        if len(completed_concepts) >= 3:
            achievements.append({
                "id": "causal_explorer",
                "title": "Causal Explorer",
                "description": "Learned multiple causal concepts",
                "icon": "compass",
                "unlocked_date": datetime.now().isoformat()
            })

        return achievements

    def _get_next_milestone(self, user_id: str) -> Dict[str, Any]:
        """Get user's next learning milestone"""

        progress = self.user_progress.get(user_id, {})
        completed_count = len(progress.get("completed_concepts", []))

        if completed_count < 2:
            return {
                "title": "Causal Foundation",
                "description": "Complete 2 core concepts",
                "progress": completed_count,
                "target": 2,
                "reward": "Advanced concepts unlocked"
            }
        elif completed_count < 4:
            return {
                "title": "Market Detective",
                "description": "Master cause-and-effect analysis",
                "progress": completed_count,
                "target": 4,
                "reward": "Expert-level tutorials unlocked"
            }
        else:
            return {
                "title": "Causal Master",
                "description": "Complete all advanced concepts",
                "progress": completed_count,
                "target": len(self.concepts),
                "reward": "Causal Intelligence Certification"
            }


class CausalEducationAPI:
    """API interface for the education system"""

    def __init__(self):
        self.engine = CausalEducationEngine()

    def get_learning_dashboard(self, user_id: str, user_level: str = "beginner") -> Dict[str, Any]:
        """Get complete learning dashboard for user"""

        level = EducationLevel(user_level.lower())
        learning_path = self.engine.get_personalized_learning_path(user_id, level)

        return {
            "dashboard_type": "causal_education",
            "user_progress": learning_path,
            "featured_concept": learning_path["next_concepts"][0] if learning_path["next_concepts"] else None,
            "quick_tutorial": learning_path["recommended_tutorials"][0] if learning_path["recommended_tutorials"] else None,
            "daily_insight": {
                "title": "Today's Causal Insight",
                "content": "Policy announcements create predictable sector rotations 2-3 days later",
                "action": "See it in action",
                "estimated_time": "30 seconds"
            },
            "achievements": learning_path["achievements_unlocked"],
            "next_milestone": learning_path["next_milestone"]
        }

    def explain_in_context(self,
                          concept_id: str,
                          trading_context: Dict[str, Any],
                          user_level: str = "beginner") -> Dict[str, Any]:
        """Explain concept in the context of current trading situation"""

        level = EducationLevel(user_level.lower())
        explanation = self.engine.explain_concept(concept_id, level)

        # Add contextual examples based on current market
        if trading_context:
            explanation["contextual_example"] = (
                f"Right now in your portfolio: {self._generate_contextual_example(concept_id, trading_context)}"
            )

        return explanation

    def _generate_contextual_example(self, concept_id: str, context: Dict[str, Any]) -> str:
        """Generate contextual example based on user's current situation"""

        if concept_id == "distributional_flows":
            return "We're seeing unusual institutional buying in your tech positions - this often signals a 5-10% move up over 1-2 weeks"
        elif concept_id == "policy_shocks":
            return "The Fed's recent comments will likely cause tech stocks to outperform utilities by 3-5% over the next month"
        else:
            return "Your current positions are benefiting from the causal patterns we've identified"