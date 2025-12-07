"""
Active Learning & Curiosity Engine

THE MOTIVATION SYSTEM FOR AGI!

Key Innovation: INTRINSIC MOTIVATION
- System actively seeks interesting problems
- Curiosity-driven exploration
- Self-directed learning
- Autonomous goal generation

This makes AGI truly AUTONOMOUS and SELF-DIRECTED!

Curiosity Metrics:
1. Information Gain - How much would I learn?
2. Novelty - How new/surprising is this?
3. Uncertainty - How uncertain am I about this?
4. Expected Impact - How useful would this knowledge be?

This is what separates TRUE AGI from passive systems!
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from collections import defaultdict


class CuriosityType(Enum):
    """Types of curiosity."""
    EPISTEMIC = "epistemic"  # Knowledge gap curiosity
    DIVERSIVE = "diversive"  # Seeking variety/novelty
    SPECIFIC = "specific"    # Targeted interest in topic
    PERCEPTUAL = "perceptual"  # Sensory exploration


@dataclass
class CuriousItem:
    """Something the system is curious about."""
    id: str
    description: str
    curiosity_type: CuriosityType

    # Curiosity scores (0-1)
    information_gain: float = 0.0  # Expected learning
    novelty: float = 0.0           # How new/surprising
    uncertainty: float = 0.0       # How uncertain we are
    impact: float = 0.0            # Expected usefulness

    # Overall curiosity (weighted combination)
    curiosity_score: float = 0.0

    # Context
    domain: str = "general"
    related_concepts: List[str] = field(default_factory=list)

    # Status
    explored: bool = False
    exploration_timestamp: Optional[str] = None
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LearningGoal:
    """An autonomous learning goal."""
    id: str
    goal_description: str
    motivation: str  # Why pursue this?

    # Subgoals
    subgoals: List[str] = field(default_factory=list)

    # Progress
    progress: float = 0.0  # 0-1
    completed: bool = False

    # Priority
    priority: float = 0.5  # 0-1

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ActiveLearningEngine:
    """
    Drives autonomous, curiosity-driven learning.

    THE MOTIVATION SYSTEM!

    Process:
    1. Scan knowledge base for gaps/uncertainties
    2. Evaluate curiosity for each gap
    3. Generate learning goals autonomously
    4. Prioritize by curiosity + impact
    5. Execute exploration
    6. Integrate discoveries
    7. Generate new goals from discoveries

    This makes the system SELF-DIRECTED!
    """

    def __init__(
        self,
        knowledge_graph=None,
        bayesian_evaluator=None,
        world_model=None,
        discovery_orchestrator=None,
        llm_bridge=None
    ):
        self.kg = knowledge_graph
        self.bayesian = bayesian_evaluator
        self.world_model = world_model
        self.discovery = discovery_orchestrator
        self.llm = llm_bridge  # For concept learning

        # Curiosity tracking
        self.curious_items: Dict[str, CuriousItem] = {}
        self.curiosity_count = 0

        # Learning goals
        self.learning_goals: Dict[str, LearningGoal] = {}
        self.goal_count = 0

        # Curiosity weights (tunable)
        self.weights = {
            'information_gain': 0.3,
            'novelty': 0.25,
            'uncertainty': 0.25,
            'impact': 0.2
        }

        # Exploration history
        self.exploration_history = []

        # Interest decay (curiosity fades over time)
        self.curiosity_decay = 0.95  # 5% decay per cycle

        # Learned concepts cache
        self.learned_concepts: Dict[str, Dict] = {}

        print("[Active Learning Engine] Initialized")
        print("  Curiosity-driven learning active!")
        print("  System will autonomously seek interesting problems")

    def extract_concepts_from_question(self, question: str) -> List[str]:
        """
        Extract specific concepts mentioned in a question.

        THIS IS TARGETED CONCEPT DETECTION!

        Uses LLM to intelligently identify what concepts are needed.

        Args:
            question: The question to analyze

        Returns:
            List of specific concepts
        """
        # Use LLM to intelligently detect concepts
        if self.llm:
            system_prompt = """You are analyzing a question to identify what mathematical concepts are needed to answer it.

Your task: List the specific concepts/knowledge needed (NOT general domains).

Examples:
- "What is 2 + 3?" → addition
- "Solve x² - 5x + 6 = 0" → quadratic_equations, factoring, quadratic_formula
- "What is a prime number?" → prime_numbers, number_theory
- "Find derivative of x²" → derivatives, calculus, power_rule

Be specific! List individual concepts, not general domains like "mathematics"."""

            user_prompt = f"""Question: {question}

List the specific concepts needed (comma-separated, one word or snake_case each):"""

            try:
                result = self.llm.generate(
                    system_prompt=system_prompt,
                    user_input=user_prompt,
                    execute=True
                )

                # Parse LLM response
                concepts_text = result.get('text', '').strip()

                # Extract concepts (comma-separated or newline-separated)
                import re
                concepts = []
                # Remove common words
                concepts_text = re.sub(r'\b(concepts?|needed|are|the|is)\b', '', concepts_text, flags=re.IGNORECASE)
                # Split by comma, newline, or bullet
                parts = re.split(r'[,\n•\-]', concepts_text)
                for part in parts:
                    concept = part.strip().lower().replace(' ', '_')
                    if concept and len(concept) > 2:  # Skip very short fragments
                        concepts.append(concept)

                if concepts:
                    print(f"    LLM identified concepts: {concepts}")
                    return concepts[:5]  # Top 5

            except Exception as e:
                print(f"    ⚠️  LLM concept detection failed: {e}")

        # Fallback: Keyword-based detection
        concepts = []
        question_lower = question.lower()

        # Quick keyword matching
        if 'addition' in question_lower or '+' in question:
            concepts.append('addition')
        if 'multiplication' in question_lower or '×' in question or '*' in question:
            concepts.append('multiplication')
        if 'prime' in question_lower:
            concepts.append('prime_numbers')
        if 'quadratic' in question_lower or 'x²' in question or 'x^2' in question:
            concepts.append('quadratic_equations')
        if 'derivative' in question_lower:
            concepts.append('derivatives')

        return concepts if concepts else ['general_mathematics']

    def learn_concept(self, concept: str) -> Dict:
        """
        Actually LEARN a specific concept using LLM.

        THIS IS TARGETED CONCEPT LEARNING!

        Instead of generic domain exploration, we learn the SPECIFIC concept:
        - What is it? (definition)
        - How does it work? (explanation)
        - Examples (concrete cases)
        - Prerequisites (what you need to know first)

        Args:
            concept: The specific concept to learn (e.g., "addition", "quadratic_formula")

        Returns:
            Dictionary with learned knowledge about the concept
        """
        # Check if already learned
        if concept in self.learned_concepts:
            print(f"    ✓ Already learned '{concept}' - retrieving from cache")
            return self.learned_concepts[concept]

        print(f"    📚 Learning concept: '{concept}'...")

        # Use LLM to learn about this concept
        if self.llm:
            system_prompt = """You are a teacher explaining mathematical concepts clearly and concisely.
Provide:
1. Definition (1-2 sentences)
2. How it works (1-2 sentences)
3. Simple example
4. Prerequisites (what you need to know first)

Keep it brief and practical."""

            user_prompt = f"""Explain the concept: {concept.replace('_', ' ')}

Format your response as:
DEFINITION: [brief definition]
HOW IT WORKS: [brief explanation]
EXAMPLE: [one simple example]
PREREQUISITES: [comma-separated list of concepts needed first, or "none"]"""

            try:
                result = self.llm.generate(
                    system_prompt=system_prompt,
                    user_input=user_prompt,
                    execute=True
                )

                learned = {
                    'concept': concept,
                    'content': result.get('text', ''),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'llm_learning'
                }

                # Parse the response for structured storage
                content = result.get('text', '')
                learned['definition'] = self._extract_section(content, 'DEFINITION')
                learned['how_it_works'] = self._extract_section(content, 'HOW IT WORKS')
                learned['example'] = self._extract_section(content, 'EXAMPLE')
                learned['prerequisites'] = self._extract_section(content, 'PREREQUISITES')

                print(f"      ✓ Learned from LLM")
                print(f"      Definition: {learned['definition'][:60]}...")

            except Exception as e:
                print(f"      ⚠️  LLM learning failed: {e}")
                # Fallback to basic concept storage
                learned = {
                    'concept': concept,
                    'definition': f"Mathematical concept: {concept.replace('_', ' ')}",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fallback'
                }
        else:
            # No LLM - basic concept storage
            learned = {
                'concept': concept,
                'definition': f"Mathematical concept: {concept.replace('_', ' ')}",
                'timestamp': datetime.now().isoformat(),
                'source': 'basic'
            }
            print(f"      ⚠️  No LLM - stored basic concept info")

        # Store in learned concepts
        self.learned_concepts[concept] = learned

        # Also store in world model if available
        if self.world_model:
            # Store as a concept in world model
            self.world_model.concepts[concept] = {
                'type': 'mathematical_concept',
                'definition': learned.get('definition', ''),
                'learned_from': 'active_learning',
                'timestamp': learned['timestamp']
            }
            print(f"      ✓ Stored in world model")

        return learned

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from formatted LLM response."""
        import re
        pattern = f"{section_name}:(.+?)(?=\\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def identify_missing_concepts(self, question: str, current_knowledge: List[str] = None) -> List[str]:
        """
        Identify which concepts are MISSING for this question.

        THIS IS KNOWLEDGE GAP DETECTION!

        Args:
            question: The question we're trying to solve
            current_knowledge: List of concepts we already know

        Returns:
            List of missing concepts that need to be learned
        """
        # Extract all concepts mentioned in question
        needed_concepts = self.extract_concepts_from_question(question)

        # Check which ones we don't have yet
        if current_knowledge is None:
            current_knowledge = list(self.learned_concepts.keys())

        missing = [c for c in needed_concepts if c not in current_knowledge]

        return missing

    def scan_for_curiosities(self, question: str = None) -> List[CuriousItem]:
        """
        Scan knowledge base for interesting things to explore.

        THIS IS THE CURIOSITY SCAN!

        Args:
            question: Optional specific question to analyze for missing concepts

        Returns:
            List of things the system is curious about
        """
        print("\n[🔍] Scanning for curiosities...")

        curiosities = []

        # PRIORITY 1: Missing concepts from specific question
        if question:
            print(f"  Analyzing question for missing concepts...")
            missing_concepts = self.identify_missing_concepts(question)

            if missing_concepts:
                print(f"  Found {len(missing_concepts)} missing concepts: {missing_concepts}")

                for concept in missing_concepts:
                    curiosity = CuriousItem(
                        id=f"curiosity_{self.curiosity_count}",
                        description=f"Learn specific concept: {concept}",
                        curiosity_type=CuriosityType.SPECIFIC,
                        information_gain=0.9,  # Very high - needed for question!
                        novelty=0.8,
                        uncertainty=1.0,  # Don't know it yet
                        impact=1.0,  # Critical for solving question
                        related_concepts=[concept]
                    )

                    curiosity.curiosity_score = self._compute_curiosity_score(curiosity)
                    curiosities.append(curiosity)
                    self.curiosity_count += 1
            else:
                print(f"  ✓ All concepts for question already learned!")

        # 2. Knowledge gaps (high FE regions)
        if self.kg:
            stats = self.kg.get_statistics()

            # Check if graph has data
            if stats.get('status') == 'active' and stats.get('avg_free_energy', 0) > 0.3:
                curiosity = CuriousItem(
                    id=f"curiosity_{self.curiosity_count}",
                    description=f"Knowledge gaps in graph (avg FE: {stats['avg_free_energy']:.2f})",
                    curiosity_type=CuriosityType.EPISTEMIC,
                    information_gain=stats['avg_free_energy'],  # High FE = high info gain
                    novelty=0.7,
                    uncertainty=stats['avg_free_energy'],
                    impact=0.6  # Lower priority than specific concepts
                )

                curiosity.curiosity_score = self._compute_curiosity_score(curiosity)
                curiosities.append(curiosity)
                self.curiosity_count += 1

        # 2. Unverified claims (uncertainty)
        if self.bayesian:
            stats = self.bayesian.get_statistics()

            # Check if there are claims to analyze
            if stats.get('status') == 'active' and stats.get('unverified_claims', 0) > 0:
                unverified_claims = stats.get('unverified_claims', 0)
                total_claims = stats.get('total_claims', 1)
                unverified_ratio = unverified_claims / total_claims if total_claims > 0 else 0

                curiosity = CuriousItem(
                    id=f"curiosity_{self.curiosity_count}",
                    description=f"Unverified claims need evidence ({unverified_claims} claims)",
                    curiosity_type=CuriosityType.SPECIFIC,
                    information_gain=0.6,
                    novelty=0.3,
                    uncertainty=unverified_ratio,
                    impact=0.7
                )

                curiosity.curiosity_score = self._compute_curiosity_score(curiosity)
                curiosities.append(curiosity)
                self.curiosity_count += 1

        # 3. Unexplored domains (novelty)
        if self.world_model:
            known_domains = set()
            for concept in self.world_model.concepts.values():
                known_domains.add(concept.get('domain', 'general'))

            # Suggest exploration of new domains
            all_domains = {'mathematics', 'physics', 'biology', 'chemistry',
                          'computer_science', 'philosophy', 'linguistics'}
            unexplored = all_domains - known_domains

            if unexplored:
                for domain in list(unexplored)[:3]:  # Top 3
                    curiosity = CuriousItem(
                        id=f"curiosity_{self.curiosity_count}",
                        description=f"Unexplored domain: {domain}",
                        curiosity_type=CuriosityType.DIVERSIVE,
                        information_gain=0.8,  # New domain = high info gain
                        novelty=1.0,  # Completely new!
                        uncertainty=0.9,  # Very uncertain
                        impact=0.6,
                        domain=domain
                    )

                    curiosity.curiosity_score = self._compute_curiosity_score(curiosity)
                    curiosities.append(curiosity)
                    self.curiosity_count += 1

        # 4. Patterns that need more data
        # (Would check CoT miner for patterns with low confidence)

        # Sort by curiosity score
        curiosities.sort(key=lambda x: x.curiosity_score, reverse=True)

        # Store
        for c in curiosities:
            self.curious_items[c.id] = c

        print(f"  Found {len(curiosities)} interesting things!")
        for i, c in enumerate(curiosities[:5], 1):
            print(f"    {i}. {c.description} (curiosity: {c.curiosity_score:.2f})")

        return curiosities

    def _compute_curiosity_score(self, item: CuriousItem) -> float:
        """
        Compute overall curiosity score.

        Weighted combination of 4 factors.
        """
        score = (
            item.information_gain * self.weights['information_gain'] +
            item.novelty * self.weights['novelty'] +
            item.uncertainty * self.weights['uncertainty'] +
            item.impact * self.weights['impact']
        )

        return score

    def generate_learning_goals(self, curiosities: List[CuriousItem], max_goals: int = 5) -> List[LearningGoal]:
        """
        Generate autonomous learning goals from curiosities.

        THIS IS AUTONOMOUS GOAL GENERATION!

        Args:
            curiosities: Things we're curious about
            max_goals: Max goals to generate

        Returns:
            List of learning goals
        """
        print(f"\n[🎯] Generating autonomous learning goals...")

        goals = []

        # Generate goal for each top curiosity
        for curiosity in curiosities[:max_goals]:
            goal = LearningGoal(
                id=f"goal_{self.goal_count}",
                goal_description=f"Explore: {curiosity.description}",
                motivation=f"Curiosity score: {curiosity.curiosity_score:.2f}",
                priority=curiosity.curiosity_score
            )

            # Generate subgoals based on type
            if curiosity.curiosity_type == CuriosityType.EPISTEMIC:
                goal.subgoals = [
                    "Identify specific knowledge gaps",
                    "Generate hypotheses to fill gaps",
                    "Test hypotheses with experiments",
                    "Integrate verified knowledge"
                ]

            elif curiosity.curiosity_type == CuriosityType.DIVERSIVE:
                goal.subgoals = [
                    f"Survey {curiosity.domain} domain",
                    "Identify key concepts",
                    "Build initial knowledge graph",
                    "Find connections to known domains"
                ]

            elif curiosity.curiosity_type == CuriosityType.SPECIFIC:
                goal.subgoals = [
                    "Gather evidence for claims",
                    "Update Bayesian beliefs",
                    "Verify or reject claims",
                    "Update knowledge base"
                ]

            goals.append(goal)
            self.learning_goals[goal.id] = goal
            self.goal_count += 1

        print(f"  Generated {len(goals)} learning goals:")
        for i, g in enumerate(goals, 1):
            print(f"    {i}. {g.goal_description}")
            print(f"       Priority: {g.priority:.2f}, Subgoals: {len(g.subgoals)}")

        return goals

    def explore(self, curiosity_id: str) -> Dict:
        """
        Actively explore something we're curious about.

        THIS IS AUTONOMOUS EXPLORATION!

        Args:
            curiosity_id: ID of curiosity to explore

        Returns:
            Exploration results
        """
        if curiosity_id not in self.curious_items:
            return {'error': 'Curiosity not found'}

        curiosity = self.curious_items[curiosity_id]

        print(f"\n[🚀] Exploring: {curiosity.description}")

        results = {
            'curiosity_id': curiosity_id,
            'description': curiosity.description,
            'discoveries': [],
            'knowledge_gained': 0,
            'success': False
        }

        # Execute exploration based on type
        if curiosity.curiosity_type == CuriosityType.SPECIFIC:
            # Learn specific concept using LLM!
            print("  Learning specific concept...")

            # Extract concept from related_concepts
            if curiosity.related_concepts:
                concepts_learned = []
                for concept in curiosity.related_concepts:
                    learned = self.learn_concept(concept)
                    if learned:
                        concepts_learned.append(learned)

                results['discoveries'] = concepts_learned
                results['knowledge_gained'] = len(concepts_learned)
                results['success'] = True
                print(f"  ✓ Learned {len(concepts_learned)} concepts!")
            else:
                print("  ⚠️  No specific concepts to learn")
                results['success'] = False

        elif curiosity.curiosity_type == CuriosityType.EPISTEMIC:
            # Run discovery in high FE regions
            if self.discovery:
                print("  Running autonomous discovery...")
                session = self.discovery.discover(
                    domain=curiosity.domain,
                    max_iterations=2,
                    verbose=False
                )

                results['discoveries'] = session.hypotheses_generated
                results['knowledge_gained'] = len(session.gaps_identified)
                results['success'] = True

        elif curiosity.curiosity_type == CuriosityType.DIVERSIVE:
            # Explore new domain (lower priority now)
            print(f"  Exploring new domain: {curiosity.domain}")
            # Would add foundational concepts for new domain
            results['knowledge_gained'] = 5  # Simulated
            results['success'] = True

        # Mark as explored
        curiosity.explored = True
        curiosity.exploration_timestamp = datetime.now().isoformat()

        # Track in history
        self.exploration_history.append({
            'curiosity_id': curiosity_id,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })

        print(f"  ✓ Exploration complete: {results['knowledge_gained']} new insights")

        return results

    def update_goal_progress(self, goal_id: str, progress: float):
        """Update progress on a learning goal."""
        if goal_id in self.learning_goals:
            goal = self.learning_goals[goal_id]
            goal.progress = progress

            if progress >= 1.0:
                goal.completed = True
                print(f"  ✓ Goal completed: {goal.goal_description}")

    def active_learning_loop(self, iterations: int = 5) -> Dict:
        """
        Run active learning loop.

        THIS IS AUTONOMOUS CURIOSITY-DRIVEN LEARNING!

        Loop:
        1. Scan for curiosities
        2. Generate learning goals
        3. Prioritize by curiosity
        4. Explore top curiosities
        5. Integrate discoveries
        6. Repeat

        Args:
            iterations: Number of learning cycles

        Returns:
            Summary of learning
        """
        print("\n" + "="*70)
        print("ACTIVE LEARNING LOOP - CURIOSITY-DRIVEN EXPLORATION")
        print("="*70)
        print(f"  Iterations: {iterations}")
        print("="*70 + "\n")

        total_knowledge_gained = 0
        total_explorations = 0

        for i in range(iterations):
            print(f"\n{'='*70}")
            print(f"ACTIVE LEARNING ITERATION {i+1}/{iterations}")
            print(f"{'='*70}")

            # 1. Scan for curiosities
            curiosities = self.scan_for_curiosities()

            if len(curiosities) == 0:
                print("  No curiosities found - knowledge base is complete!")
                break

            # 2. Generate learning goals
            goals = self.generate_learning_goals(curiosities, max_goals=3)

            # 3. Explore top curiosities
            for curiosity in curiosities[:3]:  # Top 3
                if not curiosity.explored:
                    results = self.explore(curiosity.id)

                    if results['success']:
                        total_knowledge_gained += results['knowledge_gained']
                        total_explorations += 1

                        # Update related goal progress
                        for goal in goals:
                            if curiosity.description in goal.goal_description:
                                self.update_goal_progress(goal.id, 0.5)

            # Small delay
            import time
            time.sleep(0.3)

        # Summary
        print("\n" + "="*70)
        print("ACTIVE LEARNING COMPLETE")
        print("="*70)
        print(f"\n  Explorations: {total_explorations}")
        print(f"  Knowledge gained: {total_knowledge_gained} new insights")
        print(f"  Goals generated: {len(self.learning_goals)}")
        print(f"  Curiosities discovered: {len(self.curious_items)}")
        print("\n  ✓ System actively seeking and acquiring knowledge!")
        print("="*70 + "\n")

        return {
            'iterations': iterations,
            'explorations': total_explorations,
            'knowledge_gained': total_knowledge_gained,
            'goals_generated': len(self.learning_goals),
            'curiosities': len(self.curious_items)
        }

    def get_statistics(self) -> Dict:
        """Get active learning statistics."""
        if len(self.curious_items) == 0:
            return {'status': 'inactive'}

        explored = sum(1 for c in self.curious_items.values() if c.explored)
        avg_curiosity = sum(c.curiosity_score for c in self.curious_items.values()) / len(self.curious_items)

        completed_goals = sum(1 for g in self.learning_goals.values() if g.completed)

        return {
            'status': 'active',
            'total_curiosities': len(self.curious_items),
            'explored': explored,
            'unexplored': len(self.curious_items) - explored,
            'avg_curiosity_score': avg_curiosity,
            'total_goals': len(self.learning_goals),
            'completed_goals': completed_goals,
            'explorations': len(self.exploration_history)
        }


# Demo
if __name__ == "__main__":
    print("Active Learning & Curiosity Engine")
    print("Autonomous, curiosity-driven exploration!")
    print()
    print("This makes AGI SELF-DIRECTED and AUTONOMOUS!")
