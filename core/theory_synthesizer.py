"""
Theory Synthesizer

Synthesizes broader theories from verified hypotheses!

Key Innovation:
- Integrates multiple verified hypotheses into unified theory
- Identifies common patterns and principles
- Generates explanatory frameworks
- Creates testable meta-predictions
- Builds hierarchical knowledge structures

This is how DISCOVERIES become THEORIES!

Example:
Hypotheses:
  1. "Prime gaps grow as numbers increase"
  2. "Prime density ≈ 1/log(n)"
  3. "Twin primes become rarer"

Synthesized Theory:
  "Prime Distribution Theory: Primes thin out predictably
   according to logarithmic density, with gaps following
   probabilistic patterns governed by local density."

This is SCIENTIFIC SYNTHESIS!
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict


class TheoryType(Enum):
    """Types of theories."""
    EXPLANATORY = "explanatory"  # Explains WHY
    UNIFYING = "unifying"        # Unifies separate phenomena
    PREDICTIVE = "predictive"    # Makes new predictions
    STRUCTURAL = "structural"    # Describes structure/relationships


@dataclass
class Theory:
    """A synthesized scientific theory."""
    id: str
    name: str
    theory_type: TheoryType

    # Core content
    statement: str  # Theory statement
    explanation: str  # Detailed explanation
    principles: List[str] = field(default_factory=list)  # Key principles

    # Supporting evidence
    supporting_hypotheses: List[str] = field(default_factory=list)  # Hypothesis IDs
    evidence_count: int = 0
    confidence: float = 0.0

    # Predictions
    predictions: List[str] = field(default_factory=list)  # New testable predictions
    scope: str = ""  # Domain/scope of theory

    # Relationships
    explains: List[str] = field(default_factory=list)  # What phenomena this explains
    unifies: List[str] = field(default_factory=list)  # What concepts this unifies
    contradicts: List[str] = field(default_factory=list)  # What this contradicts

    # Metadata
    novelty: float = 0.0  # How novel is this theory?
    generality: float = 0.0  # How general/broad?
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TheorySynthesizer:
    """
    Synthesizes theories from verified hypotheses.

    SCIENTIFIC SYNTHESIS ENGINE!

    Process:
    1. Take verified hypotheses
    2. Identify common patterns/themes
    3. Group related hypotheses
    4. Extract unifying principles
    5. Synthesize into coherent theory
    6. Generate meta-predictions
    7. Assess theory quality

    This is how we go from FACTS to UNDERSTANDING!
    """

    def __init__(
        self,
        llm_bridge=None,
        bayesian_evaluator=None,
        knowledge_graph=None
    ):
        self.llm = llm_bridge
        self.bayesian = bayesian_evaluator
        self.kg = knowledge_graph

        # Storage
        self.theories: Dict[str, Theory] = {}
        self.theory_count = 0

        # Configuration
        self.min_hypotheses_for_theory = 2  # Need at least 2 hypotheses to synthesize
        self.confidence_threshold = 0.7

        print("[Theory Synthesizer] Initialized - Ready to synthesize theories!")

    def synthesize(
        self,
        hypothesis_results: List[Dict],
        domain: Optional[str] = None
    ) -> List[Theory]:
        """
        Synthesize theories from verified hypotheses.

        Args:
            hypothesis_results: List of dicts with hypothesis evaluation results
                Each dict should have: hypothesis_id, claim, status, posterior, confidence
            domain: Domain to focus on (None = all)

        Returns:
            List of synthesized theories
        """
        # Filter for verified hypotheses
        verified = [
            h for h in hypothesis_results
            if h.get('status') == 'verified' and h.get('confidence', 0) >= self.confidence_threshold
        ]

        if len(verified) < self.min_hypotheses_for_theory:
            print(f"[Theory Synth] Need at least {self.min_hypotheses_for_theory} verified hypotheses to synthesize")
            return []

        print(f"\n[🔬] Synthesizing theories from {len(verified)} verified hypotheses...")

        theories = []

        # Strategy 1: Group by domain/topic
        grouped = self._group_hypotheses_by_topic(verified)

        # Strategy 2: For each group, synthesize theory
        for group_name, hypotheses in grouped.items():
            if len(hypotheses) >= self.min_hypotheses_for_theory:
                theory = self._synthesize_from_group(group_name, hypotheses, domain)
                if theory:
                    theories.append(theory)
                    self.theories[theory.id] = theory

        print(f"[🔬] Synthesized {len(theories)} theories")

        return theories

    def _group_hypotheses_by_topic(self, hypotheses: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group hypotheses by topic/theme.

        Uses simple keyword matching (in production, use embeddings).
        """
        groups = defaultdict(list)

        # Extract keywords from each hypothesis
        for hyp in hypotheses:
            claim = hyp.get('claim', '')

            # Simple grouping by domain or key terms
            # In production, use semantic similarity

            # Check for common mathematical terms
            if any(term in claim.lower() for term in ['prime', 'number', 'divisor', 'factor']):
                groups['number_theory'].append(hyp)
            elif any(term in claim.lower() for term in ['graph', 'network', 'connection', 'node']):
                groups['graph_theory'].append(hyp)
            elif any(term in claim.lower() for term in ['learn', 'knowledge', 'pattern']):
                groups['learning_theory'].append(hyp)
            else:
                groups['general'].append(hyp)

        return dict(groups)

    def _synthesize_from_group(
        self,
        group_name: str,
        hypotheses: List[Dict],
        domain: Optional[str]
    ) -> Optional[Theory]:
        """Synthesize theory from group of related hypotheses."""

        print(f"  Synthesizing theory from {len(hypotheses)} hypotheses in '{group_name}'")

        # Extract common patterns
        patterns = self._extract_common_patterns(hypotheses)

        # Generate theory statement
        if self.llm:
            theory_statement, explanation = self._generate_theory_with_llm(hypotheses, patterns, group_name)
        else:
            theory_statement, explanation = self._generate_theory_simple(hypotheses, patterns, group_name)

        # Extract principles
        principles = patterns[:5]  # Top 5 patterns as principles

        # Create theory
        theory = Theory(
            id=f"theory_{self.theory_count}",
            name=f"{group_name.replace('_', ' ').title()} Theory",
            theory_type=TheoryType.UNIFYING,
            statement=theory_statement,
            explanation=explanation,
            principles=principles,
            supporting_hypotheses=[h['hypothesis_id'] for h in hypotheses],
            evidence_count=sum(1 for h in hypotheses if h.get('status') == 'verified'),
            scope=group_name
        )

        self.theory_count += 1

        # Calculate confidence (average of supporting hypotheses)
        confidences = [h.get('posterior', 0) * h.get('confidence', 0) for h in hypotheses]
        theory.confidence = sum(confidences) / len(confidences) if confidences else 0

        # Assess novelty and generality
        theory.novelty = self._assess_novelty(theory)
        theory.generality = self._assess_generality(theory, hypotheses)

        # Generate meta-predictions
        theory.predictions = self._generate_meta_predictions(theory, hypotheses)

        return theory

    def _extract_common_patterns(self, hypotheses: List[Dict]) -> List[str]:
        """Extract common patterns/themes from hypotheses."""
        patterns = []

        # Simple pattern extraction: find common words
        # In production, use more sophisticated NLP

        all_claims = ' '.join(h.get('claim', '') for h in hypotheses).lower()

        # Common mathematical patterns
        if 'increase' in all_claims or 'grow' in all_claims or 'larger' in all_claims:
            patterns.append("Quantities tend to increase/grow")

        if 'decrease' in all_claims or 'reduce' in all_claims or 'smaller' in all_claims:
            patterns.append("Quantities tend to decrease/reduce")

        if 'proportion' in all_claims or 'ratio' in all_claims or 'density' in all_claims:
            patterns.append("Involves proportional relationships")

        if 'probability' in all_claims or 'likely' in all_claims or 'random' in all_claims:
            patterns.append("Has probabilistic nature")

        if 'pattern' in all_claims or 'regular' in all_claims or 'periodic' in all_claims:
            patterns.append("Exhibits regular patterns")

        # Default pattern
        if len(patterns) == 0:
            patterns.append("Multiple related phenomena observed")

        return patterns

    def _generate_theory_with_llm(
        self,
        hypotheses: List[Dict],
        patterns: List[str],
        group_name: str
    ) -> Tuple[str, str]:
        """Generate theory using LLM."""

        prompt = f"""You are a scientific theory synthesizer.

Given these VERIFIED hypotheses, synthesize a unified theory:

HYPOTHESES:
{chr(10).join(f"{i+1}. {h.get('claim', '')}" for i, h in enumerate(hypotheses))}

OBSERVED PATTERNS:
{chr(10).join(f"- {p}" for p in patterns)}

DOMAIN: {group_name}

Generate a unified theory that:
1. Explains all these hypotheses
2. Identifies the underlying principles
3. Makes the theory general and elegant

Format:
THEORY: [One concise statement that unifies all hypotheses]
EXPLANATION: [2-3 sentences explaining the theory and why it accounts for all observations]

Example:
THEORY: Prime Distribution follows logarithmic density with probabilistic gaps
EXPLANATION: The density of primes decreases as 1/log(n) because larger numbers have more opportunities to be composite. Gaps between primes follow probabilistic patterns governed by local density. This explains both the overall thinning and specific gap distributions.
"""

        try:
            response = self.llm.generate(prompt)
            text = response.get("text", "") if isinstance(response, dict) else str(response)

            # Parse response
            import re
            theory_match = re.search(r'THEORY:\s*(.+?)(?=\n|EXPLANATION:|$)', text, re.DOTALL)
            explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=\n\n|$)', text, re.DOTALL)

            theory_statement = theory_match.group(1).strip() if theory_match else f"Unified {group_name} theory"
            explanation = explanation_match.group(1).strip() if explanation_match else "Synthesized from verified hypotheses"

            return theory_statement, explanation

        except Exception as e:
            print(f"[Theory Synth] LLM generation failed: {e}")
            return self._generate_theory_simple(hypotheses, patterns, group_name)

    def _generate_theory_simple(
        self,
        hypotheses: List[Dict],
        patterns: List[str],
        group_name: str
    ) -> Tuple[str, str]:
        """Generate theory using simple template (fallback)."""

        # Simple template
        theory_statement = f"Unified theory of {group_name}: " + " and ".join(patterns[:2])

        explanation = f"This theory synthesizes {len(hypotheses)} verified hypotheses in {group_name}. "
        explanation += f"Key patterns include: {', '.join(patterns[:3])}. "
        explanation += "These observations suggest a common underlying principle."

        return theory_statement, explanation

    def _assess_novelty(self, theory: Theory) -> float:
        """
        Assess how novel this theory is.

        Novel if:
        - Connects previously unconnected concepts
        - Provides new explanatory framework
        - Makes unexpected predictions
        """
        # Simple heuristic: more supporting hypotheses = more novel
        # (because it connects more ideas)

        novelty = min(1.0, len(theory.supporting_hypotheses) / 10)

        # Boost if it makes predictions
        if len(theory.predictions) > 0:
            novelty += 0.2

        return min(1.0, novelty)

    def _assess_generality(self, theory: Theory, hypotheses: List[Dict]) -> float:
        """
        Assess how general/broad this theory is.

        General if:
        - Applies to many cases
        - Has broad scope
        - Explains diverse phenomena
        """
        # Simple heuristic: number of hypotheses unified
        generality = min(1.0, len(hypotheses) / 5)

        return generality

    def _generate_meta_predictions(
        self,
        theory: Theory,
        hypotheses: List[Dict]
    ) -> List[str]:
        """
        Generate new predictions from theory.

        Theory should make predictions beyond its supporting hypotheses!
        """
        predictions = []

        if self.llm:
            prompt = f"""Given this theory, generate 3 NEW testable predictions:

THEORY: {theory.statement}

EXPLANATION: {theory.explanation}

Generate predictions that:
- Go beyond the original hypotheses
- Are testable
- Would further validate the theory

Format:
1. [Prediction 1]
2. [Prediction 2]
3. [Prediction 3]
"""

            try:
                response = self.llm.generate(prompt)
                text = response.get("text", "") if isinstance(response, dict) else str(response)

                # Extract numbered predictions
                import re
                pred_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)'
                matches = re.findall(pred_pattern, text, re.DOTALL)

                predictions = [m.strip() for m in matches[:3]]

            except:
                pass

        # Fallback predictions
        if len(predictions) == 0:
            predictions = [
                f"The theory should apply to related domains",
                f"Similar patterns should be observable in analogous systems",
                f"The underlying principles should generalize"
            ]

        return predictions

    def get_theory_report(self, theory_id: str) -> str:
        """Generate comprehensive theory report."""
        if theory_id not in self.theories:
            return "Theory not found"

        theory = self.theories[theory_id]

        report = f"""
{'='*70}
THEORY: {theory.name}
{'='*70}

TYPE: {theory.theory_type.value}
SCOPE: {theory.scope}

STATEMENT:
{theory.statement}

EXPLANATION:
{theory.explanation}

PRINCIPLES:
{chr(10).join(f"  {i+1}. {p}" for i, p in enumerate(theory.principles))}

SUPPORTING EVIDENCE:
  Verified hypotheses: {len(theory.supporting_hypotheses)}
  Evidence pieces: {theory.evidence_count}
  Confidence: {theory.confidence:.2%}

QUALITY METRICS:
  Novelty: {theory.novelty:.2f}
  Generality: {theory.generality:.2f}
  Confidence: {theory.confidence:.2f}

NEW PREDICTIONS:
{chr(10).join(f"  {i+1}. {p}" for i, p in enumerate(theory.predictions))}

"""

        if theory.explains:
            report += f"EXPLAINS: {', '.join(theory.explains)}\n"

        if theory.unifies:
            report += f"UNIFIES: {', '.join(theory.unifies)}\n"

        report += f"\nCreated: {theory.created_at}\n"
        report += "=" * 70

        return report

    def get_all_theories(self) -> List[Theory]:
        """Get all synthesized theories."""
        return list(self.theories.values())

    def get_statistics(self) -> Dict:
        """Get theory synthesis statistics."""
        if len(self.theories) == 0:
            return {'status': 'no_theories'}

        # Type distribution
        type_counts = defaultdict(int)
        for theory in self.theories.values():
            type_counts[theory.theory_type.value] += 1

        # Quality metrics
        avg_novelty = sum(t.novelty for t in self.theories.values()) / len(self.theories)
        avg_generality = sum(t.generality for t in self.theories.values()) / len(self.theories)
        avg_confidence = sum(t.confidence for t in self.theories.values()) / len(self.theories)

        # Coverage
        total_hypotheses = sum(len(t.supporting_hypotheses) for t in self.theories.values())

        return {
            'status': 'active',
            'total_theories': len(self.theories),
            'by_type': dict(type_counts),

            # Quality
            'avg_novelty': avg_novelty,
            'avg_generality': avg_generality,
            'avg_confidence': avg_confidence,

            # Coverage
            'total_hypotheses_synthesized': total_hypotheses,
            'avg_hypotheses_per_theory': total_hypotheses / len(self.theories) if len(self.theories) > 0 else 0,

            # Predictions
            'total_predictions': sum(len(t.predictions) for t in self.theories.values())
        }

    def demonstrate_theory_synthesis(self):
        """Demonstrate theory synthesis."""
        print("\n" + "=" * 70)
        print("THEORY SYNTHESIZER - Demonstration")
        print("=" * 70)

        stats = self.get_statistics()

        if stats['status'] == 'no_theories':
            print("\n[!] No theories synthesized yet")
            return

        print(f"\n📊 STATISTICS:")
        print(f"  Total theories: {stats['total_theories']}")
        print(f"  Hypotheses synthesized: {stats['total_hypotheses_synthesized']}")
        print(f"  Avg hypotheses per theory: {stats['avg_hypotheses_per_theory']:.1f}")
        print(f"  Total predictions: {stats['total_predictions']}")

        print(f"\n🔬 BY TYPE:")
        for t, count in stats['by_type'].items():
            print(f"  {t}: {count}")

        print(f"\n⭐ QUALITY:")
        print(f"  Average novelty: {stats['avg_novelty']:.2f}")
        print(f"  Average generality: {stats['avg_generality']:.2f}")
        print(f"  Average confidence: {stats['avg_confidence']:.2%}")

        # Show example theory
        if len(self.theories) > 0:
            example = list(self.theories.values())[0]
            print(f"\n📖 EXAMPLE THEORY:")
            print(f"  Name: {example.name}")
            print(f"  Statement: {example.statement[:100]}...")
            print(f"  Confidence: {example.confidence:.2%}")
            print(f"  Predictions: {len(example.predictions)}")

        print("\n" + "=" * 70)


# Demo
if __name__ == "__main__":
    print("Theory Synthesizer")
    print("Synthesizes theories from verified hypotheses!")
    print()

    # Create synthesizer
    synthesizer = TheorySynthesizer()

    # Example verified hypotheses
    hypotheses = [
        {
            'hypothesis_id': 'hyp_1',
            'claim': 'Prime density decreases as numbers grow',
            'status': 'verified',
            'posterior': 0.9,
            'confidence': 0.85
        },
        {
            'hypothesis_id': 'hyp_2',
            'claim': 'Prime gaps increase on average',
            'status': 'verified',
            'posterior': 0.88,
            'confidence': 0.8
        },
        {
            'hypothesis_id': 'hyp_3',
            'claim': 'Twin primes become rarer',
            'status': 'verified',
            'posterior': 0.85,
            'confidence': 0.75
        }
    ]

    # Synthesize
    theories = synthesizer.synthesize(hypotheses, domain='number_theory')

    # Print report
    if theories:
        print(synthesizer.get_theory_report(theories[0].id))

    # Demonstrate
    synthesizer.demonstrate_theory_synthesis()
