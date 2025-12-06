"""
Autonomous Hypothesis Generator

THE CORE OF DISCOVERY!

Generates novel hypotheses from:
- Knowledge gaps (high FEP regions)
- Unexplained patterns
- Anomalies in data
- Contradictions

This is what makes the system DISCOVER autonomously!
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime


class HypothesisType(Enum):
    """Types of hypotheses."""
    EXPLANATORY = "explanatory"  # Explains WHY something happens
    PREDICTIVE = "predictive"    # Predicts WHAT will happen
    CAUSAL = "causal"            # Claims X causes Y
    STRUCTURAL = "structural"     # Claims X and Y share structure
    EXISTENCE = "existence"       # Claims something exists/doesn't exist


@dataclass
class Prediction:
    """A testable prediction from a hypothesis."""
    statement: str
    test_method: str  # How to verify this
    expected_if_true: str
    expected_if_false: str
    testability_score: float  # 0-1, how easy to test


@dataclass
class Hypothesis:
    """
    A scientific hypothesis with predictions.

    This is what the discovery machine generates!
    """
    id: str
    claim: str  # The hypothesis statement
    type: HypothesisType
    reasoning: str  # WHY this hypothesis makes sense
    predictions: List[Prediction] = field(default_factory=list)

    # Scores
    novelty_score: float = 0.5  # How novel is this? (0-1)
    plausibility_score: float = 0.5  # How plausible? (0-1)
    testability_score: float = 0.5  # How testable? (0-1)
    impact_score: float = 0.5  # How important if true? (0-1)
    fep_reduction: float = 0.0  # How much does this reduce free energy?

    # Metadata
    generated_from: str = ""  # What triggered this hypothesis
    confidence: float = 0.0  # Updated after testing
    status: str = "untested"  # untested, testing, confirmed, rejected
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def overall_score(self) -> float:
        """Combined score for ranking hypotheses."""
        return (
            0.3 * self.novelty_score +
            0.25 * self.plausibility_score +
            0.25 * self.testability_score +
            0.2 * self.impact_score
        )


class HypothesisGenerator:
    """
    Generates hypotheses autonomously from knowledge gaps and patterns.

    This is THE core discovery capability!
    """

    def __init__(self, llm_bridge=None, knowledge_graph=None, fep_learner=None):
        self.llm = llm_bridge
        self.kg = knowledge_graph
        self.fep = fep_learner

        self.generated_hypotheses: List[Hypothesis] = []
        self.hypothesis_count = 0

    def identify_knowledge_gaps(
        self,
        domain: Optional[str] = None,
        min_fe_threshold: float = 0.6
    ) -> List[Dict]:
        """
        Identify knowledge gaps using FEP.

        High free energy = poor prediction = knowledge gap!

        Args:
            domain: Specific domain to check (None = all)
            min_fe_threshold: Minimum FE to consider a gap

        Returns:
            List of gaps with FE scores
        """
        gaps = []

        if not self.kg:
            print("[Hypothesis Generator] No knowledge graph available")
            return gaps

        print(f"\n[🔍] Scanning for knowledge gaps (FE threshold: {min_fe_threshold})...")

        # If we have FEP learner, use it to compute free energy
        if self.fep:
            # Get all concepts
            concepts = self.kg.get_all_concepts() if hasattr(self.kg, 'get_all_concepts') else []

            for concept in concepts:
                # Compute free energy for this concept
                fe = self._compute_concept_free_energy(concept)

                if fe >= min_fe_threshold:
                    gaps.append({
                        'concept': concept,
                        'free_energy': fe,
                        'type': 'high_fe',
                        'description': f"Poorly explained concept with FE={fe:.2f}"
                    })

        # Also find structural gaps (missing connections)
        if hasattr(self.kg, 'find_missing_connections'):
            missing = self.kg.find_missing_connections()
            for conn in missing:
                gaps.append({
                    'concept': f"{conn['source']} <-> {conn['target']}",
                    'free_energy': conn.get('expected_fe_reduction', 0.5),
                    'type': 'missing_connection',
                    'description': f"Missing connection between {conn['source']} and {conn['target']}"
                })

        # Find unexplained patterns
        patterns = self._find_unexplained_patterns()
        gaps.extend(patterns)

        # Sort by free energy (highest = biggest gaps)
        gaps.sort(key=lambda x: x['free_energy'], reverse=True)

        print(f"[🔍] Found {len(gaps)} knowledge gaps")
        for i, gap in enumerate(gaps[:5]):
            print(f"  {i+1}. {gap['concept']}: FE={gap['free_energy']:.2f}")

        return gaps

    def _compute_concept_free_energy(self, concept: str) -> float:
        """
        Compute free energy for a concept.

        FE = Prediction Error + Complexity

        High FE = poorly understood/connected
        """
        if not self.kg:
            return 0.5

        # Prediction error: Can we predict this concept from its neighbors?
        neighbors = self.kg.get_neighbors(concept) if hasattr(self.kg, 'get_neighbors') else []

        if len(neighbors) == 0:
            prediction_error = 1.0  # No connections = can't predict
        else:
            # Simple heuristic: more connections = better prediction
            prediction_error = max(0.1, 1.0 - (len(neighbors) / 10.0))

        # Complexity: How unusual is this concept?
        # Simple heuristic: rare domains = high complexity
        complexity = 0.3  # Default moderate complexity

        # Total free energy
        free_energy = prediction_error + complexity

        return free_energy

    def _find_unexplained_patterns(self) -> List[Dict]:
        """Find patterns that don't have explanations yet."""
        patterns = []

        # Example patterns (would be detected automatically in full implementation)
        # For now, this is a placeholder for pattern detection logic

        return patterns

    def generate_hypotheses(
        self,
        gap: Dict,
        num_hypotheses: int = 3
    ) -> List[Hypothesis]:
        """
        Generate hypotheses to explain a knowledge gap.

        This is where the MAGIC happens - autonomous hypothesis generation!

        Args:
            gap: Knowledge gap dict from identify_knowledge_gaps()
            num_hypotheses: Number of alternative hypotheses to generate

        Returns:
            List of Hypothesis objects
        """
        if not self.llm:
            print("[!] No LLM available for hypothesis generation")
            return []

        print(f"\n[💡] Generating {num_hypotheses} hypotheses for: {gap['concept']}")

        # Build prompt for LLM
        prompt = f"""You are a scientific hypothesis generator.

KNOWLEDGE GAP:
Concept: {gap['concept']}
Free Energy: {gap['free_energy']:.2f} (high = poorly understood)
Type: {gap['type']}
Description: {gap['description']}

Generate {num_hypotheses} NOVEL scientific hypotheses to explain this gap.

For each hypothesis:
1. State the hypothesis clearly
2. Explain WHY this hypothesis makes sense
3. Provide 2-3 testable predictions
4. Rate novelty (0-1), plausibility (0-1), testability (0-1), impact (0-1)

Format:
HYPOTHESIS 1:
CLAIM: [hypothesis statement]
TYPE: [explanatory/predictive/causal/structural/existence]
REASONING: [why this makes sense]
PREDICTION_1: [what should happen if true] | TEST: [how to verify]
PREDICTION_2: [what should happen if true] | TEST: [how to verify]
NOVELTY: [0.0-1.0]
PLAUSIBILITY: [0.0-1.0]
TESTABILITY: [0.0-1.0]
IMPACT: [0.0-1.0]

Example for gap "Why do primes become rarer?":
HYPOTHESIS 1:
CLAIM: Prime density decreases because composite numbers have more factorizations available as numbers grow
REASONING: Larger numbers can be formed by multiplying smaller primes in more combinations
PREDICTION_1: Density of primes ~1/log(n) | TEST: Count primes up to n, verify ratio
PREDICTION_2: Composite numbers have more divisors on average as n grows | TEST: Compute average number of divisors
NOVELTY: 0.3 (known result)
PLAUSIBILITY: 0.9 (mathematically proven)
TESTABILITY: 1.0 (easily computable)
IMPACT: 0.6 (fundamental number theory)

Now generate for the gap above:
"""

        # Get LLM response
        response = self.llm.generate(prompt)
        text = response.get("text", "") if isinstance(response, dict) else str(response)

        # Parse hypotheses from response
        hypotheses = self._parse_hypotheses(text, gap)

        # Generate predictions for each
        for h in hypotheses:
            if len(h.predictions) == 0:
                h.predictions = self._generate_predictions(h)

        # Store
        self.generated_hypotheses.extend(hypotheses)

        # Print summary
        for i, h in enumerate(hypotheses, 1):
            print(f"\n  Hypothesis {i}: {h.claim[:80]}...")
            print(f"    Novelty: {h.novelty_score:.2f}, Plausibility: {h.plausibility_score:.2f}")
            print(f"    Testability: {h.testability_score:.2f}, Impact: {h.impact_score:.2f}")
            print(f"    Overall Score: {h.overall_score():.2f}")

        return hypotheses

    def _parse_hypotheses(self, text: str, gap: Dict) -> List[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        hypotheses = []

        # Split by hypothesis markers
        sections = re.split(r'HYPOTHESIS \d+:', text)

        for section in sections[1:]:  # Skip first empty split
            try:
                # Extract fields
                claim_match = re.search(r'CLAIM:\s*(.+?)(?=\n|TYPE:|$)', section, re.DOTALL)
                type_match = re.search(r'TYPE:\s*(\w+)', section)
                reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\nPREDICTION|NOVELTY|$)', section, re.DOTALL)

                # Extract scores
                novelty_match = re.search(r'NOVELTY:\s*([\d.]+)', section)
                plausibility_match = re.search(r'PLAUSIBILITY:\s*([\d.]+)', section)
                testability_match = re.search(r'TESTABILITY:\s*([\d.]+)', section)
                impact_match = re.search(r'IMPACT:\s*([\d.]+)', section)

                # Extract predictions
                predictions = []
                pred_pattern = r'PREDICTION_\d+:\s*(.+?)\s*\|\s*TEST:\s*(.+?)(?=\n|PREDICTION|NOVELTY|$)'
                for pred_match in re.finditer(pred_pattern, section):
                    pred = Prediction(
                        statement=pred_match.group(1).strip(),
                        test_method=pred_match.group(2).strip(),
                        expected_if_true="Prediction should hold",
                        expected_if_false="Prediction should fail",
                        testability_score=0.8
                    )
                    predictions.append(pred)

                if claim_match:
                    # Determine type
                    type_str = type_match.group(1).lower() if type_match else "explanatory"
                    try:
                        hyp_type = HypothesisType(type_str)
                    except:
                        hyp_type = HypothesisType.EXPLANATORY

                    h = Hypothesis(
                        id=f"hyp_{self.hypothesis_count}",
                        claim=claim_match.group(1).strip(),
                        type=hyp_type,
                        reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
                        predictions=predictions,
                        novelty_score=float(novelty_match.group(1)) if novelty_match else 0.5,
                        plausibility_score=float(plausibility_match.group(1)) if plausibility_match else 0.5,
                        testability_score=float(testability_match.group(1)) if testability_match else 0.5,
                        impact_score=float(impact_match.group(1)) if impact_match else 0.5,
                        fep_reduction=gap.get('free_energy', 0.0),
                        generated_from=gap['concept']
                    )

                    hypotheses.append(h)
                    self.hypothesis_count += 1

            except Exception as e:
                print(f"[!] Failed to parse hypothesis section: {e}")
                continue

        return hypotheses

    def _generate_predictions(self, hypothesis: Hypothesis) -> List[Prediction]:
        """Generate testable predictions from a hypothesis."""
        if not self.llm:
            return []

        prompt = f"""Given this hypothesis, generate 3 specific, testable predictions:

HYPOTHESIS: {hypothesis.claim}

For each prediction:
- State what should happen if the hypothesis is TRUE
- State what should happen if the hypothesis is FALSE
- Explain how to test it

Format:
PREDICTION: [specific statement]
IF_TRUE: [expected outcome]
IF_FALSE: [expected outcome]
TEST_METHOD: [how to verify]
"""

        response = self.llm.generate(prompt)
        text = response.get("text", "") if isinstance(response, dict) else str(response)

        # Parse predictions (simplified)
        predictions = []
        # TODO: Implement proper parsing

        return predictions

    def rank_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        criteria: str = "overall"
    ) -> List[Hypothesis]:
        """
        Rank hypotheses by various criteria.

        Args:
            hypotheses: List of hypotheses to rank
            criteria: "overall", "novelty", "testability", "impact", "plausibility"

        Returns:
            Sorted list (best first)
        """
        if criteria == "overall":
            key_func = lambda h: h.overall_score()
        elif criteria == "novelty":
            key_func = lambda h: h.novelty_score
        elif criteria == "testability":
            key_func = lambda h: h.testability_score
        elif criteria == "impact":
            key_func = lambda h: h.impact_score
        elif criteria == "plausibility":
            key_func = lambda h: h.plausibility_score
        else:
            key_func = lambda h: h.overall_score()

        return sorted(hypotheses, key=key_func, reverse=True)

    def get_top_hypotheses(
        self,
        domain: Optional[str] = None,
        k: int = 5,
        min_score: float = 0.5
    ) -> List[Hypothesis]:
        """
        Get top k hypotheses for a domain.

        Args:
            domain: Domain to filter by (None = all)
            k: Number to return
            min_score: Minimum overall score

        Returns:
            Top k hypotheses
        """
        # Filter by domain if specified
        if domain:
            filtered = [h for h in self.generated_hypotheses if domain in h.generated_from.lower()]
        else:
            filtered = self.generated_hypotheses

        # Filter by score
        filtered = [h for h in filtered if h.overall_score() >= min_score]

        # Rank and return top k
        ranked = self.rank_hypotheses(filtered)
        return ranked[:k]

    def discover_autonomous(
        self,
        domain: Optional[str] = None,
        num_gaps: int = 5,
        hypotheses_per_gap: int = 3
    ) -> Dict:
        """
        Autonomous discovery process!

        1. Identify knowledge gaps
        2. Generate hypotheses for each gap
        3. Rank all hypotheses
        4. Return top discoveries

        This is the FULL DISCOVERY LOOP!

        Args:
            domain: Domain to focus on (None = all)
            num_gaps: How many gaps to investigate
            hypotheses_per_gap: Hypotheses to generate per gap

        Returns:
            Discovery results dict
        """
        print("\n" + "="*70)
        print("🔬 AUTONOMOUS DISCOVERY MODE 🔬")
        print("="*70)

        # 1. Find gaps
        gaps = self.identify_knowledge_gaps(domain)
        gaps = gaps[:num_gaps]

        if len(gaps) == 0:
            print("\n[✓] No significant knowledge gaps found!")
            print("    System has good coverage of this domain.")
            return {
                'gaps_found': 0,
                'hypotheses_generated': 0,
                'top_hypotheses': []
            }

        # 2. Generate hypotheses for each gap
        all_hypotheses = []
        for i, gap in enumerate(gaps, 1):
            print(f"\n--- Gap {i}/{len(gaps)} ---")
            hyps = self.generate_hypotheses(gap, hypotheses_per_gap)
            all_hypotheses.extend(hyps)

        # 3. Rank all hypotheses
        ranked = self.rank_hypotheses(all_hypotheses)

        # 4. Return results
        print("\n" + "="*70)
        print("🎯 TOP DISCOVERIES:")
        print("="*70)

        for i, h in enumerate(ranked[:5], 1):
            print(f"\n{i}. {h.claim}")
            print(f"   Novelty: {h.novelty_score:.2f} | Plausibility: {h.plausibility_score:.2f} | Testability: {h.testability_score:.2f}")
            print(f"   Impact: {h.impact_score:.2f} | Overall: {h.overall_score():.2f}")
            print(f"   Predictions: {len(h.predictions)}")

        print("\n" + "="*70)

        return {
            'gaps_found': len(gaps),
            'hypotheses_generated': len(all_hypotheses),
            'top_hypotheses': ranked[:10],
            'all_hypotheses': all_hypotheses
        }


# Example usage
if __name__ == "__main__":
    print("Hypothesis Generator - Core Discovery Engine")
    print("This module generates autonomous hypotheses from knowledge gaps.")
    print("\nIntegrate with:")
    print("  - Knowledge Graph (to find gaps)")
    print("  - FEP Learner (to compute free energy)")
    print("  - LLM Bridge (to generate creative hypotheses)")
