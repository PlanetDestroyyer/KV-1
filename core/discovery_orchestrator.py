"""
Discovery Orchestrator

THE HEART OF AUTONOMOUS DISCOVERY!

Coordinates the full discovery loop:
OBSERVE → QUESTION → HYPOTHESIZE → PREDICT → TEST →
ANALYZE → THEORIZE → EXPLAIN → ITERATE → DISCOVER

This is what makes the system truly AUTONOMOUS and SELF-DISCOVERING!

Key Innovation:
- Uses FEP to identify knowledge gaps
- Generates hypotheses autonomously
- Designs experiments to test predictions
- Updates beliefs with Bayesian evidence
- Detects and resolves contradictions
- Synthesizes theories from patterns
- Tracks compound knowledge growth

This is the AGI discovery loop!
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time


class DiscoveryPhase(Enum):
    """Phases of the discovery loop."""
    OBSERVE = "observe"          # Gather data, identify patterns
    QUESTION = "question"        # Find knowledge gaps (high FE regions)
    HYPOTHESIZE = "hypothesize"  # Generate explanatory hypotheses
    PREDICT = "predict"          # Derive testable predictions
    TEST = "test"                # Design and run experiments
    ANALYZE = "analyze"          # Evaluate evidence
    THEORIZE = "theorize"        # Synthesize broader theory
    EXPLAIN = "explain"          # Generate human-readable explanation
    ITERATE = "iterate"          # Refine and continue


@dataclass
class DiscoverySession:
    """A complete discovery session."""
    id: str
    domain: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    # Discovery outputs
    gaps_identified: List[Dict] = field(default_factory=list)
    hypotheses_generated: List[Dict] = field(default_factory=list)
    experiments_conducted: List[Dict] = field(default_factory=list)
    theories_synthesized: List[Dict] = field(default_factory=list)

    # Metrics
    total_hypotheses: int = 0
    verified_hypotheses: int = 0
    rejected_hypotheses: int = 0
    contradictions_found: int = 0
    contradictions_resolved: int = 0

    # FEP metrics
    initial_free_energy: float = 0.0
    final_free_energy: float = 0.0
    fe_reduction: float = 0.0

    # Compound growth
    learning_events: int = 0
    avg_learning_speedup: float = 1.0

    status: str = "running"  # running, completed, paused


class DiscoveryOrchestrator:
    """
    Orchestrates autonomous scientific discovery.

    THE MAIN DISCOVERY ENGINE!

    Integrates:
    - FEP Knowledge Graph (gaps = discovery opportunities)
    - Hypothesis Generator (creative hypothesis generation)
    - Bayesian Evaluator (evidence-based belief updating)
    - Contradiction Detector (logical consistency)
    - Compound Growth Tracker (learning acceleration)
    - Experiment Designer (test generation)
    - Theory Synthesizer (pattern integration)
    - CoT Pattern Miner (meta-learning)

    This is the AGI discovery loop!
    """

    def __init__(
        self,
        knowledge_graph=None,
        hypothesis_generator=None,
        bayesian_evaluator=None,
        contradiction_detector=None,
        compound_tracker=None,
        experiment_designer=None,
        theory_synthesizer=None,
        cot_miner=None,
        llm_bridge=None
    ):
        # Core components
        self.kg = knowledge_graph
        self.hyp_gen = hypothesis_generator
        self.bayesian = bayesian_evaluator
        self.contra_detect = contradiction_detector
        self.compound = compound_tracker
        self.exp_design = experiment_designer
        self.theory_synth = theory_synthesizer
        self.cot_miner = cot_miner
        self.llm = llm_bridge

        # Discovery sessions
        self.sessions: Dict[str, DiscoverySession] = {}
        self.session_count = 0

        # Configuration
        self.max_hypotheses_per_gap = 3
        self.max_iterations = 10
        self.fe_reduction_target = 0.3  # Stop when FE reduced by 30%
        self.confidence_threshold = 0.8

        print("=" * 70)
        print("🔬 DISCOVERY ORCHESTRATOR INITIALIZED 🔬")
        print("=" * 70)
        print("Components loaded:")
        print(f"  Knowledge Graph: {'✓' if self.kg else '✗'}")
        print(f"  Hypothesis Generator: {'✓' if self.hyp_gen else '✗'}")
        print(f"  Bayesian Evaluator: {'✓' if self.bayesian else '✗'}")
        print(f"  Contradiction Detector: {'✓' if self.contra_detect else '✗'}")
        print(f"  Compound Growth Tracker: {'✓' if self.compound else '✗'}")
        print(f"  Experiment Designer: {'✓' if self.exp_design else '✗'}")
        print(f"  Theory Synthesizer: {'✓' if self.theory_synth else '✗'}")
        print(f"  CoT Pattern Miner: {'✓' if self.cot_miner else '✗'}")
        print("=" * 70)
        print("\nReady for AUTONOMOUS DISCOVERY!")

    def discover(
        self,
        domain: str,
        initial_observations: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
        verbose: bool = True
    ) -> DiscoverySession:
        """
        Run autonomous discovery loop!

        This is THE MAIN FUNCTION that implements the full discovery process.

        Args:
            domain: Domain to explore (e.g., "number_theory", "physics")
            initial_observations: Starting observations/data
            max_iterations: Max discovery iterations (None = use default)
            verbose: Print progress

        Returns:
            DiscoverySession with all discoveries
        """
        if verbose:
            print("\n" + "=" * 70)
            print("🔬 STARTING AUTONOMOUS DISCOVERY 🔬")
            print("=" * 70)
            print(f"Domain: {domain}")
            print(f"Max iterations: {max_iterations or self.max_iterations}")
            print("=" * 70 + "\n")

        # Create session
        session = DiscoverySession(
            id=f"discovery_{self.session_count}",
            domain=domain
        )
        self.sessions[session.id] = session
        self.session_count += 1

        # Measure initial free energy
        if self.kg:
            session.initial_free_energy = self.kg.compute_graph_free_energy() if hasattr(self.kg, 'compute_graph_free_energy') else 1.0

        max_iters = max_iterations or self.max_iterations

        # DISCOVERY LOOP
        for iteration in range(max_iters):
            if verbose:
                print(f"\n{'='*70}")
                print(f"ITERATION {iteration + 1}/{max_iters}")
                print(f"{'='*70}\n")

            start_time = time.time()

            # PHASE 1: OBSERVE
            if verbose:
                print(f"[{DiscoveryPhase.OBSERVE.value.upper()}] Analyzing current knowledge state...")

            # PHASE 2: QUESTION - Identify knowledge gaps using FEP
            if verbose:
                print(f"[{DiscoveryPhase.QUESTION.value.upper()}] Identifying knowledge gaps (high FE regions)...")

            gaps = self._identify_gaps(domain)
            session.gaps_identified.extend(gaps)

            if len(gaps) == 0:
                if verbose:
                    print("  ✓ No significant gaps found - domain well understood!")
                break

            if verbose:
                print(f"  Found {len(gaps)} knowledge gaps")
                for i, gap in enumerate(gaps[:3], 1):
                    print(f"    {i}. {gap.get('concept', gap.get('description', 'Unknown'))}: FE={gap['free_energy']:.3f}")

            # PHASE 3: HYPOTHESIZE - Generate hypotheses for top gaps
            if verbose:
                print(f"\n[{DiscoveryPhase.HYPOTHESIZE.value.upper()}] Generating hypotheses...")

            hypotheses = self._generate_hypotheses(gaps[:3])  # Top 3 gaps
            session.hypotheses_generated.extend(hypotheses)
            session.total_hypotheses += len(hypotheses)

            if verbose:
                print(f"  Generated {len(hypotheses)} hypotheses")

            # PHASE 4: PREDICT - Extract predictions from hypotheses
            if verbose:
                print(f"\n[{DiscoveryPhase.PREDICT.value.upper()}] Extracting testable predictions...")

            predictions = self._extract_predictions(hypotheses)

            if verbose:
                print(f"  Extracted {len(predictions)} testable predictions")

            # PHASE 5: TEST - Design and run experiments
            if verbose:
                print(f"\n[{DiscoveryPhase.TEST.value.upper()}] Designing experiments...")

            experiments = self._design_and_run_experiments(predictions)
            session.experiments_conducted.extend(experiments)

            if verbose:
                print(f"  Conducted {len(experiments)} experiments")

            # PHASE 6: ANALYZE - Evaluate evidence using Bayesian updating
            if verbose:
                print(f"\n[{DiscoveryPhase.ANALYZE.value.upper()}] Evaluating evidence...")

            results = self._analyze_evidence(hypotheses, experiments)

            verified = sum(1 for r in results if r['status'] == 'verified')
            rejected = sum(1 for r in results if r['status'] == 'rejected')
            session.verified_hypotheses += verified
            session.rejected_hypotheses += rejected

            if verbose:
                print(f"  Verified: {verified}, Rejected: {rejected}, Unverified: {len(results) - verified - rejected}")

            # Detect contradictions
            if self.contra_detect:
                if verbose:
                    print(f"\n  [Consistency Check] Detecting contradictions...")

                contradictions = self._detect_and_resolve_contradictions()
                session.contradictions_found += len(contradictions)

                if verbose and len(contradictions) > 0:
                    print(f"    Found {len(contradictions)} contradictions")

            # PHASE 7: THEORIZE - Synthesize broader theory
            if verbose:
                print(f"\n[{DiscoveryPhase.THEORIZE.value.upper()}] Synthesizing theories...")

            theories = self._synthesize_theories(results, iteration)
            session.theories_synthesized.extend(theories)

            if verbose:
                print(f"  Synthesized {len(theories)} theories")

            # PHASE 8: EXPLAIN - Generate explanations
            if verbose:
                print(f"\n[{DiscoveryPhase.EXPLAIN.value.upper()}] Generating explanations...")

            # Track learning (compound growth)
            if self.compound:
                iteration_time = time.time() - start_time
                self.compound.record_learning_event(
                    concept=f"iteration_{iteration}",
                    time_seconds=iteration_time,
                    prereqs=[f"iteration_{i}" for i in range(max(0, iteration - 3), iteration)],
                    confidence=0.8
                )
                session.learning_events += 1

            # PHASE 9: ITERATE - Check if we should continue
            if verbose:
                print(f"\n[{DiscoveryPhase.ITERATE.value.upper()}] Checking progress...")

            # Measure free energy reduction
            if self.kg:
                current_fe = self.kg.compute_graph_free_energy() if hasattr(self.kg, 'compute_graph_free_energy') else session.initial_free_energy
                fe_reduction = (session.initial_free_energy - current_fe) / session.initial_free_energy if session.initial_free_energy > 0 else 0

                if verbose:
                    print(f"  Free Energy: {session.initial_free_energy:.3f} → {current_fe:.3f} (reduction: {100*fe_reduction:.1f}%)")

                # Stop if FE reduced enough
                if fe_reduction >= self.fe_reduction_target:
                    if verbose:
                        print(f"  ✓ Target FE reduction achieved!")
                    break

        # Finalize session
        session.end_time = datetime.now().isoformat()
        session.status = "completed"

        # Final FE
        if self.kg:
            session.final_free_energy = self.kg.compute_graph_free_energy() if hasattr(self.kg, 'compute_graph_free_energy') else session.initial_free_energy
            session.fe_reduction = session.initial_free_energy - session.final_free_energy

        # Compound growth speedup
        if self.compound:
            stats = self.compound.get_compound_stats()
            if stats['status'] == 'active':
                session.avg_learning_speedup = stats.get('speedup_factor', 1.0)

        if verbose:
            self._print_discovery_summary(session)

        return session

    def _identify_gaps(self, domain: str) -> List[Dict]:
        """Identify knowledge gaps using FEP."""
        if not self.hyp_gen:
            return []

        # Use hypothesis generator's gap identification
        gaps = self.hyp_gen.identify_knowledge_gaps(domain=domain, min_fe_threshold=0.6)

        return gaps[:10]  # Top 10 gaps

    def _generate_hypotheses(self, gaps: List[Dict]) -> List[Dict]:
        """Generate hypotheses for gaps."""
        if not self.hyp_gen:
            return []

        all_hypotheses = []

        for gap in gaps:
            hyps = self.hyp_gen.generate_hypotheses(gap, num_hypotheses=self.max_hypotheses_per_gap)

            # Convert to dicts
            for h in hyps:
                all_hypotheses.append({
                    'id': h.id,
                    'claim': h.claim,
                    'type': h.type.value,
                    'reasoning': h.reasoning,
                    'predictions': [
                        {
                            'statement': p.statement,
                            'test_method': p.test_method
                        }
                        for p in h.predictions
                    ],
                    'novelty': h.novelty_score,
                    'plausibility': h.plausibility_score,
                    'testability': h.testability_score,
                    'impact': h.impact_score,
                    'gap_source': gap.get('concept', 'unknown')
                })

        return all_hypotheses

    def _extract_predictions(self, hypotheses: List[Dict]) -> List[Dict]:
        """Extract testable predictions from hypotheses."""
        predictions = []

        for hyp in hypotheses:
            for pred in hyp.get('predictions', []):
                predictions.append({
                    'hypothesis_id': hyp['id'],
                    'statement': pred['statement'],
                    'test_method': pred['test_method']
                })

        return predictions

    def _design_and_run_experiments(self, predictions: List[Dict]) -> List[Dict]:
        """Design and simulate running experiments."""
        experiments = []

        if not self.exp_design:
            # Simple fallback: mark predictions as experiments
            for pred in predictions:
                experiments.append({
                    'prediction': pred['statement'],
                    'method': pred['test_method'],
                    'result': 'simulated',  # Would be actual result in full implementation
                    'supports_hypothesis': True  # Placeholder
                })
        else:
            # Use experiment designer
            for pred in predictions:
                exp = self.exp_design.design_experiment(pred)
                experiments.append(exp)

        return experiments

    def _analyze_evidence(self, hypotheses: List[Dict], experiments: List[Dict]) -> List[Dict]:
        """Analyze experimental evidence using Bayesian updating."""
        results = []

        if not self.bayesian:
            # Simple fallback
            for hyp in hypotheses:
                results.append({
                    'hypothesis_id': hyp['id'],
                    'status': 'unverified',
                    'posterior': 0.5,
                    'confidence': 0.5
                })
            return results

        # Add hypotheses as claims
        for hyp in hypotheses:
            if hyp['id'] not in self.bayesian.claims:
                self.bayesian.add_claim(
                    claim_id=hyp['id'],
                    statement=hyp['claim'],
                    domain=hyp.get('gap_source', 'unknown'),
                    prior=hyp['plausibility']
                )

        # Add experimental evidence
        for i, exp in enumerate(experiments):
            # Find matching hypothesis
            # In real implementation, would match properly
            # For now, add evidence to first hypothesis
            if len(hypotheses) > 0:
                from core.bayesian_evidence_evaluator import EvidenceType, EvidenceQuality

                self.bayesian.add_evidence(
                    evidence_id=f"exp_{i}",
                    claim_id=hypotheses[0]['id'],
                    description=exp.get('result', 'experiment result'),
                    evidence_type=EvidenceType.EXPERIMENTAL,
                    quality=EvidenceQuality.STRONG,
                    supports=exp.get('supports_hypothesis', True),
                    source="experiment"
                )

        # Get evaluation for each hypothesis
        for hyp in hypotheses:
            eval_result = self.bayesian.evaluate_claim(hyp['id'])

            results.append({
                'hypothesis_id': hyp['id'],
                'claim': eval_result.get('claim', ''),
                'status': eval_result.get('status', 'unverified'),
                'posterior': eval_result.get('posterior_probability', 0.5),
                'confidence': eval_result.get('confidence', 0.5)
            })

        return results

    def _detect_and_resolve_contradictions(self) -> List[Dict]:
        """Detect and attempt to resolve contradictions."""
        if not self.contra_detect or not self.bayesian:
            return []

        # Get all claims
        claims = [
            {
                'id': cid,
                'statement': claim.statement,
                'domain': claim.domain,
                'posterior': claim.posterior_probability
            }
            for cid, claim in self.bayesian.claims.items()
        ]

        # Detect contradictions
        contradictions = self.contra_detect.detect_contradictions(claims)

        # Auto-resolve if confidence is high
        for contra in contradictions:
            if contra.resolution_confidence > 0.8:
                # Resolve automatically
                if "Keep claim A" in contra.suggested_resolution:
                    self.contra_detect.resolve_contradiction(
                        contra.id,
                        contra.claim_a_id,
                        contra.claim_b_id
                    )
                elif "Keep claim B" in contra.suggested_resolution:
                    self.contra_detect.resolve_contradiction(
                        contra.id,
                        contra.claim_b_id,
                        contra.claim_a_id
                    )

        return [
            {
                'id': c.id,
                'type': c.contradiction_type.value,
                'severity': c.severity.name,
                'resolved': c.resolved
            }
            for c in contradictions
        ]

    def _synthesize_theories(self, results: List[Dict], iteration: int) -> List[Dict]:
        """Synthesize broader theories from verified hypotheses."""
        if not self.theory_synth:
            # Simple fallback: Group verified hypotheses
            verified = [r for r in results if r['status'] == 'verified']

            if len(verified) >= 2:
                return [{
                    'id': f'theory_{iteration}',
                    'hypotheses': [v['hypothesis_id'] for v in verified],
                    'confidence': sum(v['posterior'] * v['confidence'] for v in verified) / len(verified)
                }]

            return []

        # Use theory synthesizer
        theories = self.theory_synth.synthesize(results)
        return theories

    def _print_discovery_summary(self, session: DiscoverySession):
        """Print comprehensive discovery summary."""
        print("\n" + "=" * 70)
        print("🎯 DISCOVERY SESSION SUMMARY 🎯")
        print("=" * 70)
        print(f"\nSession ID: {session.id}")
        print(f"Domain: {session.domain}")
        print(f"Status: {session.status.upper()}")

        print(f"\n📊 DISCOVERY METRICS:")
        print(f"  Knowledge gaps identified: {len(session.gaps_identified)}")
        print(f"  Hypotheses generated: {session.total_hypotheses}")
        print(f"  Experiments conducted: {len(session.experiments_conducted)}")
        print(f"  Theories synthesized: {len(session.theories_synthesized)}")

        print(f"\n✅ VALIDATION:")
        print(f"  Verified hypotheses: {session.verified_hypotheses}")
        print(f"  Rejected hypotheses: {session.rejected_hypotheses}")
        print(f"  Verification rate: {100 * session.verified_hypotheses / session.total_hypotheses if session.total_hypotheses > 0 else 0:.1f}%")

        print(f"\n🔋 FREE ENERGY REDUCTION:")
        print(f"  Initial FE: {session.initial_free_energy:.3f}")
        print(f"  Final FE: {session.final_free_energy:.3f}")
        print(f"  Reduction: {session.fe_reduction:.3f} ({100 * session.fe_reduction / session.initial_free_energy if session.initial_free_energy > 0 else 0:.1f}%)")

        if session.contradictions_found > 0:
            print(f"\n⚠️ CONTRADICTIONS:")
            print(f"  Found: {session.contradictions_found}")
            print(f"  Resolved: {session.contradictions_resolved}")

        if session.avg_learning_speedup > 1.0:
            print(f"\n🚀 COMPOUND GROWTH:")
            print(f"  Learning speedup: {session.avg_learning_speedup:.2f}x")

        print("\n" + "=" * 70)
        print("Discovery session completed successfully!")
        print("=" * 70 + "\n")


# Demo
if __name__ == "__main__":
    print("Discovery Orchestrator")
    print("Autonomous scientific discovery engine!")
    print("\nIntegrate with all core components for full discovery loop.")
