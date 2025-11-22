"""
Tensor Reasoning System

Complete integration of all tensor-based mathematical reasoning components.

This is the MAIN INTERFACE for tensor reasoning!

Components:
1. Mathematical Primitives (axioms, operations)
2. Symbolic Math Engine (SymPy reasoning)
3. Geometric Knowledge Space (Riemannian manifold)
4. Exploration Engine (proof search)

Usage:
    system = TensorReasoningSystem()
    result = await system.solve("Prove all primes > 2 are odd")
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from core.math_primitives import MathematicalPrimitives, MathDomain
from core.symbolic_math_engine import SymbolicMathEngine, ProofStatus
from core.geometric_knowledge_space import RiemannianKnowledgeManifold
from core.mathematical_exploration_engine import MathematicalExplorationEngine, ExplorationResult


@dataclass
class ReasoningResult:
    """Result of tensor reasoning"""
    success: bool
    answer: str
    proof_steps: List[str]
    confidence: float
    method: str
    exploration_stats: Dict[str, Any]


class TensorReasoningSystem:
    """
    Complete tensor-based mathematical reasoning system.

    Thinks in PURE MATHEMATICS:
    - Symbolic reasoning (SymPy)
    - Geometric intuition (tensors + manifold)
    - Exhaustive exploration (proof search)
    - Pattern learning (meta-learning)

    NOT just an LLM wrapper - this is genuine mathematical reasoning!
    """

    def __init__(self, dimension: int = 768):
        print("[🚀] Initializing Tensor Reasoning System...")

        # Initialize components
        self.primitives = MathematicalPrimitives()
        self.symbolic = SymbolicMathEngine()
        self.geometry = RiemannianKnowledgeManifold(dimension=dimension)
        self.explorer = MathematicalExplorationEngine(
            symbolic_engine=self.symbolic,
            geometric_space=self.geometry,
            max_depth=20,
            max_states=100000
        )

        print("[✅] Symbolic Math Engine initialized")
        print("[✅] Geometric Knowledge Space initialized")
        print("[✅] Exploration Engine initialized")
        print(f"[✅] Knowledge Manifold: {dimension}-dimensional")

        # Initialize with basic concepts
        self._initialize_basic_concepts()

    def _initialize_basic_concepts(self):
        """Embed basic mathematical concepts in manifold"""
        print("\n[📚] Embedding basic mathematical concepts...")

        basic_concepts = [
            ('prime', {'domain': 'number_theory', 'is_prime': True, 'complexity': 0.3}),
            ('composite', {'domain': 'number_theory', 'is_prime': False, 'complexity': 0.3}),
            ('even', {'domain': 'number_theory', 'is_even': True, 'complexity': 0.1}),
            ('odd', {'domain': 'number_theory', 'is_even': False, 'complexity': 0.1}),
            ('integer', {'domain': 'number_theory', 'complexity': 0.2}),
            ('rational', {'domain': 'number_theory', 'complexity': 0.4}),
        ]

        for name, props in basic_concepts:
            self.geometry.embed_concept(name, None, props)

        print(f"[✅] Embedded {len(basic_concepts)} basic concepts")

    async def solve(
        self,
        problem: str,
        method: str = 'auto',
        timeout: Optional[int] = None
    ) -> ReasoningResult:
        """
        Solve a mathematical problem using tensor reasoning.

        Args:
            problem: Mathematical problem in natural language
            method: 'auto', 'symbolic', 'computational', or 'exploration'
            timeout: Optional timeout in seconds

        Returns:
            ReasoningResult with solution
        """
        print(f"\n{'='*60}")
        print(f"[🧮] TENSOR REASONING SYSTEM")
        print(f"{'='*60}")
        print(f"Problem: {problem}")
        print(f"Method: {method}")
        print(f"{'='*60}\n")

        # Parse problem
        symbolic_form = self.symbolic.parse_mathematical_statement(problem)

        if symbolic_form is None:
            return ReasoningResult(
                success=False,
                answer="Could not parse problem",
                proof_steps=["Parse failed"],
                confidence=0.0,
                method="failed",
                exploration_stats={}
            )

        # Choose method
        if method == 'auto':
            # Try computational first (fast)
            result = self.symbolic.verify_by_computation(symbolic_form, test_range=100)

            if result.status == ProofStatus.PROVEN:
                return self._format_result(result, "computational")

            # Try symbolic next
            result = self.symbolic.prove_by_contradiction(symbolic_form)

            if result.status == ProofStatus.PROVEN:
                return self._format_result(result, "symbolic")

            # Fall back to exploration
            method = 'exploration'

        if method == 'computational':
            result = self.symbolic.verify_by_computation(symbolic_form, test_range=1000)
            return self._format_result(result, "computational")

        elif method == 'symbolic':
            result = self.symbolic.prove_by_contradiction(symbolic_form)
            return self._format_result(result, "symbolic")

        elif method == 'exploration':
            exploration_result = await self.explorer.explore_conjecture(
                problem,
                timeout_seconds=timeout
            )

            return ReasoningResult(
                success=exploration_result.success,
                answer="Proof found!" if exploration_result.success else "No proof found",
                proof_steps=exploration_result.proof.steps if exploration_result.proof else [],
                confidence=exploration_result.proof.confidence if exploration_result.proof else 0.0,
                method="exploration",
                exploration_stats={
                    'states_explored': exploration_result.states_explored,
                    'time_seconds': exploration_result.time_seconds
                }
            )

        return ReasoningResult(
            success=False,
            answer="Unknown method",
            proof_steps=[],
            confidence=0.0,
            method="failed",
            exploration_stats={}
        )

    def _format_result(self, proof_result, method: str) -> ReasoningResult:
        """Format proof result as reasoning result"""
        return ReasoningResult(
            success=proof_result.status == ProofStatus.PROVEN,
            answer=str(proof_result.conclusion),
            proof_steps=proof_result.steps,
            confidence=proof_result.confidence,
            method=method,
            exploration_stats={}
        )

    async def explore_goldbach(self, limit: int = 10000) -> Dict[str, Any]:
        """
        Explore Goldbach's conjecture computationally.

        This demonstrates the power of exhaustive exploration!
        """
        return await self.explorer.explore_goldbach_computational(limit)

    def find_relations(self, concept_a: str, concept_b: str) -> List[Dict]:
        """
        Find mathematical relations between concepts.

        Uses both symbolic and geometric reasoning!
        """
        print(f"\n[🔍] Finding relations between '{concept_a}' and '{concept_b}'...")

        # Symbolic relations
        symbolic_rels = self.symbolic.discover_relations(concept_a, concept_b)

        # Geometric relations
        geometric_rels = []
        if concept_a in self.geometry.concepts and concept_b in self.geometry.concepts:
            distance = self.geometry.riemannian_distance(concept_a, concept_b)
            geometric_rels.append({
                'type': 'geometric_distance',
                'value': distance,
                'interpretation': 'close' if distance < 1.0 else 'distant'
            })

        all_relations = symbolic_rels + geometric_rels

        print(f"[✓] Found {len(all_relations)} relations")
        for rel in all_relations:
            print(f"   - {rel}")

        return all_relations

    def get_learning_path(self, start: str, goal: str) -> List[str]:
        """
        Find optimal learning path from start to goal.

        Uses geodesic in knowledge manifold!
        """
        print(f"\n[🎯] Finding learning path: {start} → {goal}")

        geodesic = self.geometry.find_geodesic(start, goal)

        if geodesic:
            print(f"[✓] Path found with {len(geodesic.steps)} steps:")
            for i, step in enumerate(geodesic.steps, 1):
                print(f"   {i}. {step}")
            print(f"[📊] Total distance: {geodesic.distance:.3f}")

            return geodesic.steps
        else:
            print("[!] Could not find path")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'geometric_space': self.geometry.get_stats(),
            'primitives': {
                'axioms': len(self.primitives.axioms),
                'operations': len(self.primitives.operations),
                'theorems': len(self.primitives.theorems)
            }
        }

    def print_stats(self):
        """Print system statistics"""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("📊 TENSOR REASONING SYSTEM STATS")
        print("="*60)

        print(f"\nGeometric Knowledge Space:")
        print(f"  Concepts: {stats['geometric_space']['num_concepts']}")
        print(f"  Dimension: {stats['geometric_space']['dimension']}")
        print(f"  Device: {stats['geometric_space']['device']}")
        print(f"  Cache size: {stats['geometric_space']['cache_size']}")

        print(f"\nMathematical Primitives:")
        print(f"  Axioms: {stats['primitives']['axioms']}")
        print(f"  Operations: {stats['primitives']['operations']}")
        print(f"  Theorems: {stats['primitives']['theorems']}")

        print("="*60)


# Example usage
async def example_usage():
    """Demonstrate the tensor reasoning system"""

    system = TensorReasoningSystem()

    # Example 1: Simple verification
    print("\n" + "="*60)
    print("EXAMPLE 1: Verify computational statement")
    print("="*60)

    result = await system.solve("4 + 6 = 10", method='computational')
    print(f"\nResult: {result.answer}")
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence}")

    # Example 2: Explore Goldbach
    print("\n" + "="*60)
    print("EXAMPLE 2: Explore Goldbach's Conjecture")
    print("="*60)

    goldbach_result = await system.explore_goldbach(limit=100)

    # Example 3: Find relations
    print("\n" + "="*60)
    print("EXAMPLE 3: Find Relations Between Concepts")
    print("="*60)

    relations = system.find_relations('prime', 'composite')

    # Stats
    system.print_stats()


if __name__ == "__main__":
    asyncio.run(example_usage())
