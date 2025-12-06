"""
Chain-of-Thought Pattern Miner

Extracts meta-learning patterns from LLM reasoning traces!

Key Innovation:
- Mines successful reasoning strategies from CoT outputs
- Identifies reusable problem-solving patterns
- Builds pattern library for future use
- Accelerates learning through pattern transfer

This implements META-LEARNING - learning how to learn!

Example Patterns:
- "Break complex problems into subproblems"
- "Look for analogies to known solutions"
- "Verify edge cases"
- "Check dimensional consistency"

These patterns ACCELERATE future problem-solving!
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict, Counter
from datetime import datetime
import json
import os


class PatternType(Enum):
    """Types of reasoning patterns."""
    DECOMPOSITION = "decomposition"  # Breaking problems into parts
    ANALOGY = "analogy"              # Using analogies
    VERIFICATION = "verification"    # Checking work
    ABSTRACTION = "abstraction"      # Finding general principles
    CONSTRAINT = "constraint"        # Identifying constraints
    CASE_ANALYSIS = "case_analysis"  # Analyzing different cases
    INDUCTION = "induction"          # Inductive reasoning
    CONTRADICTION = "contradiction"  # Proof by contradiction
    SYMMETRY = "symmetry"            # Exploiting symmetry
    INVARIANT = "invariant"          # Finding invariants


@dataclass
class ReasoningStep:
    """A single step in CoT reasoning."""
    step_number: int
    text: str
    step_type: Optional[PatternType] = None
    confidence: float = 0.0


@dataclass
class ReasoningTrace:
    """A complete CoT reasoning trace."""
    id: str
    problem: str
    solution: str
    steps: List[ReasoningStep] = field(default_factory=list)
    success: bool = True  # Did this reasoning lead to correct solution?
    domain: str = "general"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Pattern:
    """A reusable reasoning pattern."""
    id: str
    pattern_type: PatternType
    description: str
    template: str  # Template for applying this pattern

    # Effectiveness metrics
    success_count: int = 0  # Times this pattern led to success
    failure_count: int = 0  # Times this pattern failed
    success_rate: float = 0.0

    # Usage
    domains: Set[str] = field(default_factory=set)
    example_traces: List[str] = field(default_factory=list)  # Trace IDs

    # Metadata
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    times_used: int = 0

    def update_success_rate(self):
        """Update success rate based on counts."""
        total = self.success_count + self.failure_count
        if total > 0:
            self.success_rate = self.success_count / total


class CoTPatternMiner:
    """
    Mines reasoning patterns from Chain-of-Thought traces.

    META-LEARNING SYSTEM!

    Process:
    1. Collect CoT reasoning traces from LLM
    2. Parse into structured steps
    3. Identify pattern types in each step
    4. Extract reusable patterns
    5. Build pattern library
    6. Apply patterns to new problems

    This enables COMPOUND LEARNING GROWTH:
    - Early: Learn individual facts slowly
    - Later: Apply meta-patterns to learn faster
    - Result: Exponential acceleration!
    """

    def __init__(
        self,
        storage_path: str = "./cot_patterns.json"
    ):
        self.storage_path = storage_path

        # Storage
        self.traces: Dict[str, ReasoningTrace] = {}
        self.patterns: Dict[str, Pattern] = {}

        self.trace_count = 0
        self.pattern_count = 0

        # Pattern keywords for detection
        self.pattern_keywords = {
            PatternType.DECOMPOSITION: [
                'break down', 'split', 'divide', 'subproblem', 'step by step',
                'first', 'then', 'next', 'finally', 'decompose'
            ],
            PatternType.ANALOGY: [
                'similar to', 'like', 'analogous', 'reminds me of', 'just as',
                'compare', 'parallel', 'corresponds to'
            ],
            PatternType.VERIFICATION: [
                'check', 'verify', 'test', 'confirm', 'validate', 'double-check',
                'make sure', 'ensure', 'sanity check'
            ],
            PatternType.ABSTRACTION: [
                'in general', 'pattern', 'principle', 'generalize', 'abstract',
                'rule', 'applies to', 'always true'
            ],
            PatternType.CONSTRAINT: [
                'constraint', 'limitation', 'must', 'cannot', 'requirement',
                'boundary', 'condition', 'restriction'
            ],
            PatternType.CASE_ANALYSIS: [
                'case', 'scenario', 'situation', 'if', 'when', 'either', 'or',
                'consider', 'suppose', 'assume'
            ],
            PatternType.INDUCTION: [
                'base case', 'inductive', 'for all', 'follows that',
                'therefore', 'implies', 'pattern holds'
            ],
            PatternType.CONTRADICTION: [
                'contradiction', 'assume not', 'suppose false', 'impossible',
                'cannot be', 'contradicts'
            ],
            PatternType.SYMMETRY: [
                'symmetr', 'symmetric', 'mirror', 'reverse', 'flip',
                'invariant under', 'same as'
            ],
            PatternType.INVARIANT: [
                'invariant', 'conserved', 'preserved', 'unchanged',
                'constant', 'remains'
            ]
        }

        self.load()

        print("[CoT Pattern Miner] Initialized - Meta-learning active!")
        print(f"  Traces: {len(self.traces)}")
        print(f"  Patterns: {len(self.patterns)}")

    def add_trace(
        self,
        trace_id: str,
        problem: str,
        solution: str,
        cot_text: str,
        success: bool = True,
        domain: str = "general"
    ) -> ReasoningTrace:
        """
        Add a CoT reasoning trace and extract patterns.

        Args:
            trace_id: Unique identifier
            problem: Problem being solved
            solution: Final solution
            cot_text: Chain-of-thought reasoning text
            success: Whether reasoning was successful
            domain: Problem domain

        Returns:
            ReasoningTrace object
        """
        # Parse CoT into steps
        steps = self._parse_cot_steps(cot_text)

        # Classify each step
        for step in steps:
            step.step_type = self._classify_step(step.text)
            step.confidence = 0.8  # Confidence in classification

        # Create trace
        trace = ReasoningTrace(
            id=trace_id,
            problem=problem,
            solution=solution,
            steps=steps,
            success=success,
            domain=domain
        )

        self.traces[trace_id] = trace

        # Extract patterns from successful traces
        if success:
            self._extract_patterns_from_trace(trace)

        print(f"[CoT Miner] Added trace '{trace_id}' with {len(steps)} steps")

        # Save periodically
        if len(self.traces) % 10 == 0:
            self.save()

        return trace

    def _parse_cot_steps(self, cot_text: str) -> List[ReasoningStep]:
        """
        Parse CoT text into structured steps.

        Looks for:
        - Numbered steps (1., 2., etc.)
        - Sentence boundaries
        - Reasoning markers (First, Then, Finally, etc.)
        """
        steps = []

        # Split by numbered markers or newlines
        # Pattern: "1. ...", "Step 1:", etc.
        numbered_pattern = r'(?:^|\n)(?:\d+\.|\d+\)|\*|•|Step \d+:?)'

        parts = re.split(numbered_pattern, cot_text)

        # Clean and filter
        step_num = 1
        for part in parts:
            text = part.strip()
            if len(text) > 10:  # Minimum length
                steps.append(ReasoningStep(
                    step_number=step_num,
                    text=text
                ))
                step_num += 1

        # If no numbered steps found, split by sentences
        if len(steps) == 0:
            sentences = re.split(r'[.!?]+', cot_text)
            step_num = 1
            for sent in sentences:
                text = sent.strip()
                if len(text) > 10:
                    steps.append(ReasoningStep(
                        step_number=step_num,
                        text=text
                    ))
                    step_num += 1

        return steps

    def _classify_step(self, step_text: str) -> Optional[PatternType]:
        """
        Classify a reasoning step by pattern type.

        Uses keyword matching (simple but effective).
        """
        text_lower = step_text.lower()

        # Score each pattern type
        scores = {}
        for pattern_type, keywords in self.pattern_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[pattern_type] = score

        # Return highest scoring pattern
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return None

    def _extract_patterns_from_trace(self, trace: ReasoningTrace):
        """
        Extract reusable patterns from a successful trace.

        Creates pattern templates that can be applied to future problems.
        """
        # Count pattern types used
        pattern_usage = Counter(
            step.step_type for step in trace.steps if step.step_type is not None
        )

        # For each pattern type used, update or create pattern
        for pattern_type, count in pattern_usage.items():
            # Find or create pattern
            pattern_id = f"pattern_{pattern_type.value}"

            if pattern_id not in self.patterns:
                # Create new pattern
                pattern = Pattern(
                    id=pattern_id,
                    pattern_type=pattern_type,
                    description=self._generate_pattern_description(pattern_type),
                    template=self._generate_pattern_template(pattern_type)
                )
                self.patterns[pattern_id] = pattern
                self.pattern_count += 1
            else:
                pattern = self.patterns[pattern_id]

            # Update pattern metrics
            if trace.success:
                pattern.success_count += 1
            else:
                pattern.failure_count += 1

            pattern.update_success_rate()
            pattern.domains.add(trace.domain)
            pattern.times_used += 1

            if len(pattern.example_traces) < 5:  # Keep top 5 examples
                pattern.example_traces.append(trace.id)

    def _generate_pattern_description(self, pattern_type: PatternType) -> str:
        """Generate human-readable description of pattern."""
        descriptions = {
            PatternType.DECOMPOSITION: "Break complex problems into smaller, manageable subproblems",
            PatternType.ANALOGY: "Find similarities to previously solved problems",
            PatternType.VERIFICATION: "Check your work and validate intermediate results",
            PatternType.ABSTRACTION: "Identify general principles that apply broadly",
            PatternType.CONSTRAINT: "Identify and respect problem constraints",
            PatternType.CASE_ANALYSIS: "Analyze different cases or scenarios systematically",
            PatternType.INDUCTION: "Use inductive reasoning to prove general statements",
            PatternType.CONTRADICTION: "Assume the opposite and show it leads to contradiction",
            PatternType.SYMMETRY: "Exploit symmetry to simplify the problem",
            PatternType.INVARIANT: "Find quantities that remain constant"
        }
        return descriptions.get(pattern_type, "Unknown pattern")

    def _generate_pattern_template(self, pattern_type: PatternType) -> str:
        """Generate template for applying pattern."""
        templates = {
            PatternType.DECOMPOSITION: "1. Identify subproblems\n2. Solve each subproblem\n3. Combine solutions",
            PatternType.ANALOGY: "1. Recall similar problems\n2. Map current problem to known solution\n3. Adapt solution",
            PatternType.VERIFICATION: "1. State expected result\n2. Check against result\n3. Verify edge cases",
            PatternType.ABSTRACTION: "1. Identify specific instances\n2. Find common pattern\n3. Generalize",
            PatternType.CONSTRAINT: "1. List all constraints\n2. Check each constraint\n3. Ensure solution respects all",
            PatternType.CASE_ANALYSIS: "1. Enumerate cases\n2. Analyze each case\n3. Combine results",
            PatternType.INDUCTION: "1. Prove base case\n2. Assume for n\n3. Prove for n+1",
            PatternType.CONTRADICTION: "1. Assume negation\n2. Derive contradiction\n3. Conclude original is true",
            PatternType.SYMMETRY: "1. Identify symmetry\n2. Use symmetry to reduce problem\n3. Solve reduced problem",
            PatternType.INVARIANT: "1. Find invariant quantity\n2. Track how it changes\n3. Use to prove result"
        }
        return templates.get(pattern_type, "Apply pattern appropriately")

    def get_recommended_patterns(
        self,
        problem: str,
        domain: str = "general",
        top_k: int = 5
    ) -> List[Pattern]:
        """
        Recommend patterns for a new problem.

        Args:
            problem: Problem description
            domain: Problem domain
            top_k: Number of patterns to recommend

        Returns:
            List of recommended patterns (sorted by relevance)
        """
        # Score patterns by:
        # 1. Domain match
        # 2. Success rate
        # 3. Usage frequency

        scored_patterns = []

        for pattern in self.patterns.values():
            score = 0.0

            # Domain match (highest weight)
            if domain in pattern.domains:
                score += 0.5

            # Success rate
            score += 0.3 * pattern.success_rate

            # Usage frequency (normalized)
            max_usage = max((p.times_used for p in self.patterns.values()), default=1)
            score += 0.2 * (pattern.times_used / max_usage)

            scored_patterns.append((pattern, score))

        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return [p for p, _ in scored_patterns[:top_k]]

    def apply_pattern(
        self,
        pattern_id: str,
        problem: str,
        llm_bridge=None
    ) -> Optional[str]:
        """
        Apply a pattern to solve a problem.

        Args:
            pattern_id: ID of pattern to apply
            problem: Problem to solve
            llm_bridge: LLM for generating solution (optional)

        Returns:
            Generated reasoning following the pattern
        """
        if pattern_id not in self.patterns:
            return None

        pattern = self.patterns[pattern_id]

        # Build prompt incorporating pattern
        prompt = f"""Solve this problem using the following reasoning pattern:

PATTERN: {pattern.description}

TEMPLATE:
{pattern.template}

PROBLEM:
{problem}

Apply the pattern step-by-step:"""

        if llm_bridge:
            response = llm_bridge.generate(prompt)
            return response.get("text", "") if isinstance(response, dict) else str(response)

        return f"Apply pattern: {pattern.description}\n\nTemplate:\n{pattern.template}"

    def get_pattern_statistics(self) -> Dict:
        """Get pattern mining statistics."""
        if len(self.patterns) == 0:
            return {'status': 'no_patterns'}

        # Pattern type distribution
        type_counts = Counter(p.pattern_type for p in self.patterns.values())

        # Success rates
        success_rates = [p.success_rate for p in self.patterns.values() if p.success_count + p.failure_count > 0]
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0

        # Most successful pattern
        best_pattern = max(
            (p for p in self.patterns.values() if p.success_count > 0),
            key=lambda p: p.success_rate,
            default=None
        )

        return {
            'status': 'active',
            'total_traces': len(self.traces),
            'total_patterns': len(self.patterns),
            'successful_traces': sum(1 for t in self.traces.values() if t.success),

            # Pattern metrics
            'pattern_types': {pt.value: count for pt, count in type_counts.items()},
            'avg_success_rate': avg_success,
            'total_pattern_uses': sum(p.times_used for p in self.patterns.values()),

            # Best pattern
            'most_successful_pattern': {
                'type': best_pattern.pattern_type.value,
                'success_rate': best_pattern.success_rate,
                'uses': best_pattern.times_used
            } if best_pattern else None
        }

    def save(self):
        """Save patterns and traces to disk."""
        try:
            data = {
                'patterns': {
                    pid: {
                        'pattern_type': p.pattern_type.value,
                        'description': p.description,
                        'template': p.template,
                        'success_count': p.success_count,
                        'failure_count': p.failure_count,
                        'success_rate': p.success_rate,
                        'domains': list(p.domains),
                        'example_traces': p.example_traces,
                        'times_used': p.times_used,
                        'discovered_at': p.discovered_at
                    }
                    for pid, p in self.patterns.items()
                },
                'trace_summary': {
                    'total': len(self.traces),
                    'successful': sum(1 for t in self.traces.values() if t.success)
                }
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[CoT Miner] Saved {len(self.patterns)} patterns")

        except Exception as e:
            print(f"[CoT Miner] Failed to save: {e}")

    def load(self):
        """Load patterns from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load patterns
            for pid, pdata in data.get('patterns', {}).items():
                pattern = Pattern(
                    id=pid,
                    pattern_type=PatternType(pdata['pattern_type']),
                    description=pdata['description'],
                    template=pdata['template'],
                    success_count=pdata['success_count'],
                    failure_count=pdata['failure_count'],
                    success_rate=pdata['success_rate'],
                    domains=set(pdata['domains']),
                    example_traces=pdata['example_traces'],
                    times_used=pdata['times_used'],
                    discovered_at=pdata['discovered_at']
                )
                self.patterns[pid] = pattern

            print(f"[CoT Miner] Loaded {len(self.patterns)} patterns")

        except Exception as e:
            print(f"[CoT Miner] Failed to load: {e}")

    def demonstrate_pattern_mining(self):
        """Demonstrate pattern mining."""
        print("\n" + "=" * 70)
        print("CoT PATTERN MINER - Demonstration")
        print("=" * 70)

        stats = self.get_pattern_statistics()

        if stats['status'] == 'no_patterns':
            print("\n[!] No patterns mined yet")
            return

        print(f"\n📊 STATISTICS:")
        print(f"  Total traces: {stats['total_traces']}")
        print(f"  Successful traces: {stats['successful_traces']}")
        print(f"  Patterns discovered: {stats['total_patterns']}")
        print(f"  Total pattern uses: {stats['total_pattern_uses']}")
        print(f"  Average success rate: {stats['avg_success_rate']:.1%}")

        print(f"\n🧩 PATTERN TYPES:")
        for ptype, count in stats['pattern_types'].items():
            print(f"  {ptype}: {count}")

        if stats['most_successful_pattern']:
            best = stats['most_successful_pattern']
            print(f"\n⭐ MOST SUCCESSFUL PATTERN:")
            print(f"  Type: {best['type']}")
            print(f"  Success rate: {best['success_rate']:.1%}")
            print(f"  Times used: {best['uses']}")

        print("\n💡 META-LEARNING ACTIVE:")
        print("  Patterns extracted from successful reasoning")
        print("  Can be applied to accelerate future problem-solving")
        print("  Implements compound knowledge growth!")

        print("\n" + "=" * 70)


# Demo
if __name__ == "__main__":
    print("CoT Pattern Miner")
    print("Meta-learning from Chain-of-Thought reasoning!")
    print()

    # Create miner
    miner = CoTPatternMiner()

    # Example CoT trace
    cot_example = """
    1. First, let me break down this problem into smaller parts.
    2. I notice this is similar to a problem I've seen before with prime numbers.
    3. Let me verify my approach by checking a simple case.
    4. The general principle here is that we need to find an invariant.
    5. Finally, let me check that this works for edge cases.
    """

    # Add trace
    miner.add_trace(
        trace_id="example_1",
        problem="Find pattern in prime numbers",
        solution="Pattern found",
        cot_text=cot_example,
        success=True,
        domain="number_theory"
    )

    # Demonstrate
    miner.demonstrate_pattern_mining()
