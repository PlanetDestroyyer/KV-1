"""
Shared utilities for benchmarking KV-1 against baselines.
"""

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkProblem:
    """A single problem for benchmarking."""
    id: int
    difficulty: str
    category: str
    problem: str
    expected_answer: str
    notes: str


@dataclass
class BenchmarkResult:
    """Result of running a benchmark on one problem."""
    problem_id: int
    problem_text: str
    method: str
    answer: str
    expected_answer: str
    correct: bool
    time_seconds: float
    error: Optional[str] = None


def load_benchmark_problems() -> List[BenchmarkProblem]:
    """Load the 19 benchmark problems."""
    problems = [
        BenchmarkProblem(
            id=1,
            difficulty="🔥🔥🔥",
            category="exponential_equations",
            problem="Find the value of x where x^x = 256",
            expected_answer="4",
            notes="Requires exponential reasoning and systematic testing"
        ),
        BenchmarkProblem(
            id=2,
            difficulty="🔥🔥",
            category="number_theory",
            problem="Express 100 as the sum of two prime numbers in all possible ways",
            expected_answer="6 pairs: (3,97), (11,89), (17,83), (29,71), (41,59), (47,53)",
            notes="Goldbach conjecture verification"
        ),
        BenchmarkProblem(
            id=3,
            difficulty="🔥🔥🔥",
            category="factorization",
            problem="Find the prime factorization of 8633",
            expected_answer="89 × 97",
            notes="Large number factorization"
        ),
        BenchmarkProblem(
            id=4,
            difficulty="🔥🔥",
            category="exponential_decay",
            problem="A bacteria culture has 10,000 cells and doubles every hour. At what time in the past did it have 625 cells?",
            expected_answer="4 hours ago",
            notes="Inverse exponential growth"
        ),
        BenchmarkProblem(
            id=5,
            difficulty="🔥🔥🔥",
            category="sequences",
            problem="For n=27, show the full Collatz sequence until reaching 1. How many steps?",
            expected_answer="111 steps",
            notes="Long iterative algorithm"
        ),
        BenchmarkProblem(
            id=6,
            difficulty="🔥🔥🔥🔥",
            category="modular_arithmetic",
            problem="Find the smallest positive integer n where: n ≡ 2 (mod 3), n ≡ 3 (mod 5), n ≡ 2 (mod 7)",
            expected_answer="23",
            notes="Chinese Remainder Theorem"
        ),
        BenchmarkProblem(
            id=7,
            difficulty="🔥🔥",
            category="quadratic_equations",
            problem="Solve x^2 - 5x + 6 = 0",
            expected_answer="x = 2 and x = 3",
            notes="Basic quadratic factoring"
        ),
        BenchmarkProblem(
            id=8,
            difficulty="🔥🔥🔥",
            category="systems_of_equations",
            problem="Solve the system: 2x + 3y = 13, 3x - y = 3",
            expected_answer="x = 2, y = 3",
            notes="Two equations, two unknowns"
        ),
    ]
    return problems


def check_answer(answer: str, expected: str) -> bool:
    """
    Check if answer matches expected (fuzzy matching).
    """
    answer_clean = answer.lower().strip()
    expected_clean = expected.lower().strip()

    # Exact match
    if answer_clean == expected_clean:
        return True

    # Check if key numbers match
    import re
    answer_nums = set(re.findall(r'\d+', answer))
    expected_nums = set(re.findall(r'\d+', expected))

    if answer_nums and expected_nums:
        # At least 80% of expected numbers should appear in answer
        intersection = answer_nums & expected_nums
        if len(intersection) >= len(expected_nums) * 0.8:
            return True

    return False


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a nice table."""

    # Group by method
    methods = {}
    for result in results:
        if result.method not in methods:
            methods[result.method] = []
        methods[result.method].append(result)

    # Calculate stats per method
    stats = {}
    for method, method_results in methods.items():
        total = len(method_results)
        correct = sum(1 for r in method_results if r.correct)
        avg_time = sum(r.time_seconds for r in method_results) / total if total > 0 else 0

        stats[method] = {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "avg_time": avg_time
        }

    # Format table
    output = []
    output.append("\n" + "=" * 70)
    output.append("KV-1 BENCHMARK COMPARISON RESULTS")
    output.append("=" * 70)
    output.append("")
    output.append(f"{'Method':<25} {'Accuracy':<12} {'Time (avg)':<12} {'Score'}")
    output.append("-" * 70)

    for method, s in sorted(stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        accuracy_pct = s['accuracy'] * 100
        output.append(
            f"{method:<25} {s['correct']}/{s['total']} ({accuracy_pct:>5.1f}%)   "
            f"{s['avg_time']:>6.1f}s        {accuracy_pct:>5.1f}/100"
        )

    output.append("=" * 70)

    return "\n".join(output)
