"""
Few-Shot Learning Baseline

Tests LLM with example problems provided in the prompt.
Shows the LLM worked examples before asking it to solve the target problem.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List
from core.llm import LLMBridge
from benchmarks.benchmark_utils import BenchmarkProblem, BenchmarkResult, check_answer


class FewShotBaseline:
    """LLM with worked examples provided in prompt."""

    def __init__(self, llm: LLMBridge):
        self.llm = llm

        # Example problems for few-shot learning
        self.examples = {
            "exponential_equations": """Example: Solve 2^x = 16
Solution: 2^4 = 16, so x = 4""",

            "number_theory": """Example: Express 20 as sum of two primes
Solution: Check primes: 2,3,5,7,11,13,17,19
Pairs: (3,17), (7,13) - so 2 pairs""",

            "factorization": """Example: Prime factorization of 21
Solution: 21 = 3 × 7 (both prime)""",

            "exponential_decay": """Example: 1000 cells double every hour. When was it 250?
Solution: 1000/2 = 500, 500/2 = 250. Answer: 2 hours ago""",

            "sequences": """Example: Collatz sequence for n=5
5 → 16 → 8 → 4 → 2 → 1 (5 steps)""",

            "modular_arithmetic": """Example: Find n where n ≡ 1 (mod 2) and n ≡ 2 (mod 3)
Solution: Try values: n=3,5,7,9,11... n=5 works: 5÷2=2r1, 5÷3=1r2""",

            "quadratic_equations": """Example: Solve x^2 - 3x + 2 = 0
Solution: Factor: (x-1)(x-2) = 0, so x=1 or x=2""",

            "systems_of_equations": """Example: Solve x+y=5, x-y=1
Solution: Add equations: 2x=6, so x=3. Then y=2""",
        }

    def solve_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        """
        Solve problem using LLM with few-shot examples.
        """
        start_time = time.time()

        try:
            # Get relevant example for this category
            example = self.examples.get(problem.category, "")

            system_prompt = """You are a mathematical problem solver.
I'll show you an example problem, then you solve a similar one."""

            user_input = f"""{example}

Now solve this problem:
{problem.problem}

Provide a clear solution and final answer."""

            response = self.llm.generate(system_prompt, user_input, execute=True)
            answer = response.get("text", "")
            error = response.get("error")

            # Check correctness
            correct = check_answer(answer, problem.expected_answer)

            elapsed = time.time() - start_time

            return BenchmarkResult(
                problem_id=problem.id,
                problem_text=problem.problem,
                method="LLM + Few-shot",
                answer=answer[:200],
                expected_answer=problem.expected_answer,
                correct=correct,
                time_seconds=elapsed,
                error=error
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return BenchmarkResult(
                problem_id=problem.id,
                problem_text=problem.problem,
                method="LLM + Few-shot",
                answer="",
                expected_answer=problem.expected_answer,
                correct=False,
                time_seconds=elapsed,
                error=str(e)
            )

    def run_benchmark(self, problems: List[BenchmarkProblem]) -> List[BenchmarkResult]:
        """Run benchmark on all problems."""
        results = []
        for i, problem in enumerate(problems, 1):
            print(f"[LLM + Few-shot] Problem {i}/{len(problems)}: {problem.problem[:50]}...")
            result = self.solve_problem(problem)
            results.append(result)
            status = "✓" if result.correct else "✗"
            print(f"  {status} {result.time_seconds:.1f}s")

        return results


def main():
    """Test few-shot baseline."""
    from benchmarks.benchmark_utils import load_benchmark_problems

    llm = LLMBridge(
        provider="gemini",
        default_model="gemini-1.5-flash"
    )

    baseline = FewShotBaseline(llm)
    problems = load_benchmark_problems()[:3]

    print("Testing Few-Shot Baseline...")
    results = baseline.run_benchmark(problems)

    correct = sum(1 for r in results if r.correct)
    print(f"\nResults: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
