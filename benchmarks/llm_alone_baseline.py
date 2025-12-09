"""
LLM-Alone Baseline

Tests the LLM directly without any learning, retrieval, or examples.
This is the simplest baseline - just ask the question and get an answer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List
from core.llm import LLMBridge
from benchmarks.benchmark_utils import BenchmarkProblem, BenchmarkResult, check_answer


class LLMAloneBaseline:
    """Direct LLM queries without any assistance."""

    def __init__(self, llm: LLMBridge):
        self.llm = llm

    def solve_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        """
        Solve a problem using only the LLM, no learning or retrieval.
        """
        start_time = time.time()

        system_prompt = "You are a mathematical problem solver. Provide clear, accurate answers."
        user_input = f"Problem: {problem.problem}\n\nProvide the answer clearly and concisely."

        try:
            response = self.llm.generate(system_prompt, user_input, execute=True)
            answer = response.get("text", "")
            error = response.get("error")

            # Check correctness
            correct = check_answer(answer, problem.expected_answer)

            elapsed = time.time() - start_time

            return BenchmarkResult(
                problem_id=problem.id,
                problem_text=problem.problem,
                method="LLM Alone",
                answer=answer[:200],  # Truncate long answers
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
                method="LLM Alone",
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
            print(f"[LLM Alone] Problem {i}/{len(problems)}: {problem.problem[:50]}...")
            result = self.solve_problem(problem)
            results.append(result)
            status = "✓" if result.correct else "✗"
            print(f"  {status} {result.time_seconds:.1f}s")

        return results


def main():
    """Test LLM-alone baseline."""
    from benchmarks.benchmark_utils import load_benchmark_problems

    # Use Gemini for testing
    llm = LLMBridge(
        provider="gemini",
        default_model="gemini-1.5-flash"  # Will use gemini-2.5-flash if available
    )

    baseline = LLMAloneBaseline(llm)
    problems = load_benchmark_problems()[:3]  # Test on first 3 problems

    print("Testing LLM-Alone Baseline...")
    results = baseline.run_benchmark(problems)

    # Summary
    correct = sum(1 for r in results if r.correct)
    print(f"\nResults: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
