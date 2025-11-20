"""
RAG (Retrieval Augmented Generation) Baseline

Tests LLM + web retrieval for each problem.
Like KV-1, but WITHOUT persistent learning - each problem starts fresh.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List
from core.llm import LLMBridge
from core.web_researcher import WebResearcher
from benchmarks.benchmark_utils import BenchmarkProblem, BenchmarkResult, check_answer


class RAGBaseline:
    """LLM + retrieval for each problem, but no memory between problems."""

    def __init__(self, llm: LLMBridge, web_researcher: WebResearcher):
        self.llm = llm
        self.web = web_researcher

    def solve_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        """
        Solve problem using LLM + web retrieval.
        Searches web for relevant info, then asks LLM to solve.
        """
        start_time = time.time()

        try:
            # Step 1: Extract key concepts from problem
            system_prompt = "Extract 1-2 key mathematical concepts from this problem."
            user_input = f"Problem: {problem.problem}\n\nWhat concepts are needed?"

            response = self.llm.generate(system_prompt, user_input, execute=True)
            concepts_text = response.get("text", "")

            # Step 2: Search web for first concept
            # Simple extraction: take first few words
            search_query = concepts_text.split('\n')[0][:50]
            web_result = self.web.fetch(search_query, mode="scrape")

            context = ""
            if web_result:
                context = web_result.text[:1000]  # First 1000 chars

            # Step 3: Ask LLM to solve with context
            system_prompt = "You are a mathematical problem solver. Use the provided context if helpful."
            user_input = f"""Context from web:
{context}

Problem: {problem.problem}

Solve this problem clearly and provide the final answer."""

            response = self.llm.generate(system_prompt, user_input, execute=True)
            answer = response.get("text", "")
            error = response.get("error")

            # Check correctness
            correct = check_answer(answer, problem.expected_answer)

            elapsed = time.time() - start_time

            return BenchmarkResult(
                problem_id=problem.id,
                problem_text=problem.problem,
                method="LLM + RAG",
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
                method="LLM + RAG",
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
            print(f"[LLM + RAG] Problem {i}/{len(problems)}: {problem.problem[:50]}...")
            result = self.solve_problem(problem)
            results.append(result)
            status = "✓" if result.correct else "✗"
            print(f"  {status} {result.time_seconds:.1f}s")

        return results


def main():
    """Test RAG baseline."""
    from benchmarks.benchmark_utils import load_benchmark_problems

    llm = LLMBridge(
        provider="gemini",
        default_model="gemini-1.5-flash"
    )
    web = WebResearcher(
        cache_dir="./benchmark_cache",
        daily_cap=100
    )

    baseline = RAGBaseline(llm, web)
    problems = load_benchmark_problems()[:3]

    print("Testing RAG Baseline...")
    results = baseline.run_benchmark(problems)

    correct = sum(1 for r in results if r.correct)
    print(f"\nResults: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
