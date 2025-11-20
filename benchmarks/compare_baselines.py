#!/usr/bin/env python3
"""
KV-1 Benchmark Comparison Runner

Compares 4 methods on the 19-problem benchmark suite:
1. KV-1 with Learning (autonomous web-powered learning)
2. LLM Alone (direct queries)
3. LLM + RAG (retrieval per problem, no memory)
4. LLM + Few-shot (examples in prompt)

Usage:
    python benchmarks/compare_baselines.py --provider gemini --api-key YOUR_KEY
    python benchmarks/compare_baselines.py --provider gemini --model gemini-2.5-flash
    python benchmarks/compare_baselines.py --quick  # Test on first 3 problems only
"""

import argparse
import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import List

from core.llm import LLMBridge
from core.web_researcher import WebResearcher
from benchmarks.benchmark_utils import (
    load_benchmark_problems,
    BenchmarkResult,
    format_results_table
)
from benchmarks.llm_alone_baseline import LLMAloneBaseline
from benchmarks.rag_baseline import RAGBaseline
from benchmarks.few_shot_baseline import FewShotBaseline


def run_kv1_benchmark(llm: LLMBridge, web: WebResearcher, problems) -> List[BenchmarkResult]:
    """Run KV-1 self-discovery learning on problems."""
    import time
    from self_discovery_orchestrator import SelfDiscoveryOrchestrator
    from benchmarks.benchmark_utils import check_answer

    results = []

    print("\n" + "=" * 70)
    print("Running KV-1 with Learning...")
    print("=" * 70)

    for i, problem in enumerate(problems, 1):
        print(f"\n[KV-1] Problem {i}/{len(problems)}: {problem.problem[:60]}...")
        start_time = time.time()

        try:
            # Create orchestrator for this problem
            orchestrator = SelfDiscoveryOrchestrator(
                goal=problem.problem,
                ltm_path=f"./benchmark_ltm_{i}.json",  # Separate LTM per problem for fair comparison
                data_dir="./benchmark_data",
                max_depth=5
            )

            # Override with benchmark LLM and web researcher
            orchestrator.llm = llm
            orchestrator.web_researcher = web

            # Run learning
            success = asyncio.run(orchestrator.pursue_goal(max_attempts=10))

            # Get the final answer from journal
            answer = ""
            if orchestrator.journal:
                last_attempt = [j for j in orchestrator.journal if j.get("type") == "goal_attempt"]
                if last_attempt:
                    answer = last_attempt[-1].get("result", "")

            elapsed = time.time() - start_time
            correct = check_answer(answer, problem.expected_answer)

            results.append(BenchmarkResult(
                problem_id=problem.id,
                problem_text=problem.problem,
                method="KV-1 + Learning",
                answer=answer[:200],
                expected_answer=problem.expected_answer,
                correct=correct,
                time_seconds=elapsed,
                error=None if success else "Failed to solve"
            ))

            status = "✓" if correct else "✗"
            print(f"  {status} {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Error: {e}")
            results.append(BenchmarkResult(
                problem_id=problem.id,
                problem_text=problem.problem,
                method="KV-1 + Learning",
                answer="",
                expected_answer=problem.expected_answer,
                correct=False,
                time_seconds=elapsed,
                error=str(e)
            ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run KV-1 benchmark comparisons")
    parser.add_argument("--provider", default="gemini", help="LLM provider (gemini or ollama)")
    parser.add_argument("--model", default=None, help="Model name (e.g., gemini-2.5-flash or qwen3:4b)")
    parser.add_argument("--api-key", default=None, help="API key for provider")
    parser.add_argument("--quick", action="store_true", help="Test on first 3 problems only")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to test: llm-alone, rag, few-shot, kv1 (default: all)")

    args = parser.parse_args()

    # Setup LLM
    print("=" * 70)
    print("KV-1 BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)

    # Initialize components
    llm = LLMBridge(
        provider=args.provider,
        api_key=args.api_key,
        default_model=args.model
    )

    if not llm.is_configured():
        print("❌ LLM not configured properly!")
        print(f"   For Gemini: Set GEMINI_API_KEY or use --api-key")
        print(f"   For Ollama: Make sure Ollama is running")
        sys.exit(1)

    print(f"✓ LLM configured: {llm.describe()}\n")

    web = WebResearcher(
        cache_dir="./benchmark_cache",
        daily_cap=200
    )

    # Load problems
    problems = load_benchmark_problems()
    if args.quick:
        problems = problems[:3]
        print(f"Quick mode: Testing on {len(problems)} problems\n")
    else:
        print(f"Full mode: Testing on {len(problems)} problems\n")

    # Determine which methods to run
    all_methods = ["llm-alone", "few-shot", "rag", "kv1"]
    if args.methods:
        methods_to_run = [m.lower() for m in args.methods]
    else:
        methods_to_run = all_methods

    all_results = []

    # Run baselines
    if "llm-alone" in methods_to_run:
        print("\n" + "=" * 70)
        print("Method 1: LLM Alone")
        print("=" * 70)
        baseline1 = LLMAloneBaseline(llm)
        results1 = baseline1.run_benchmark(problems)
        all_results.extend(results1)

    if "few-shot" in methods_to_run:
        print("\n" + "=" * 70)
        print("Method 2: LLM + Few-Shot Examples")
        print("=" * 70)
        baseline2 = FewShotBaseline(llm)
        results2 = baseline2.run_benchmark(problems)
        all_results.extend(results2)

    if "rag" in methods_to_run:
        print("\n" + "=" * 70)
        print("Method 3: LLM + RAG (Retrieval)")
        print("=" * 70)
        baseline3 = RAGBaseline(llm, web)
        results3 = baseline3.run_benchmark(problems)
        all_results.extend(results3)

    if "kv1" in methods_to_run:
        print("\n" + "=" * 70)
        print("Method 4: KV-1 with Autonomous Learning")
        print("=" * 70)
        results4 = run_kv1_benchmark(llm, web, problems)
        all_results.extend(results4)

    # Print comparison
    print("\n")
    print(format_results_table(all_results))

    # Save detailed results
    output_file = f"BENCHMARK_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(format_results_table(all_results))
        f.write("\n\n")
        f.write("Detailed Results:\n")
        f.write("=" * 70 + "\n")
        for result in all_results:
            f.write(f"\nProblem {result.problem_id}: {result.problem_text}\n")
            f.write(f"Method: {result.method}\n")
            f.write(f"Answer: {result.answer}\n")
            f.write(f"Expected: {result.expected_answer}\n")
            f.write(f"Correct: {result.correct}\n")
            f.write(f"Time: {result.time_seconds:.1f}s\n")
            if result.error:
                f.write(f"Error: {result.error}\n")
            f.write("-" * 70 + "\n")

    print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
