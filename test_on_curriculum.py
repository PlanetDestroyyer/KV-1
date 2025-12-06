"""
Test Problem-Solving Engine on Full Curriculum

Runs the problem solver on all 272 questions from LEARNING_CURRICULUM.md
and tracks performance, learning acceleration, and pattern development.
"""

import sys
import os
import re
import time
import json
from datetime import datetime

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from problem_solving_engine import ProblemSolvingEngine, Problem
from llm import LLMBridge


def parse_curriculum(filepath: str) -> list:
    """Parse the curriculum file and extract all questions."""
    questions = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all numbered questions (pattern: "123. Question text?")
    pattern = r'^(\d+)\.\s+(.+?)$'

    for line in content.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            num, question_text = match.groups()
            questions.append({
                'number': int(num),
                'question': question_text.strip(),
                'domain': infer_domain(question_text)
            })

    return questions


def infer_domain(question: str) -> str:
    """Infer the domain from question text."""
    question_lower = question.lower()

    # Domain keywords
    domains = {
        'arithmetic': ['addition', 'multiplication', 'divide', 'subtract', 'number'],
        'algebra': ['polynomial', 'quadratic', 'equation', 'variable', 'algebra'],
        'geometry': ['triangle', 'circle', 'area', 'volume', 'angle', 'pythagorean'],
        'trigonometry': ['sin', 'cos', 'tan', 'trigonometric', 'radian'],
        'calculus': ['derivative', 'integral', 'limit', 'differential', 'continuous'],
        'number_theory': ['prime', 'divisor', 'modulo', 'congruence', 'riemann'],
        'complex_analysis': ['complex', 'analytic', 'holomorphic', 'residue', 'contour'],
        'topology': ['topology', 'homeomorphism', 'compact', 'connected', 'continuous'],
        'analysis': ['converge', 'series', 'sequence', 'metric', 'banach'],
    }

    for domain, keywords in domains.items():
        if any(keyword in question_lower for keyword in keywords):
            return domain

    return 'general_mathematics'


def estimate_difficulty(question_number: int, total: int) -> float:
    """Estimate difficulty based on position in curriculum (0-1)."""
    # Early questions are easier, later questions are harder
    base_difficulty = question_number / total

    # Add some variation
    import random
    random.seed(question_number)  # Consistent per question
    variation = random.uniform(-0.1, 0.1)

    difficulty = max(0.1, min(0.9, base_difficulty + variation))
    return round(difficulty, 2)


def test_curriculum(
    max_questions: int = None,
    start_from: int = 1,
    save_results: bool = True,
    verbose: bool = True
):
    """
    Test the problem solver on curriculum questions.

    Args:
        max_questions: Maximum number of questions to test (None = all)
        start_from: Question number to start from
        save_results: Save results to JSON
        verbose: Print detailed progress
    """
    print("\n" + "="*80)
    print(" "*15 + "PROBLEM-SOLVING ENGINE - CURRICULUM TEST")
    print("="*80)

    # Parse curriculum
    print("\n[1/5] Loading curriculum...")
    curriculum_path = 'LEARNING_CURRICULUM.md'
    questions = parse_curriculum(curriculum_path)

    total_questions = len(questions)
    print(f"  ✓ Loaded {total_questions} questions from curriculum")

    # Filter questions
    questions = [q for q in questions if q['number'] >= start_from]
    if max_questions:
        questions = questions[:max_questions]

    print(f"  Testing on {len(questions)} questions (#{start_from} to #{questions[-1]['number']})")

    # Initialize LLM
    print("\n[2/5] Initializing LLM Bridge...")
    try:
        llm = LLMBridge(provider="ollama", default_model="qwen3:4b")
        print("  ✓ LLM ready (Qwen3:4b via Ollama)")
        print("  Note: Will use fallback if Ollama unavailable")
    except Exception as e:
        print(f"  ⚠️  LLM initialization warning: {e}")
        llm = None

    # Initialize problem solver
    print("\n[3/5] Initializing Problem-Solving Engine...")
    solver = ProblemSolvingEngine(llm_bridge=llm)

    # Test on questions
    print("\n[4/5] Testing on Questions...")
    print("="*80 + "\n")

    results = []
    start_time = time.time()

    for i, q in enumerate(questions, 1):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Question {i}/{len(questions)} (Curriculum #{q['number']})")
            print(f"{'='*80}")
            print(f"Q: {q['question']}")
            print(f"Domain: {q['domain']}")
            print(f"{'-'*80}")

        # Create problem
        problem = Problem(
            id=f"curriculum_{q['number']}",
            description=q['question'],
            domain=q['domain'],
            difficulty=estimate_difficulty(q['number'], total_questions)
        )

        # Solve
        try:
            solution = solver.solve_problem(problem, verbose=verbose)

            result = {
                'question_number': q['number'],
                'question': q['question'],
                'domain': q['domain'],
                'difficulty': problem.difficulty,
                'solution': solution.solution[:200] + '...' if len(solution.solution) > 200 else solution.solution,
                'confidence': solution.confidence,
                'time_taken': solution.time_taken,
                'patterns_used': len(solution.patterns_used),
                'success': True
            }

            if verbose:
                print(f"\n✓ Solved in {solution.time_taken:.2f}s (confidence: {solution.confidence:.1%})")

        except Exception as e:
            print(f"\n✗ Error solving question: {e}")
            result = {
                'question_number': q['number'],
                'question': q['question'],
                'domain': q['domain'],
                'error': str(e),
                'success': False
            }

        results.append(result)

        # Progress update every 10 questions
        if not verbose and i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(questions) - i)
            print(f"Progress: {i}/{len(questions)} ({i/len(questions)*100:.1f}%) - ETA: {remaining/60:.1f}m")

    # Final statistics
    print("\n" + "="*80)
    print("[5/5] FINAL RESULTS & STATISTICS")
    print("="*80)

    stats = solver.get_statistics()
    total_time = time.time() - start_time

    # Success rate
    successful = sum(1 for r in results if r.get('success', False))
    success_rate = successful / len(results) if results else 0

    print(f"\n📊 TEST SUMMARY:")
    print(f"  Questions attempted: {len(results)}")
    print(f"  Questions solved: {successful}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average time per question: {total_time/len(results):.2f}s")

    if stats['status'] == 'active':
        print(f"\n🧠 LEARNING PROGRESS:")
        print(f"  Problems solved: {stats['problems_solved']}")
        print(f"  Patterns learned: {stats['patterns_learned']}")
        print(f"  Memories stored: {stats['memories_stored']}")
        print(f"  Knowledge concepts: {stats['knowledge_concepts']}")
        print(f"  Average solve time: {stats['avg_solve_time']:.2f}s")
        print(f"  Speedup factor: {stats['speedup_factor']:.2f}x")

        if stats['speedup_factor'] > 1.2:
            print(f"  ✅ LEARNING ACCELERATION DETECTED! {stats['speedup_factor']:.2f}x faster!")

        if stats['compound_growth']['status'] == 'active':
            cg = stats['compound_growth']
            print(f"\n🚀 COMPOUND GROWTH:")
            print(f"  Growth rate: {cg['growth_rate']:.4f}")
            print(f"  Acceleration: {cg['acceleration_percent']:.1f}%")
            print(f"  Learning speedup: {cg['speedup_factor']:.2f}x")

    # Domain breakdown
    print(f"\n📚 PERFORMANCE BY DOMAIN:")
    domain_stats = {}
    for r in results:
        domain = r.get('domain', 'unknown')
        if domain not in domain_stats:
            domain_stats[domain] = {'attempted': 0, 'solved': 0, 'total_time': 0}

        domain_stats[domain]['attempted'] += 1
        if r.get('success'):
            domain_stats[domain]['solved'] += 1
            domain_stats[domain]['total_time'] += r.get('time_taken', 0)

    for domain, stats_dict in sorted(domain_stats.items()):
        success_rate = stats_dict['solved'] / stats_dict['attempted'] if stats_dict['attempted'] > 0 else 0
        avg_time = stats_dict['total_time'] / stats_dict['solved'] if stats_dict['solved'] > 0 else 0
        print(f"  {domain:20s}: {stats_dict['solved']}/{stats_dict['attempted']} ({success_rate:.1%}) - avg {avg_time:.2f}s")

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"curriculum_test_results_{timestamp}.json"

        output = {
            'metadata': {
                'timestamp': timestamp,
                'total_questions': len(results),
                'success_rate': success_rate,
                'total_time_seconds': total_time,
                'start_question': start_from,
                'end_question': questions[-1]['number'] if questions else 0
            },
            'summary': {
                'problems_solved': stats.get('problems_solved', 0),
                'patterns_learned': stats.get('patterns_learned', 0),
                'speedup_factor': stats.get('speedup_factor', 1.0),
                'compound_growth': stats.get('compound_growth', {})
            },
            'domain_stats': domain_stats,
            'results': results
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

    print("\n" + "="*80)
    print(" "*20 + "CURRICULUM TEST COMPLETE!")
    print("="*80 + "\n")

    return results, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test problem solver on curriculum')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of questions to test')
    parser.add_argument('--start', type=int, default=1, help='Question number to start from')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (faster)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')

    args = parser.parse_args()

    try:
        test_curriculum(
            max_questions=args.max,
            start_from=args.start,
            save_results=not args.no_save,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
