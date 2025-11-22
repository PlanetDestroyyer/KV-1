#!/usr/bin/env python3
"""
Automated Learning Curriculum Runner

Runs the complete learning curriculum automatically, building toward
the mathematical knowledge needed to attempt the Riemann Hypothesis.

Usage:
    # Run full curriculum
    python run_curriculum.py --phase all

    # Run specific phase
    python run_curriculum.py --phase 1

    # Continue from where you left off
    python run_curriculum.py --resume

    # Skip failed questions
    python run_curriculum.py --resume --skip-failed
"""

import asyncio
import argparse
import json
import os
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from core.logger import setup_logging

# All curriculum questions organized by phase
CURRICULUM = {
    "Phase 1: Foundational Mathematics": [
        # Arithmetic & Algebra
        "What is addition and how does it work?",
        "What is multiplication and how does it relate to addition?",
        "What are prime numbers and why are they important?",
        "How do you find the greatest common divisor of two numbers?",
        "What is the distributive property and why does it work?",
        "What are exponents and how do they work?",
        "What is a square root and how do you calculate it?",
        "What are negative numbers and how do they behave?",
        "What is the quadratic formula and how do you derive it?",
        "What are polynomials and how do you factor them?",
        "What is the binomial theorem?",
        "What are logarithms and how do they relate to exponents?",
        "What is the difference between rational and irrational numbers?",
        "What makes pi (π) special?",
        "What is e (Euler's number) and why is it important?",
        "What are imaginary numbers and why do we need them?",
        "What are complex numbers and how do they work?",
        "What is the fundamental theorem of algebra?",
        "What are sequences and series?",
        "What is an arithmetic progression vs geometric progression?",
        # Geometry & Trigonometry
        "What is the Pythagorean theorem and how do you prove it?",
        "What are the basic trigonometric functions (sin, cos, tan)?",
        "How do you prove the trigonometric identity: sin²θ + cos²θ = 1?",
        "What are the angle addition formulas for sine and cosine?",
        "What is the unit circle and why is it useful?",
        "What are radians and how do they relate to degrees?",
        "What is the law of sines and when do you use it?",
        "What is the law of cosines and how does it generalize Pythagorean theorem?",
        "What are polar coordinates and how do they differ from Cartesian?",
        "What is Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)?",
        "What are conic sections (parabola, ellipse, hyperbola)?",
        "What is the area of a circle and how do you derive it?",
        "What is the volume of a sphere?",
        "What are vectors and how do you work with them?",
        "What is the dot product and cross product of vectors?",
    ],

    "Phase 2: Calculus & Analysis": [
        # Limits & Continuity
        "What is a limit and why is it important?",
        "What does it mean for a function to be continuous?",
        "What is the epsilon-delta definition of a limit?",
        "What are infinite limits and limits at infinity?",
        "What is L'Hôpital's rule and when do you use it?",
        "What is the squeeze theorem?",
        "What are one-sided limits?",
        "What is the intermediate value theorem?",
        "What makes a function differentiable?",
        "What is the formal definition of continuity?",
        # Derivatives
        "What is a derivative and what does it represent?",
        "How do you derive the power rule for derivatives?",
        "What is the product rule for derivatives?",
        "What is the quotient rule for derivatives?",
        "What is the chain rule and why is it so powerful?",
        "How do you find the derivative of sin(x)?",
        "How do you find the derivative of e^x?",
        "How do you find the derivative of ln(x)?",
        "What is implicit differentiation?",
        "What are higher-order derivatives?",
        "What is the mean value theorem?",
        "How do you use derivatives to find maxima and minima?",
        "What is a critical point?",
        "What is concavity and how do you test for it?",
        "What are inflection points?",
        # Integrals
        "What is an integral and what does it represent?",
        "What is the fundamental theorem of calculus?",
        "How do you integrate x^n?",
        "What is integration by substitution?",
        "What is integration by parts?",
        "How do you integrate sin(x) and cos(x)?",
        "How do you integrate e^x?",
        "How do you integrate 1/x?",
        "What are definite vs indefinite integrals?",
        "How do you calculate the area under a curve?",
        "How do you calculate the volume of a solid of revolution?",
        "What are improper integrals?",
        "What is the trapezoid rule for numerical integration?",
        "What is Simpson's rule?",
        "How do you integrate rational functions using partial fractions?",
        # Series & Sequences
        "What is the sum of an infinite geometric series?",
        "What are Taylor series and how do they work?",
        "What is the Maclaurin series?",
        "What is the Taylor series for e^x?",
        "What is the Taylor series for sin(x) and cos(x)?",
        "What are convergence tests for series?",
        "What is the ratio test for convergence?",
        "What is the root test?",
        "What is absolute vs conditional convergence?",
        "What is a power series?",
    ],

    "Phase 3: Advanced Mathematics": [
        # Linear Algebra
        "What is a matrix and how do you multiply matrices?",
        "What is the determinant of a matrix?",
        "What is an inverse matrix and how do you find it?",
        "What are eigenvalues and eigenvectors?",
        "What is a vector space?",
        "What is a basis for a vector space?",
        "What is linear independence?",
        "What is the rank of a matrix?",
        "What is the trace of a matrix?",
        "What are orthogonal matrices?",
        # Discrete Mathematics
        "What is mathematical induction and how does it work?",
        "What are combinations and permutations?",
        "What is the binomial coefficient and how do you calculate it?",
        "What is the pigeonhole principle?",
        "What are recurrence relations?",
        "What is modular arithmetic?",
        "What is Fermat's Little Theorem?",
        "What is Euler's totient function φ(n)?",
        "What is the Chinese Remainder Theorem?",
        "What are groups, rings, and fields in abstract algebra?",
    ],

    "Phase 4: Number Theory": [
        "What is the division algorithm?",
        "What is the Euclidean algorithm for finding GCD?",
        "What is the fundamental theorem of arithmetic?",
        "How many primes are there? (Infinitude of primes)",
        "What is the sieve of Eratosthenes?",
        "What are twin primes?",
        "What is the prime number theorem?",
        "What is the distribution of prime numbers?",
        "What is a Diophantine equation?",
        "What is the Riemann zeta function ζ(s)?",
        "What are the trivial zeros of ζ(s)?",
        "What is the critical strip for the zeta function?",
        "What is the Euler product formula for ζ(s)?",
        "What does ζ(s) have to do with prime numbers?",
        "What is the prime counting function π(x)?",
    ],

    "Phase 5: Complex Analysis": [
        "What is a complex function?",
        "What does it mean for a complex function to be analytic?",
        "What are the Cauchy-Riemann equations?",
        "What is a holomorphic function?",
        "What are singularities in complex analysis?",
        "What is a pole of a function?",
        "What is a residue?",
        "What is the residue theorem?",
        "What is Cauchy's integral theorem?",
        "What is Cauchy's integral formula?",
        "What is analytic continuation?",
        "How is ζ(s) extended to the whole complex plane?",
        "What is the functional equation relating ζ(s) and ζ(1-s)?",
        "What is the critical line Re(s) = 1/2?",
        "How do you count zeros using contour integration?",
    ],

    "Phase 6: Toward Riemann Hypothesis": [
        "What exactly is the Riemann Hypothesis?",
        "Why is the Riemann Hypothesis important?",
        "What are the nontrivial zeros of ζ(s)?",
        "Where are the known nontrivial zeros located?",
        "What is the zero-free region of ζ(s)?",
        "What would happen if the Riemann Hypothesis is true?",
        "What is the connection between ζ(s) zeros and prime distribution?",
        "What is the explicit formula connecting zeros to primes?",
        "What are the generalized Riemann hypotheses?",
        "What approaches have been tried to prove RH?",
    ],
}

# Flatten all questions with phase labels
ALL_QUESTIONS = []
for phase, questions in CURRICULUM.items():
    for q in questions:
        ALL_QUESTIONS.append((phase, q))

# Progress tracking
PROGRESS_FILE = "curriculum_progress.json"


def load_progress():
    """Load learning progress from disk."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "last_index": 0,
        "completed": [],
        "failed": [],
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def save_progress(progress):
    """Save learning progress to disk."""
    progress["updated_at"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


async def ask_question(question, max_attempts=3):
    """
    Ask a single question to the system.

    Returns: (success: bool, output: str)
    """
    import subprocess

    cmd = [
        "python", "run_self_discovery.py",
        question,
        "--max-attempts", str(max_attempts)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        return success, output
    except subprocess.TimeoutExpired:
        return False, "Timeout after 5 minutes"
    except Exception as e:
        return False, str(e)


async def run_curriculum(args):
    """Run the learning curriculum."""

    # Set up logging to file (all output saved to ./logs/)
    phase_name = f"phase{args.phase}" if args.phase != "all" else "full_curriculum"
    log_file = setup_logging(session_name=f"curriculum_{phase_name}")
    print(f"[+] All output being saved to: {log_file}")

    progress = load_progress()

    # Determine which questions to ask
    if args.resume:
        start_idx = progress["last_index"]
        print(f"[+] Resuming from question {start_idx + 1}/{len(ALL_QUESTIONS)}")
    elif args.phase == "all":
        start_idx = 0
        print(f"[+] Running full curriculum ({len(ALL_QUESTIONS)} questions)")
    else:
        # Find questions for specific phase
        phase_name = f"Phase {args.phase}:"
        start_idx = 0
        for i, (phase, _) in enumerate(ALL_QUESTIONS):
            if phase.startswith(phase_name):
                start_idx = i
                break
        print(f"[+] Running {phase_name}")

    total_questions = len(ALL_QUESTIONS)

    # Issue #10: Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n[!] Interrupted by user! Saving progress...")
        save_progress(progress)
        print(f"[+] Progress saved. Resume with: python run_curriculum.py --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for i in range(start_idx, total_questions):
            phase, question = ALL_QUESTIONS[i]

            print("\n" + "="*70)
            print(f"[{i+1}/{total_questions}] {phase}")
            print("="*70)
            print(f"Question: {question}")
            print()

            # Ask the question
            success, output = await ask_question(
                question,
                args.max_attempts
            )

            if success:
                print(f"[✓] Successfully learned!")
                progress["completed"].append(i)
            else:
                print(f"[✗] Failed to learn")
                print(f"Error: {output[:500]}")
                progress["failed"].append(i)

                if not args.skip_failed:
                    print("\n[!] Stopping due to failure (use --skip-failed to continue)")
                    break

            # Update progress
            progress["last_index"] = i + 1
            save_progress(progress)

            # Rate limiting
            if args.delay > 0:
                print(f"[.] Waiting {args.delay}s before next question...")
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user! Saving progress...")
        save_progress(progress)
        print(f"[+] Progress saved. Resume with: python run_curriculum.py --resume")
        sys.exit(0)

    # Final summary
    print("\n" + "="*70)
    print("LEARNING CURRICULUM COMPLETE")
    print("="*70)
    print(f"Total questions: {total_questions}")
    print(f"Completed: {len(progress['completed'])}")
    print(f"Failed: {len(progress['failed'])}")
    print(f"Success rate: {len(progress['completed'])/total_questions*100:.1f}%")
    print("="*70)
    print("\n[i] Knowledge stored in ltm_memory.json")
    print("[i] Use 'python run_self_discovery.py --help' for more options")


def main():
    parser = argparse.ArgumentParser(
        description="Run the KV-1 learning curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full curriculum
  python run_curriculum.py --phase all

  # Run just Phase 1 (foundational)
  python run_curriculum.py --phase 1

  # Resume from where you left off
  python run_curriculum.py --resume

  # Skip failed questions and keep going
  python run_curriculum.py --resume --skip-failed

  # With longer delay (avoid rate limits)
  python run_curriculum.py --phase 1 --delay 10
        """
    )

    parser.add_argument("--phase", type=str, default="all",
                       help="Which phase to run (1-6 or 'all')")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--max-attempts", type=int, default=3,
                       help="Max attempts per question")
    parser.add_argument("--delay", type=int, default=5,
                       help="Delay between questions (seconds)")
    parser.add_argument("--skip-failed", action="store_true",
                       help="Continue even if questions fail")

    args = parser.parse_args()

    print("="*70)
    print("KV-1 LEARNING CURRICULUM")
    print("="*70)
    print(f"Provider: Ollama (qwen3:4b)")
    print(f"Max attempts per question: {args.max_attempts}")
    print(f"Delay between questions: {args.delay}s")
    print("="*70)

    asyncio.run(run_curriculum(args))


if __name__ == "__main__":
    main()
