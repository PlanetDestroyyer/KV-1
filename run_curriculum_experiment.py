#!/usr/bin/env python3
"""
Simple runner for curriculum-based learning experiment.

Usage:
    python run_curriculum_experiment.py              # Run 30 iterations (default)
    python run_curriculum_experiment.py 50           # Run 50 iterations
    python run_curriculum_experiment.py --quick      # Run 10 iterations for quick test
"""

import argparse
import asyncio
from curriculum_orchestrator import main_curriculum_experiment


def main():
    parser = argparse.ArgumentParser(description="Run curriculum-based learning experiment")
    parser.add_argument(
        "iterations",
        type=int,
        nargs="?",
        default=30,
        help="Number of learning iterations (default: 30)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (10 iterations)"
    )

    args = parser.parse_args()

    iterations = 10 if args.quick else args.iterations

    print(f"Starting curriculum experiment with {iterations} iterations...")
    print(f"This will progress through: Language ’ Numbers ’ Algebra ’ Calculus ’ Thermodynamics\n")

    asyncio.run(main_curriculum_experiment(iterations))


if __name__ == "__main__":
    main()
