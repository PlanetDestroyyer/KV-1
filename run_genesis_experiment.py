#!/usr/bin/env python3
"""
Genesis Experiment Runner

Pure emergence test:
- Start with 0-9, a-z (36 symbols)
- No OS features, no human intervention
- Let it discover: words → sentences → algebra → calculus → physics

Run with:
    python run_genesis_experiment.py

Options:
    --iterations N    Number of learning cycles (default: 480 = ~24 hours)
    --interval N      Seconds between cycles (default: 180 = 3 min)
    --quick           Quick test mode (10 iterations, 30s interval)
    --data-dir PATH   Data directory (default: ./genesis_data)
"""

import asyncio
import argparse
from genesis_orchestrator import GenesisOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run Genesis emergence experiment")
    parser.add_argument(
        "--iterations",
        type=int,
        default=480,
        help="Number of learning cycles (default: 480 = ~24 hours at 3min/cycle)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=180,
        help="Seconds between cycles (default: 180 = 3 minutes)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (10 iterations, 30s interval)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./genesis_data",
        help="Data directory (default: ./genesis_data)"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.iterations = 10
        args.interval = 30
        print("🚀 QUICK TEST MODE (10 iterations, 30s interval)")

    # Initialize orchestrator
    print("Initializing Genesis Orchestrator...")
    orchestrator = GenesisOrchestrator(data_dir=args.data_dir)

    # Run experiment
    asyncio.run(orchestrator.run_genesis_experiment(
        iterations=args.iterations,
        interval_seconds=args.interval
    ))


if __name__ == "__main__":
    main()
