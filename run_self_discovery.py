#!/usr/bin/env python3
"""
Self-Discovery Learning Runner

Run goal-driven autonomous learning experiments.

Example usage:
  python run_self_discovery.py "solve 2x + 5 = 15"
  python run_self_discovery.py "count from 1 to 10"
  python run_self_discovery.py "what is the area of a circle with radius 5"

With custom LTM file:
  python run_self_discovery.py "solve 3x - 7 = 20" --ltm my_memory.json

Reset LTM and start fresh:
  python run_self_discovery.py "solve x + 3 = 7" --reset
"""

import asyncio
import argparse
import os
from self_discovery_orchestrator import main_self_discovery
from core.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run self-discovery learning with a goal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example goals:
  Beginner:
    "Count from 1 to 10"
    "What are the letters from A to E"
    "Add 5 and 3"

  Intermediate:
    "Solve 2x + 5 = 15"
    "Calculate 25% of 80"
    "What is 3 squared"

  Advanced:
    "Solve the quadratic equation x^2 - 5x + 6 = 0"
    "Calculate the area of a circle with radius 5"
    "Explain why ice floats on water"
    "What is escape velocity"
        """
    )

    parser.add_argument(
        "goal",
        type=str,
        help="The goal to achieve through self-discovery learning"
    )

    parser.add_argument(
        "--ltm",
        type=str,
        default="./ltm_memory.json",
        help="Path to long-term memory file (default: ./ltm_memory.json)"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset LTM and start with blank memory"
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum attempts to achieve goal (default: None = unlimited, will run until success)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable multi-source validation (slower but more accurate). Default: OFF for speed"
    )

    parser.add_argument(
        "--no-rehearsal",
        action="store_true",
        help="Disable 3-stage learning rehearsal (faster but lower quality). Default: ON for quality"
    )

    parser.add_argument(
        "--target-confidence",
        type=float,
        default=0.70,
        help="Mastery threshold for 3-stage learning (0.0-1.0). Default: 0.70 (70%)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging to file (all output saved to ./logs/)
    log_file = setup_logging(session_name="self_discovery")
    print(f"[+] All output being saved to: {log_file}")

    # Reset LTM if requested
    if args.reset and os.path.exists(args.ltm):
        print(f"[!] Resetting LTM: {args.ltm}")
        os.remove(args.ltm)

    print("\n" + "="*60)
    print("SELF-DISCOVERY LEARNING SYSTEM")
    print("="*60)
    print(f"Goal: {args.goal}")
    print(f"LTM file: {args.ltm}")
    if args.max_attempts:
        print(f"Max attempts: {args.max_attempts}")
    else:
        print("Max attempts: UNLIMITED (will run until success)")
    print(f"3-Stage Learning: {'OFF (fast mode)' if args.no_rehearsal else f'ON (target confidence: {args.target_confidence:.2f})'}")
    print(f"Validation: {'ON (multi-source)' if args.validate else 'OFF (fast mode)'}")
    print(f"STM Capacity: 50 slots (GPU-optimized, was 7)")
    print(f"Parallel Processing: ⚡ Up to 10 concepts simultaneously ⚡")
    print("="*60)

    # Run self-discovery learning
    success = asyncio.run(main_self_discovery(
        goal=args.goal,
        ltm_path=args.ltm,
        max_attempts=args.max_attempts,
        enable_validation=args.validate,
        enable_rehearsal=not args.no_rehearsal,  # Inverted logic
        target_confidence=args.target_confidence
    ))

    if success:
        print("\n[OK] Next time you run with a similar goal, I'll use this knowledge!")
        print(f"[i] Try: python run_self_discovery.py 'solve 3x - 7 = 20' --ltm {args.ltm}")
    else:
        print("\n[X] Goal not achieved. The system may need more capabilities or better web content.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
