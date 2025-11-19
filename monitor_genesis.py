#!/usr/bin/env python3
"""
Genesis Experiment Monitor

Tail the emergence logs and show real-time progress.

Usage:
    python monitor_genesis.py [--data-dir ./genesis_data]
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime


def load_latest_snapshot(data_dir: str):
    """Load the latest daily snapshot."""
    log_path = Path(data_dir) / "logs" / "daily_progress.jsonl"
    if not log_path.exists():
        return None

    with open(log_path, "r") as f:
        lines = f.readlines()
        if lines:
            return json.loads(lines[-1])
    return None


def load_recent_events(data_dir: str, n: int = 10):
    """Load the N most recent events."""
    log_path = Path(data_dir) / "logs" / "emergence.jsonl"
    if not log_path.exists():
        return []

    with open(log_path, "r") as f:
        lines = f.readlines()
        recent = lines[-n:] if len(lines) >= n else lines
        return [json.loads(line) for line in recent]


def print_dashboard(data_dir: str):
    """Print a simple dashboard."""
    snapshot = load_latest_snapshot(data_dir)
    events = load_recent_events(data_dir, 5)

    print("\033[2J\033[H")  # Clear screen
    print("=" * 70)
    print("🧬 GENESIS EXPERIMENT MONITOR")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if snapshot:
        print(f"⏱️  Day: {snapshot.get('day', 0)}")
        print(f"⏱️  Hours Elapsed: {snapshot.get('hours_elapsed', 0):.2f}")
        print(f"🧠 Phase: {snapshot.get('genesis_phase', 'unknown').upper()}")
        print()
        print(f"📚 Long-term Memory: {snapshot.get('ltm_size', 0)} entries")
        print(f"🔄 Working Memory: {snapshot.get('stm_size', 0)}/7")
        print(f"⚡ Surprise Episodes: {snapshot.get('surprise_episodes', 0)}")
        print()
        print(f"📊 Total Surprises: {snapshot.get('total_surprises', 0)}")
        print(f"✅ Total Transfers: {snapshot.get('total_transfers', 0)}")
        print(f"🌐 Web Requests Today: {snapshot.get('web_requests_today', 0)}")
        print()

        # Domain progress
        progress = snapshot.get('genesis_progress', {})
        print("🎯 Domain Mastery:")
        for domain, score in progress.items():
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {domain:15s} [{bar}] {score*100:.1f}%")
        print()

        # Eval scores
        eval_scores = snapshot.get('eval_scores', {})
        if eval_scores:
            print("📝 Latest Evaluation:")
            for domain, score in eval_scores.items():
                print(f"   {domain:15s} {score*100:.1f}%")
            print()

    else:
        print("⚠️  No snapshot data yet. Experiment may not have started.")
        print()

    # Recent events
    if events:
        print("📜 Recent Events:")
        for event in events[-5:]:
            ts = event.get('timestamp', '')[:19]
            etype = event.get('type', 'unknown')
            elapsed = event.get('elapsed_hours', 0)
            print(f"   [{ts}] ({elapsed:.2f}h) {etype}")
        print()

    print("=" * 70)
    print("Press Ctrl+C to exit")


def watch_mode(data_dir: str, interval: int = 10):
    """Watch mode - refresh dashboard every N seconds."""
    try:
        while True:
            print_dashboard(data_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor Genesis experiment")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./genesis_data",
        help="Data directory (default: ./genesis_data)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit (no watch mode)"
    )

    args = parser.parse_args()

    if args.once:
        print_dashboard(args.data_dir)
    else:
        watch_mode(args.data_dir, args.interval)


if __name__ == "__main__":
    main()
