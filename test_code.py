"""Smoke test to run KV-1 autonomy loop for a short period."""

from __future__ import annotations

import time

from core import KV1Orchestrator


def main():
    kv1 = KV1Orchestrator(
        data_dir="./kv1_test_run",
        use_hsokv=False,  # assume HSOKV not installed for smoke test
        genesis_mode=True,
        llm_api_key=None,
    )

    print("Starting autonomy scheduler for 60 seconds...")
    kv1.start_autonomy()
    time.sleep(60)
    kv1.stop_autonomy()

    scores = kv1.run_evaluation_cycle()
    print("Final evaluation scores:", scores)
    print("Check kv1_test_run/logs/events.jsonl for discovery logs.")


if __name__ == "__main__":
    main()
