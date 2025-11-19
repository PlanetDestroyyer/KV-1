"""Background autonomy scheduler for KV-1."""

from __future__ import annotations

import threading
import time
from typing import Callable, List


class ScheduledJob:
    def __init__(self, name: str, interval: float, func: Callable[[], None]):
        self.name = name
        self.interval = interval
        self.func = func
        self.last_run = 0.0


class AutonomyScheduler:
    def __init__(
        self,
        orchestrator,
        *,
        self_interval: int = 300,
        curiosity_interval: int = 900,
        nightly_interval: int = 3600,
        genesis_interval: int = 3600,
        evaluation_interval: int = 7200,
    ):
        self.orchestrator = orchestrator
        self.jobs: List[ScheduledJob] = [
            ScheduledJob("self_learning", self_interval, self.orchestrator.self_learning_tick),
            ScheduledJob("curiosity", curiosity_interval, self._run_curiosity),
            ScheduledJob("nightly_reflection", nightly_interval, self._run_nightly),
            ScheduledJob("genesis_probe", genesis_interval, self._run_genesis),
            ScheduledJob("evaluation_cycle", evaluation_interval, self._run_evaluation),
        ]
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            for job in self.jobs:
                if now - job.last_run >= job.interval:
                    try:
                        job.func()
                        self.orchestrator.log_event("scheduler_job", {"job": job.name})
                    except Exception as exc:
                        self.orchestrator.log_event(
                            "scheduler_error",
                            {"job": job.name, "error": str(exc)},
                        )
                    job.last_run = now
            time.sleep(1)

    def _run_curiosity(self):
        item = self.orchestrator.next_curiosity_item()
        if item:
            self.orchestrator.log_event("curiosity_research", item)
            self.orchestrator.research(item["query"], mode="scrape")

    def _run_nightly(self):
        summary = self.orchestrator.nightly_reflection()
        self.orchestrator.log_event("nightly_summary", {"summary": summary})

    def _run_genesis(self):
        if self.orchestrator.genesis.should_trigger_learning():
            result = self.orchestrator.genesis.daily_probe()
            self.orchestrator.log_event("genesis_probe", result)

    def _run_evaluation(self):
        scores = self.orchestrator.run_evaluation_cycle()
        self.orchestrator.log_event("evaluation_scores", scores)
