"""
Genesis Mode controller for KV-1.

Forces the system to bootstrap knowledge from alphanumeric symbols and
tracks mastery benchmarks across Algebra, Calculus, and Thermodynamics.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Optional


class GenesisController:
    """Handles alphanumeric genesis benchmarking and logging."""

    REQUIRED_THRESHOLDS = {
        "algebra": 0.90,
        "calculus": 0.85,
        "thermodynamics": 0.80,
    }

    def __init__(
        self,
        orchestrator,
        *,
        enabled: bool = False,
        log_path: Optional[str] = None,
    ):
        self.orchestrator = orchestrator
        self.enabled = enabled
        self.log_path = log_path or os.path.join(orchestrator.data_dir, "genesis_log.json")
        self.progress: Dict[str, float] = {k: 0.0 for k in self.REQUIRED_THRESHOLDS}
        self.last_probe = None
        if self.enabled:
            self._initialize_alphanumeric_core()

    def _initialize_alphanumeric_core(self):
        """Reset LTM to the 0-9/a-z baseline if possible."""
        if not self.orchestrator.memory:
            return
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        for ch in alphabet:
            self.orchestrator.memory.learn(ch, f"Symbol '{ch}' baseline fact.")

    def daily_probe(self):
        """Probe each domain using the configured LLM."""
        if not self.enabled:
            return {}
        results = {}
        for domain in self.REQUIRED_THRESHOLDS:
            prompt = self._probe_prompt(domain)
            payload = self.orchestrator.generate_with_llm(prompt)
            conf = self._score_from_payload(payload)
            self.progress[domain] = max(self.progress[domain], conf)
            results[domain] = conf
        self.last_probe = datetime.now()
        self._write_log()
        return results

    def _probe_prompt(self, domain: str) -> str:
        return (
            f"Genesis mode probe ({domain}). You know only alphanumerics. "
            f"Demonstrate capability by solving a {domain} task."
        )

    def _score_from_payload(self, payload: Dict) -> float:
        # Simple heuristic: longer answers imply more confidence
        text = payload.get("body", {}).get("contents", [{}])[0].get("parts", [{}])[-1].get("text", "")
        return min(0.99, 0.5 + len(text) / 4000.0)

    def gaps(self):
        """Return domains that still need work."""
        return [
            domain
            for domain, threshold in self.REQUIRED_THRESHOLDS.items()
            if self.progress[domain] < threshold
        ]

    def should_trigger_learning(self) -> bool:
        if not self.enabled:
            return False
        if not self.last_probe:
            return True
        return datetime.now() - self.last_probe > timedelta(hours=24)

    def next_focus_query(self) -> Optional[str]:
        remaining = self.gaps()
        if not remaining:
            return None
        domain = random.choice(remaining)
        return f"beginner {domain} fundamentals"

    def _write_log(self):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "progress": self.progress,
            "remaining": self.gaps(),
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
