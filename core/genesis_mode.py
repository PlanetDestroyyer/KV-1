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
        self.genesis_start = datetime.now() if enabled else None
        self.innate_uses = 0  # Track alphanumeric baseline usage
        self.total_surprises = 0  # Track surprise episodes created
        self.total_transfers = 0  # Track LTM transfers
        if self.enabled:
            self._initialize_alphanumeric_core()

    def _initialize_alphanumeric_core(self):
        """Reset LTM to the 0-9/a-z baseline if possible."""
        if not self.orchestrator.memory:
            return

        # Clear LTM before bootstrapping (requirement #13)
        try:
            # For DualMemorySystem, clear both STM and LTM
            if hasattr(self.orchestrator.memory, 'stm'):
                self.orchestrator.memory.stm.clear()
            if hasattr(self.orchestrator.memory, 'ltm'):
                # Clear LTM internal storage
                ltm = self.orchestrator.memory.ltm
                if hasattr(ltm, 'memory'):
                    import torch
                    device = ltm.memory.keys.device if len(ltm.memory.keys) > 0 else torch.device(ltm.config.device)
                    ltm.memory.keys = torch.empty(0, ltm.memory.embedding_dim, device=device)
                    ltm.memory.values = []
                    ltm.memory.metadata = []
                    ltm.memory.labels = []
            # Legacy support for simple memory systems
            elif hasattr(self.orchestrator.memory, 'clear'):
                self.orchestrator.memory.clear()
        except Exception as e:
            print(f"[Genesis] Warning: Could not clear LTM: {e}")

        # Bootstrap with alphanumerics only
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

        # Update stats from orchestrator
        if hasattr(self.orchestrator, 'three_stage'):
            # Count transfers since last probe (approximation via LTM size delta)
            pass  # Transfers tracked via event logs

        self._write_log()
        return results

    def _probe_prompt(self, domain: str) -> str:
        return (
            f"Genesis mode probe ({domain}). You know only alphanumerics. "
            f"Demonstrate capability by solving a {domain} task."
        )

    def _score_from_payload(self, payload: Dict) -> float:
        # Simple heuristic: longer answers imply more confidence
        text = payload.get("text") or ""
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

    def record_surprise(self):
        """Track a surprise episode creation."""
        self.total_surprises += 1

    def record_transfer(self):
        """Track a transfer to LTM."""
        self.total_transfers += 1

    def record_innate_use(self):
        """Track usage of innate alphanumeric knowledge."""
        self.innate_uses += 1

    def _write_log(self):
        """Write genesis log with all required fields."""
        # Calculate day number
        day = 0
        if self.genesis_start:
            day = (datetime.now() - self.genesis_start).days + 1

        # Determine phase based on progress
        conf_avg = sum(self.progress.values()) / len(self.progress) if self.progress else 0.0
        if conf_avg < 0.3:
            phase = "bootstrap"
        elif conf_avg < 0.7:
            phase = "learning"
        else:
            phase = "mastery"

        # Calculate accuracy (domains meeting threshold)
        domains_met = sum(1 for domain, threshold in self.REQUIRED_THRESHOLDS.items()
                         if self.progress[domain] >= threshold)
        acc = domains_met / len(self.REQUIRED_THRESHOLDS) if self.REQUIRED_THRESHOLDS else 0.0

        # Get stats from orchestrator
        surprises = self.total_surprises
        transfers = self.total_transfers
        if hasattr(self.orchestrator, 'three_stage'):
            surprises = len(self.orchestrator.three_stage.episodes)

        payload = {
            "timestamp": datetime.now().isoformat(),
            "day": day,
            "phase": phase,
            "progress": self.progress,
            "conf_avg": round(conf_avg, 3),
            "acc": round(acc, 3),
            "innate_uses": self.innate_uses,
            "surprises": surprises,
            "transfers": transfers,
            "remaining": self.gaps(),
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
