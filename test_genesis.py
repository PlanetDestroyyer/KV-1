"""
Genesis mode regression test.

Simulates a 7-day bootstrap cycle and ensures confidence thresholds are
met for Algebra, Calculus, and Thermodynamics benchmarks.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from core.genesis_mode import GenesisController


class DummyMemory:
    def __init__(self):
        self.entries = {}

    def learn(self, query, answer):
        self.entries[query] = answer

    def recall(self, query):
        return self.entries.get(query)


class DummyLLM:
    def __init__(self):
        self.counter = 0

    def generate(self, system_prompt: str, user_input: str, execute: bool = True):
        self.counter += 1
        text = f"{user_input} :: " + ("x" * (self.counter * 600))
        return {
            "provider": "dummy",
            "request": {"body": {"contents": [{"parts": [{"text": system_prompt}, {"text": user_input}]}]}},
            "response": {"candidates": [{"content": {"parts": [{"text": text}]}}]},
            "text": text,
            "executed": True,
        }


class DummyOrchestrator:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.memory = DummyMemory()
        self._llm = DummyLLM()
        self.logs = []

    def generate_with_llm(self, user_input: str, system_prompt: str = "Genesis test", execute: bool = False):
        return self._llm.generate(system_prompt, user_input, execute=execute)

    def log_event(self, event_type: str, payload: Optional[dict] = None):
        self.logs.append((event_type, payload))


def test_genesis_bootstrap():
    with tempfile.TemporaryDirectory() as tmp:
        orch = DummyOrchestrator(tmp)
        genesis = GenesisController(orch, enabled=True, log_path=os.path.join(tmp, "log.json"))

        print("Genesis bootstrap test started.")
        print("Agent baseline knowledge: digits 0-9 and letters a-z only.")
        print("Simulated Ollama/web research loop engaged...")

        for _ in range(7):
            genesis.daily_probe()
            focus = genesis.next_focus_query()
            if focus:
                print(f"Simulated research query: {focus}")

        for domain, threshold in GenesisController.REQUIRED_THRESHOLDS.items():
            assert genesis.progress[domain] >= threshold, f"{domain} below threshold"

        with open(genesis.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "progress" in data


if __name__ == "__main__":
    test_genesis_bootstrap()
    print("Genesis test passed.")
