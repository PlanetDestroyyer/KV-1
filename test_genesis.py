"""
Genesis mode regression test.

Simulates a 7-day bootstrap cycle and ensures confidence thresholds are
met for Algebra, Calculus, and Thermodynamics benchmarks.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

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

    def generate(self, system_prompt: str, user_input: str):
        self.counter += 1
        text = f"{user_input} :: " + ("x" * (self.counter * 600))
        return {
            "body": {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt},
                            {"text": text},
                        ]
                    }
                ]
            }
        }


class DummyOrchestrator:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.memory = DummyMemory()
        self._llm = DummyLLM()

    def generate_with_llm(self, user_input: str, system_prompt: str = "Genesis test"):
        return self._llm.generate(system_prompt, user_input)


def test_genesis_bootstrap():
    with tempfile.TemporaryDirectory() as tmp:
        orch = DummyOrchestrator(tmp)
        genesis = GenesisController(orch, enabled=True, log_path=os.path.join(tmp, "log.json"))

        for _ in range(7):
            genesis.daily_probe()

        for domain, threshold in GenesisController.REQUIRED_THRESHOLDS.items():
            assert genesis.progress[domain] >= threshold, f"{domain} below threshold"

        with open(genesis.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "progress" in data


if __name__ == "__main__":
    test_genesis_bootstrap()
    print("Genesis test passed.")
