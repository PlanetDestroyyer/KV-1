from __future__ import annotations

from types import SimpleNamespace

from core.evaluation import EvaluationHarness, EvaluationTask


class DummyLLMOrchestrator:
    def __init__(self):
        self.traumas = []
        self.logs = []
        self.curiosity = []
        self.three_stage = SimpleNamespace(add_curiosity_item=self._add_curiosity)
        self.curriculum = SimpleNamespace(update_progress=lambda d, s: None)

    def generate_with_llm(self, prompt: str, **kwargs):
        if "Solve for x" in prompt:
            text = "x = 4"
        elif "Differentiate" in prompt:
            text = "3x^2 - 10x + 2"
        else:
            text = "Heat example"
        return {"text": text}

    def add_trauma(self, trigger: str, pain: float, context: str):
        self.traumas.append((trigger, pain, context))

    def log_event(self, event_type: str, payload):
        self.logs.append((event_type, payload))

    def _add_curiosity(self, token: str, query: str):
        self.curiosity.append((token, query))


def test_evaluation_harness():
    orch = DummyLLMOrchestrator()
    tasks = [
        EvaluationTask("algebra", "Solve for x: 3x + 5 = 17", ["4"], "algebra basics"),
        EvaluationTask("short", "Differentiate", ["3x^2"], "calc basics"),
        EvaluationTask("thermo", "Describe thermodynamics", ["energy"], "thermo law"),
    ]
    harness = EvaluationHarness(orch, tasks)
    scores = harness.run_cycle()
    assert scores["algebra"] == 1.0
    assert scores["short"] == 1.0
    assert scores["thermo"] < 1.0
    assert orch.traumas, "Thermo failure should add trauma"
    assert orch.curiosity, "Thermo failure should enqueue curiosity"


if __name__ == "__main__":
    test_evaluation_harness()
    print("evaluation test passed")
