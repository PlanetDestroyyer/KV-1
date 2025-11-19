"""Domain evaluation harness for KV-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvaluationTask:
    domain: str
    prompt: str
    keywords: List[str]
    failure_query: str


DEFAULT_TASKS: List[EvaluationTask] = [
    EvaluationTask(
        domain="algebra",
        prompt="Solve for x: 3x + 5 = 17. Provide steps.",
        keywords=["4"],
        failure_query="beginner algebra balancing equations",
    ),
    EvaluationTask(
        domain="calculus",
        prompt="Differentiate f(x)=x^3 - 5x^2 + 2x.",
        keywords=["3x^2 - 10x + 2"],
        failure_query="basic calculus derivatives",
    ),
    EvaluationTask(
        domain="thermodynamics",
        prompt="State the first law of thermodynamics and give an example.",
        keywords=["energy", "conservation"],
        failure_query="thermodynamics first law basics",
    ),
]


class EvaluationHarness:
    def __init__(self, orchestrator, tasks: List[EvaluationTask] = None):
        self.orchestrator = orchestrator
        self.tasks = tasks or DEFAULT_TASKS
        self.last_scores: Dict[str, float] = {task.domain: 0.0 for task in self.tasks}

    def run_cycle(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for task in self.tasks:
            result = self._run_task(task)
            scores[task.domain] = result
            self.last_scores[task.domain] = result
        return scores

    def _run_task(self, task: EvaluationTask) -> float:
        payload = self.orchestrator.generate_with_llm(task.prompt)
        answer = payload.get("text") or ""
        score = self._score_answer(answer, task.keywords)
        if score < 1.0:
            self._handle_failure(task, answer, score)
        else:
            self.orchestrator.log_event("eval_success", {"domain": task.domain})
            # Update curriculum if available (Genesis mode doesn't have curriculum)
            if hasattr(self.orchestrator, 'curriculum') and self.orchestrator.curriculum:
                self.orchestrator.curriculum.update_progress(task.domain, score)
        return score

    def _score_answer(self, answer: str, keywords: List[str]) -> float:
        hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
        return hits / max(1, len(keywords))

    def _handle_failure(self, task: EvaluationTask, answer: str, score: float):
        context = f"Eval fail ({task.domain}) score={score:.2f}"
        # Add to curiosity queue for research
        self.orchestrator.three_stage.add_curiosity_item(task.domain, task.failure_query)
        self.orchestrator.log_event(
            "eval_failure",
            {"domain": task.domain, "score": score, "answer": answer[:200]},
        )
