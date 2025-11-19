"""Domain curriculum manager for KV-1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CurriculumStage:
    domain: str
    level: str
    resources: List[str]
    eval_threshold: float
    next_stage: Optional[str] = None


class CurriculumManager:
    def __init__(self):
        self.stages: Dict[str, List[CurriculumStage]] = {
            "algebra": [
                CurriculumStage("algebra", "basics", ["Khan: Solving equations", "Wiki: Linear equations"], 0.9, "intermediate"),
                CurriculumStage("algebra", "intermediate", ["MIT OCW Algebra"], 0.95, "advanced"),
                CurriculumStage("algebra", "advanced", ["Art of Problem Solving"] , 0.99, None),
            ],
            "calculus": [
                CurriculumStage("calculus", "basics", ["Khan: Derivatives", "Wiki: Differential calculus"], 0.85, "intermediate"),
                CurriculumStage("calculus", "intermediate", ["MIT Calculus"], 0.9, "advanced"),
                CurriculumStage("calculus", "advanced", ["Spivak Calculus"], 0.95, None),
            ],
            "thermodynamics": [
                CurriculumStage("thermodynamics", "basics", ["Khan: Thermodynamics", "Wiki: First law"], 0.8, "intermediate"),
                CurriculumStage("thermodynamics", "intermediate", ["MIT Thermodynamics"], 0.85, "advanced"),
                CurriculumStage("thermodynamics", "advanced", ["Callen Thermodynamics"], 0.9, None),
            ],
        }
        self.progress: Dict[str, str] = {domain: "basics" for domain in self.stages}

    def current_stage(self, domain: str) -> CurriculumStage:
        level = self.progress.get(domain, "basics")
        return next(stage for stage in self.stages[domain] if stage.level == level)

    def update_progress(self, domain: str, score: float) -> Optional[CurriculumStage]:
        stage = self.current_stage(domain)
        if score >= stage.eval_threshold and stage.next_stage:
            self.progress[domain] = stage.next_stage
            return self.current_stage(domain)
        return None

    def get_resources(self, domain: str) -> List[str]:
        return self.current_stage(domain).resources
