"""
KV-1 Trauma Memory System

Tracks disappointments and failures with pain levels that heal over time.
Used to prevent repeated mistakes and learn from negative experiences.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import json


@dataclass
class TraumaMemory:
    """A single traumatic memory"""
    trigger: str              # What caused the disappointment
    pain_level: float         # 0.0 - 10.0 (10 = most painful)
    timestamp: datetime       # When it happened
    context: str             # Additional context
    healed: bool = False     # Has this trauma been processed?

    def heal_over_time(self, days_passed: float) -> None:
        """Pain decreases over time (half-life: 7 days)"""
        if self.healed:
            return

        # Exponential decay: pain = initial * (0.5 ^ (days / 7))
        decay_factor = 0.5 ** (days_passed / 7.0)
        self.pain_level *= decay_factor

        # If pain drops below 1.0, mark as healed
        if self.pain_level < 1.0:
            self.healed = True
            self.pain_level = 0.0

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "pain_level": self.pain_level,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "healed": self.healed
        }

    @staticmethod
    def from_dict(data: dict) -> 'TraumaMemory':
        return TraumaMemory(
            trigger=data["trigger"],
            pain_level=data["pain_level"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data["context"],
            healed=data["healed"]
        )


class TraumaSystem:
    """Manages all trauma memories for KV-1"""

    def __init__(self):
        self.traumas: List[TraumaMemory] = []

    def add_trauma(self, trigger: str, pain_level: float, context: str = "") -> None:
        """Record a new trauma"""
        trauma = TraumaMemory(
            trigger=trigger,
            pain_level=min(10.0, max(0.0, pain_level)),  # Clamp 0-10
            timestamp=datetime.now(),
            context=context
        )
        self.traumas.append(trauma)

    def get_active_traumas(self) -> List[TraumaMemory]:
        """Get all unhealed traumas"""
        return [t for t in self.traumas if not t.healed]

    def get_top_traumas(self, n: int = 3) -> List[TraumaMemory]:
        """Get top N most painful active traumas"""
        active = self.get_active_traumas()
        return sorted(active, key=lambda t: t.pain_level, reverse=True)[:n]

    def update_healing(self) -> None:
        """Update all traumas based on time passed"""
        now = datetime.now()
        for trauma in self.traumas:
            if not trauma.healed:
                days_passed = (now - trauma.timestamp).total_seconds() / 86400
                trauma.heal_over_time(days_passed)

    def is_trigger_recent(self, trigger: str, within_hours: int = 24) -> bool:
        """Check if this trigger happened recently (avoid duplicate traumas)"""
        cutoff = datetime.now() - timedelta(hours=within_hours)
        for trauma in self.traumas:
            if trauma.trigger == trigger and trauma.timestamp > cutoff:
                return True
        return False

    def get_trauma_summary(self) -> str:
        """Get human-readable summary for system prompt"""
        active = self.get_top_traumas(3)
        if not active:
            return "No active traumas"

        lines = []
        for trauma in active:
            lines.append(f"- {trauma.trigger} (pain: {trauma.pain_level:.1f}/10)")

        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """Persist traumas to disk"""
        data = [t.to_dict() for t in self.traumas]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load traumas from disk"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.traumas = [TraumaMemory.from_dict(t) for t in data]
            self.update_healing()  # Update healing status after load
        except FileNotFoundError:
            pass  # No traumas yet
