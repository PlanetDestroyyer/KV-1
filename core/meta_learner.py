"""
Meta-Learning System - Learning HOW to Learn

Tracks learning strategies, analyzes what works, and improves over time.
This is what makes KV-1 get BETTER at learning with each concept.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class LearningAttempt:
    """Record of a single learning attempt."""
    concept: str
    domain: str
    attempts: int
    time_seconds: float
    final_confidence: float
    success: bool
    prerequisites_learned: int
    web_searches: int
    rehearsal_rounds: int
    timestamp: str


class MetaLearner:
    """
    Learns HOW to learn better over time.

    Tracks:
    - Which learning strategies work best
    - Which concepts are hardest
    - Optimal prerequisite paths
    - Learning speed improvements
    """

    def __init__(self, storage_path: str = "./meta_learning.json"):
        self.storage_path = storage_path
        self.learning_history: List[LearningAttempt] = []
        self.concept_difficulty: Dict[str, float] = {}  # How hard is each concept?
        self.domain_performance: Dict[str, List[float]] = defaultdict(list)
        self.strategy_success_rate: Dict[str, float] = {}
        self.optimal_paths: Dict[str, List[str]] = {}  # Best prerequisite order

        self.load()

    def record_attempt(self, attempt: LearningAttempt):
        """Record a learning attempt and update meta-knowledge."""
        self.learning_history.append(attempt)

        # Update difficulty rating
        difficulty = self._calculate_difficulty(attempt)
        self.concept_difficulty[attempt.concept] = difficulty

        # Track domain performance
        self.domain_performance[attempt.domain].append(
            attempt.final_confidence if attempt.success else 0.0
        )

        # Save after each attempt
        self.save()

    def _calculate_difficulty(self, attempt: LearningAttempt) -> float:
        """
        Calculate how difficult a concept was to learn.

        Returns: 0.0 (easy) to 1.0 (very hard)
        """
        # Factors that indicate difficulty:
        # - More attempts needed
        # - More prerequisites
        # - More rehearsal rounds
        # - Lower final confidence
        # - Longer time

        difficulty = 0.0

        # Attempts (more = harder)
        difficulty += min(attempt.attempts / 10.0, 0.3)  # Max 0.3

        # Prerequisites (deep tree = harder)
        difficulty += min(attempt.prerequisites_learned / 20.0, 0.3)  # Max 0.3

        # Rehearsal rounds (more practice needed = harder)
        difficulty += min(attempt.rehearsal_rounds / 5.0, 0.2)  # Max 0.2

        # Confidence (lower = harder)
        difficulty += (1.0 - attempt.final_confidence) * 0.2  # Max 0.2

        return min(difficulty, 1.0)

    def get_learning_strategy(self, concept: str, domain: str) -> Dict[str, any]:
        """
        Recommend learning strategy based on past experience.

        Returns: Dict with recommendations like:
            - max_rehearsals: How many rounds to do
            - confidence_threshold: When to stop practicing
            - check_prerequisites_deeply: Whether to go deep on prereqs
        """
        strategy = {
            "max_rehearsals": 4,
            "confidence_threshold": 0.70,
            "check_prerequisites_deeply": True,
            "parallel_learning": True
        }

        # If we've seen similar concepts, adjust strategy
        difficulty = self.concept_difficulty.get(concept, 0.5)

        if difficulty > 0.7:
            # Hard concept - be more thorough
            strategy["max_rehearsals"] = 6
            strategy["confidence_threshold"] = 0.75
            strategy["check_prerequisites_deeply"] = True
        elif difficulty < 0.3:
            # Easy concept - can be faster
            strategy["max_rehearsals"] = 2
            strategy["confidence_threshold"] = 0.65
            strategy["check_prerequisites_deeply"] = False

        # Check domain performance
        if domain in self.domain_performance:
            avg_performance = sum(self.domain_performance[domain]) / len(self.domain_performance[domain])
            if avg_performance > 0.8:
                # We're good at this domain - can be faster
                strategy["confidence_threshold"] -= 0.05

        return strategy

    def analyze_improvement(self) -> Dict[str, float]:
        """
        Analyze how much we've improved at learning over time.

        Returns: Dict with metrics like speed improvement, confidence improvement
        """
        if len(self.learning_history) < 10:
            return {"not_enough_data": True}

        # Split history into early and recent
        split_point = len(self.learning_history) // 2
        early = self.learning_history[:split_point]
        recent = self.learning_history[split_point:]

        # Compare metrics
        early_avg_attempts = sum(a.attempts for a in early) / len(early)
        recent_avg_attempts = sum(a.attempts for a in recent) / len(recent)

        early_avg_confidence = sum(a.final_confidence for a in early if a.success) / max(sum(1 for a in early if a.success), 1)
        recent_avg_confidence = sum(a.final_confidence for a in recent if a.success) / max(sum(1 for a in recent if a.success), 1)

        early_avg_time = sum(a.time_seconds for a in early) / len(early)
        recent_avg_time = sum(a.time_seconds for a in recent) / len(recent)

        return {
            "attempts_improvement": (early_avg_attempts - recent_avg_attempts) / early_avg_attempts if early_avg_attempts > 0 else 0,
            "confidence_improvement": recent_avg_confidence - early_avg_confidence,
            "speed_improvement": (early_avg_time - recent_avg_time) / early_avg_time if early_avg_time > 0 else 0,
            "total_concepts_learned": len(self.learning_history),
            "current_avg_confidence": recent_avg_confidence
        }

    def identify_weak_areas(self) -> List[Tuple[str, float]]:
        """
        Identify knowledge areas that need improvement.

        Returns: List of (domain, avg_performance) sorted by worst performance
        """
        domain_scores = []
        for domain, scores in self.domain_performance.items():
            if len(scores) >= 3:  # Need at least 3 attempts to judge
                avg = sum(scores) / len(scores)
                domain_scores.append((domain, avg))

        # Sort by worst performance first
        domain_scores.sort(key=lambda x: x[1])
        return domain_scores

    def save(self):
        """Save meta-learning data to disk."""
        data = {
            "learning_history": [asdict(a) for a in self.learning_history],
            "concept_difficulty": self.concept_difficulty,
            "domain_performance": dict(self.domain_performance),
            "strategy_success_rate": self.strategy_success_rate,
            "optimal_paths": self.optimal_paths
        }

        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load meta-learning data from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.learning_history = [
                LearningAttempt(**a) for a in data.get("learning_history", [])
            ]
            self.concept_difficulty = data.get("concept_difficulty", {})
            self.domain_performance = defaultdict(list, data.get("domain_performance", {}))
            self.strategy_success_rate = data.get("strategy_success_rate", {})
            self.optimal_paths = data.get("optimal_paths", {})

            print(f"[Meta] Loaded {len(self.learning_history)} past learning experiences")
        except Exception as e:
            print(f"[!] Failed to load meta-learning data: {e}")
