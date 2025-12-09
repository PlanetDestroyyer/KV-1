"""
Self-Prompt Optimization Module

THE MOST ADVANCED AGI FEATURE:
The system analyzes its own prompts and MODIFIES them to learn better!

Key capabilities:
- Tracks which prompts lead to successful learning
- A/B tests different prompt variations
- Evolves prompts based on performance
- Self-improves over time

This is TRUE meta-cognition - the system improving its own thinking process!
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class PromptPerformance:
    """Tracks performance of a specific prompt"""
    prompt_id: str
    prompt_text: str
    success_count: int = 0
    failure_count: int = 0
    avg_confidence: float = 0.0
    avg_learning_time: float = 0.0  # seconds
    last_used: str = ""
    concepts_learned: int = 0


class SelfPromptOptimizer:
    """
    The system that makes the AGI SELF-IMPROVING.

    "An intelligence that can improve itself is an intelligence
    that can achieve superintelligence." - I.J. Good
    """

    def __init__(self, storage_path: str = "./prompt_optimization.json"):
        self.storage_path = storage_path
        self.prompts: Dict[str, PromptPerformance] = {}
        self.current_variants: Dict[str, List[str]] = {}  # prompt_type → variants
        self.load()

    def register_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        prompt_type: str = "general"
    ):
        """Register a prompt for tracking"""
        if prompt_id not in self.prompts:
            self.prompts[prompt_id] = PromptPerformance(
                prompt_id=prompt_id,
                prompt_text=prompt_text
            )
            print(f"[PromptOpt] Registered prompt: {prompt_id}")

    def record_success(
        self,
        prompt_id: str,
        confidence: float,
        learning_time: float,
        concepts_learned: int = 1
    ):
        """Record a successful use of a prompt"""
        if prompt_id not in self.prompts:
            return

        p = self.prompts[prompt_id]
        p.success_count += 1
        p.concepts_learned += concepts_learned
        p.last_used = datetime.now().isoformat()

        # Update rolling averages
        total_uses = p.success_count + p.failure_count
        p.avg_confidence = (
            (p.avg_confidence * (total_uses - 1) + confidence) / total_uses
        )
        p.avg_learning_time = (
            (p.avg_learning_time * (total_uses - 1) + learning_time) / total_uses
        )

        print(f"[PromptOpt] ✅ '{prompt_id}' success (conf: {confidence:.2f}, time: {learning_time:.1f}s)")

    def record_failure(
        self,
        prompt_id: str,
        reason: str = "unknown"
    ):
        """Record a failed use of a prompt"""
        if prompt_id not in self.prompts:
            return

        p = self.prompts[prompt_id]
        p.failure_count += 1
        p.last_used = datetime.now().isoformat()

        print(f"[PromptOpt] ❌ '{prompt_id}' failed: {reason}")

    def get_best_prompt(
        self,
        prompt_type: str = "general",
        min_uses: int = 5
    ) -> Optional[str]:
        """Get the best performing prompt for a given type"""
        candidates = []

        for prompt_id, perf in self.prompts.items():
            total_uses = perf.success_count + perf.failure_count
            if total_uses < min_uses:
                continue

            success_rate = perf.success_count / total_uses if total_uses > 0 else 0
            score = success_rate * perf.avg_confidence

            candidates.append((prompt_id, score, perf))

        if not candidates:
            return None

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, score, perf = candidates[0]

        print(f"[PromptOpt] Best prompt: '{best_id}' (score: {score:.2f}, success: {perf.success_count}/{perf.success_count + perf.failure_count})")

        return perf.prompt_text

    def evolve_prompt(
        self,
        current_prompt: str,
        performance_issue: str,
        llm
    ) -> str:
        """
        Use LLM to EVOLVE a prompt based on performance issues.

        THIS IS WHERE THE MAGIC HAPPENS!
        The system modifies its own prompts to improve itself!
        """
        print(f"[PromptOpt] 🧬 EVOLVING PROMPT...")
        print(f"[PromptOpt] Issue: {performance_issue}")

        evolution_prompt = f"""You are an AGI improving its own learning prompts.

CURRENT PROMPT:
{current_prompt}

PERFORMANCE ISSUE:
{performance_issue}

TASK: Modify the prompt to address this issue and improve learning performance.

Guidelines for improvement:
1. Make instructions clearer and more specific
2. Add examples if needed
3. Break complex instructions into steps
4. Emphasize critical requirements
5. Remove confusing or contradictory parts

Respond with:
ANALYSIS: [what's wrong with current prompt]
IMPROVED PROMPT: [the evolved prompt]
EXPECTED IMPROVEMENT: [why this should work better]
"""

        response = llm.generate(
            system_prompt="You are an AGI improving its own prompts through self-reflection. Be critical and make significant improvements.",
            user_input=evolution_prompt
        )

        text = response.get("text", "")

        # Extract improved prompt
        if "IMPROVED PROMPT:" in text:
            parts = text.split("IMPROVED PROMPT:")
            if len(parts) > 1:
                improved = parts[1].strip()
                # Take until next section marker
                for marker in ["EXPECTED IMPROVEMENT:", "ANALYSIS:"]:
                    if marker in improved:
                        improved = improved.split(marker)[0].strip()

                print(f"[PromptOpt] ✨ Created evolved prompt!")
                return improved

        print(f"[PromptOpt] ⚠️ Could not extract improved prompt, using original")
        return current_prompt

    def suggest_ab_test(
        self,
        base_prompt: str,
        llm
    ) -> List[str]:
        """
        Generate A/B test variants of a prompt.

        Returns 3-5 variations to test in parallel.
        """
        print(f"[PromptOpt] Creating A/B test variants...")

        prompt = f"""Create 3 variations of this prompt for A/B testing:

BASE PROMPT:
{base_prompt}

Create variations that test different approaches:
1. More concise vs more detailed
2. Different instruction order
3. Different emphasis on requirements

Respond with:
VARIANT A: [first variation]
VARIANT B: [second variation]
VARIANT C: [third variation]
"""

        response = llm.generate(
            system_prompt="You create prompt variations for A/B testing.",
            user_input=prompt
        )

        text = response.get("text", "")
        variants = []

        for marker in ["VARIANT A:", "VARIANT B:", "VARIANT C:"]:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    variant = parts[1].strip()
                    # Extract until next marker
                    for next_marker in ["VARIANT A:", "VARIANT B:", "VARIANT C:"]:
                        if next_marker in variant:
                            variant = variant.split(next_marker)[0].strip()
                    if variant and len(variant) > 50:
                        variants.append(variant)

        print(f"[PromptOpt] Generated {len(variants)} test variants")
        return variants

    def get_optimization_report(self) -> str:
        """Generate report on prompt performance"""
        if not self.prompts:
            return "[PromptOpt] No data yet"

        report = [
            "\n" + "="*60,
            "PROMPT OPTIMIZATION REPORT",
            "="*60,
            ""
        ]

        # Sort by success count
        sorted_prompts = sorted(
            self.prompts.items(),
            key=lambda x: x[1].success_count,
            reverse=True
        )

        for prompt_id, perf in sorted_prompts[:10]:
            total = perf.success_count + perf.failure_count
            if total == 0:
                continue

            success_rate = (perf.success_count / total) * 100

            report.append(f"Prompt: {prompt_id}")
            report.append(f"  Success Rate: {success_rate:.1f}% ({perf.success_count}/{total})")
            report.append(f"  Avg Confidence: {perf.avg_confidence:.2f}")
            report.append(f"  Avg Time: {perf.avg_learning_time:.1f}s")
            report.append(f"  Concepts Learned: {perf.concepts_learned}")
            report.append("")

        report.append("="*60)
        return "\n".join(report)

    def save(self):
        """Save prompt performance data"""
        data = {}
        for prompt_id, perf in self.prompts.items():
            data[prompt_id] = {
                "prompt_text": perf.prompt_text,
                "success_count": perf.success_count,
                "failure_count": perf.failure_count,
                "avg_confidence": perf.avg_confidence,
                "avg_learning_time": perf.avg_learning_time,
                "last_used": perf.last_used,
                "concepts_learned": perf.concepts_learned
            }

        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[PromptOpt] Saved optimization data ({len(data)} prompts)")

    def load(self):
        """Load prompt performance data"""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for prompt_id, perf_data in data.items():
                self.prompts[prompt_id] = PromptPerformance(
                    prompt_id=prompt_id,
                    prompt_text=perf_data["prompt_text"],
                    success_count=perf_data.get("success_count", 0),
                    failure_count=perf_data.get("failure_count", 0),
                    avg_confidence=perf_data.get("avg_confidence", 0.0),
                    avg_learning_time=perf_data.get("avg_learning_time", 0.0),
                    last_used=perf_data.get("last_used", ""),
                    concepts_learned=perf_data.get("concepts_learned", 0)
                )

            print(f"[PromptOpt] Loaded {len(self.prompts)} prompt histories")

        except Exception as e:
            print(f"[PromptOpt] Failed to load: {e}")
