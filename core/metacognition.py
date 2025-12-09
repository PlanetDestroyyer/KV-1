"""
Metacognitive Layer - Think About Thinking

Enables self-reflection, error analysis, and detection of when
the system is going off track.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FailureAnalysis:
    """Analysis of why learning failed."""
    goal: str
    attempts: int
    missing_concepts: List[str]
    root_cause: str
    suggested_fix: str
    confidence: float


class MetacognitiveLayer:
    """
    Self-awareness and reflection on the learning process.

    Capabilities:
    - Analyze why learning failed
    - Detect when going off track
    - Assess true confidence
    - Identify knowledge gaps proactively
    """

    def __init__(self, llm_bridge):
        self.llm = llm_bridge
        self.reflection_history: List[Dict] = []

    async def analyze_failure(
        self,
        goal: str,
        attempts: int,
        missing_concepts: List[str],
        learned_concepts: List[str]
    ) -> FailureAnalysis:
        """
        Deep analysis of why we failed to achieve a goal.

        This is metacognition in action - thinking about why we failed.
        """
        print("\n" + "="*60)
        print("🧠 METACOGNITIVE ANALYSIS: Why did we fail?")
        print("="*60)

        analysis_prompt = f"""You attempted to achieve this goal but failed:

Goal: {goal}
Attempts made: {attempts}
Concepts we learned: {', '.join(learned_concepts[:10])}
Still missing: {', '.join(missing_concepts)}

Analyze WHY you failed. Consider:
1. Are the missing concepts ACTUALLY needed, or are you overthinking?
2. Did you go down rabbit holes learning irrelevant things?
3. Is the goal itself unclear or impossible with current knowledge?
4. Did you learn the WRONG concepts (e.g., medical terms instead of math)?

Provide:
ROOT_CAUSE: [One sentence: what went wrong]
SUGGESTED_FIX: [Specific action to take]
CONFIDENCE: [0.0-1.0: how sure are you of this analysis]
"""

        response = self.llm.generate(
            system_prompt="You are analyzing your own learning failure. Be brutally honest.",
            user_input=analysis_prompt
        )

        text = response.get("text", "")

        # Parse response
        root_cause = "Unknown"
        suggested_fix = "Retry with same approach"
        confidence = 0.5

        for line in text.split('\n'):
            if "ROOT_CAUSE:" in line:
                root_cause = line.split("ROOT_CAUSE:", 1)[1].strip()
            elif "SUGGESTED_FIX:" in line:
                suggested_fix = line.split("SUGGESTED_FIX:", 1)[1].strip()
            elif "CONFIDENCE:" in line:
                try:
                    conf_str = line.split("CONFIDENCE:", 1)[1].strip()
                    confidence = float(conf_str.split()[0])
                except:
                    pass

        analysis = FailureAnalysis(
            goal=goal,
            attempts=attempts,
            missing_concepts=missing_concepts,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            confidence=confidence
        )

        print(f"[🧠] Root cause: {root_cause}")
        print(f"[🧠] Suggested fix: {suggested_fix}")
        print(f"[🧠] Confidence in analysis: {confidence:.2f}")
        print("="*60 + "\n")

        self.reflection_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "failure_analysis",
            "goal": goal,
            "analysis": analysis.__dict__
        })

        return analysis

    async def check_if_on_track(
        self,
        original_goal: str,
        current_concept: str,
        concepts_learned: List[str],
        depth: int
    ) -> Tuple[bool, str]:
        """
        Check if we're still on track for the original goal.

        Returns: (on_track: bool, reason: str)
        """
        # If we've learned too many concepts, something's wrong
        if len(concepts_learned) > 20:
            return False, f"Learned {len(concepts_learned)} concepts - way too many for one goal!"

        # If we're too deep, we're probably lost
        if depth > 5:
            return False, f"Depth {depth} - went too deep into prerequisites!"

        # Use LLM to check relevance
        check_prompt = f"""Original goal: {original_goal}

Currently learning: {current_concept}
Already learned: {', '.join(concepts_learned[-10:])}
Current depth: {depth} levels

Question: Is "{current_concept}" ACTUALLY necessary for "{original_goal}"?

Answer with:
ON_TRACK: yes/no
REASON: [One sentence explanation]
"""

        response = self.llm.generate(
            system_prompt="You are checking if your learning is still relevant to the goal.",
            user_input=check_prompt
        )

        text = response.get("text", "").lower()

        on_track = "on_track: yes" in text or "on_track:yes" in text
        reason = "Relevance unclear"

        for line in text.split('\n'):
            if "reason:" in line.lower():
                reason = line.split(":", 1)[1].strip()
                break

        if not on_track:
            print(f"\n[🧠] METACOGNITION: Going OFF TRACK!")
            print(f"[🧠] Reason: {reason}")
            print(f"[🧠] Consider: Skip '{current_concept}' and work with what you have\n")

        return on_track, reason

    async def assess_true_confidence(
        self,
        concept: str,
        definition: str,
        stated_confidence: float
    ) -> float:
        """
        Deep check: Do we REALLY understand this?

        Not just "is it in memory" but "can we apply it in novel situations?"
        """
        test_prompt = f"""You learned: {concept}
Definition: {definition}

Your stated confidence: {stated_confidence:.2f}

Now answer: Can you REALLY apply this concept in NEW situations you haven't seen?
Not just recite the definition, but USE it creatively?

Rate your TRUE confidence (0.0-1.0):
TRUE_CONFIDENCE: [number]
REASON: [why this rating]
"""

        response = self.llm.generate(
            system_prompt="Be honest about what you truly understand vs what you just memorized.",
            user_input=test_prompt
        )

        text = response.get("text", "")

        true_confidence = stated_confidence  # Default to stated

        for line in text.split('\n'):
            if "TRUE_CONFIDENCE:" in line:
                try:
                    conf_str = line.split("TRUE_CONFIDENCE:", 1)[1].strip()
                    true_confidence = float(conf_str.split()[0])
                except:
                    pass

        # If true confidence is much lower, warn
        if true_confidence < stated_confidence - 0.15:
            print(f"[🧠] METACOGNITION: Stated confidence {stated_confidence:.2f} but TRUE confidence only {true_confidence:.2f}")
            print(f"[🧠] You may have memorized but not truly understood!")

        return true_confidence

    def identify_knowledge_gaps(
        self,
        goal_domain: str,
        known_concepts: List[str]
    ) -> List[str]:
        """
        Proactively identify what's missing from knowledge base.

        Returns: List of important concepts we should learn
        """
        # Domain-specific essential concepts
        essentials = {
            "mathematics": [
                "addition", "subtraction", "multiplication", "division",
                "fractions", "decimals", "percentages", "exponents",
                "equations", "variables", "functions", "graphs"
            ],
            "programming": [
                "variables", "loops", "conditionals", "functions",
                "arrays", "objects", "classes", "algorithms"
            ],
            "science": [
                "atoms", "molecules", "energy", "force", "motion",
                "cells", "DNA", "evolution", "periodic table"
            ]
        }

        if goal_domain not in essentials:
            return []

        required = essentials[goal_domain]
        known_lower = [c.lower() for c in known_concepts]

        gaps = []
        for concept in required:
            if not any(concept in k for k in known_lower):
                gaps.append(concept)

        if gaps:
            print(f"\n[🧠] KNOWLEDGE GAPS DETECTED in {goal_domain}:")
            for gap in gaps[:5]:
                print(f"[🧠]   - Missing: {gap}")
            print()

        return gaps

    def suggest_learning_strategy(
        self,
        concept: str,
        difficulty: float,
        past_failures: int
    ) -> Dict[str, any]:
        """
        Suggest how to approach learning this concept based on reflection.
        """
        strategy = {
            "break_into_smaller_parts": difficulty > 0.7,
            "seek_multiple_sources": past_failures > 2,
            "use_analogies": difficulty > 0.6,
            "practice_more": difficulty > 0.5,
            "skip_deep_prerequisites": difficulty < 0.3 and past_failures == 0
        }

        if past_failures > 3:
            print(f"[🧠] METACOGNITION: Failed {past_failures} times on {concept}")
            print(f"[🧠] Suggestion: Try a completely different approach!")
            strategy["try_different_approach"] = True

        return strategy
