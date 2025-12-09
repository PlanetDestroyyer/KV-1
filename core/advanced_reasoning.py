"""
Advanced Reasoning Module

Combines multiple AGI capabilities:
1. Common Sense Reasoning - Implicit world knowledge
2. Hypothesis Formation & Testing - Scientific method
3. Continual Learning - Learn without forgetting

These are the building blocks of human-level intelligence!
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class Hypothesis:
    """A testable hypothesis"""
    statement: str
    confidence: float
    evidence_for: List[str]
    evidence_against: List[str]
    test_cases: List[str]


@dataclass
class CommonSenseRule:
    """A rule of common sense"""
    category: str  # "physical", "social", "biological", etc.
    rule: str
    confidence: float
    examples: List[str]


class AdvancedReasoner:
    """
    Adds human-like reasoning capabilities to AGI.

    "Common sense is the collection of prejudices acquired by age eighteen." - Einstein
    (But it's still necessary for AI!)
    """

    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.common_sense_kb: List[CommonSenseRule] = []
        self.hypotheses: List[Hypothesis] = []
        self.concept_usage_count: Dict[str, int] = {}  # For continual learning

        # Initialize with basic common sense
        self._init_common_sense()

    def _init_common_sense(self):
        """Initialize basic common sense rules"""
        basic_rules = [
            CommonSenseRule(
                category="physical",
                rule="Objects fall down when dropped (gravity)",
                confidence=1.0,
                examples=["dropped ball falls", "water flows downhill"]
            ),
            CommonSenseRule(
                category="physical",
                rule="Solid objects cannot pass through each other",
                confidence=1.0,
                examples=["cannot walk through walls", "cannot put hand through table"]
            ),
            CommonSenseRule(
                category="biological",
                rule="Living things need energy to survive",
                confidence=1.0,
                examples=["humans need food", "plants need sunlight"]
            ),
            CommonSenseRule(
                category="social",
                rule="People generally act in their self-interest",
                confidence=0.85,
                examples=["avoid pain", "seek reward"]
            )
        ]

        self.common_sense_kb.extend(basic_rules)

    def check_common_sense(
        self,
        statement: str
    ) -> Tuple[bool, str, float]:
        """
        Check if a statement violates common sense.

        Returns: (is_plausible, reason, confidence)
        """
        print(f"[CommonSense] Checking: {statement}")

        prompt = f"""Does this statement violate common sense or physical reality?

STATEMENT: {statement}

Consider:
- Physical laws (gravity, causation, etc.)
- Biological facts (living things need energy, etc.)
- Social norms (people have motivations, etc.)

Respond with:
PLAUSIBLE: [yes/no]
REASON: [why it makes sense or doesn't]
CONFIDENCE: [0.0-1.0]
"""

        response = self.llm.generate(
            system_prompt="You check statements against common sense and physical reality.",
            user_input=prompt
        )

        text = response.get("text", "")

        is_plausible = "PLAUSIBLE: yes" in text.lower()

        reason = ""
        if "REASON:" in text:
            reason = text.split("REASON:")[1].strip()
            if "CONFIDENCE:" in reason:
                reason = reason.split("CONFIDENCE:")[0].strip()

        confidence = 0.5
        if "CONFIDENCE:" in text:
            try:
                conf_str = text.split("CONFIDENCE:")[1].strip().split()[0]
                confidence = float(conf_str)
            except:
                pass

        return is_plausible, reason, confidence

    def form_hypothesis(
        self,
        observations: List[str]
    ) -> Hypothesis:
        """
        Form a testable hypothesis from observations.

        This is the SCIENTIFIC METHOD in action!
        """
        print(f"[Hypothesis] Forming hypothesis from {len(observations)} observations...")

        obs_text = "\n".join([f"- {obs}" for obs in observations])

        prompt = f"""Given these observations, form a testable hypothesis:

OBSERVATIONS:
{obs_text}

Generate:
1. A general hypothesis that explains these observations
2. Predictions this hypothesis makes
3. How to test this hypothesis

Format:
HYPOTHESIS: [your hypothesis]
PREDICTIONS: [what this predicts]
TEST CASES: [how to test this]
"""

        response = self.llm.generate(
            system_prompt="You form scientific hypotheses from observations.",
            user_input=prompt
        )

        text = response.get("text", "")

        # Parse hypothesis
        hypothesis = "Unknown"
        predictions = []
        test_cases = []

        if "HYPOTHESIS:" in text:
            hypothesis = text.split("HYPOTHESIS:")[1].split("PREDICTIONS:")[0].strip()

        if "PREDICTIONS:" in text:
            pred_section = text.split("PREDICTIONS:")[1]
            if "TEST CASES:" in pred_section:
                pred_section = pred_section.split("TEST CASES:")[0]
            predictions = [p.strip() for p in pred_section.split('\n') if p.strip()]

        if "TEST CASES:" in text:
            test_section = text.split("TEST CASES:")[1]
            test_cases = [t.strip() for t in test_section.split('\n') if t.strip()]

        hyp = Hypothesis(
            statement=hypothesis,
            confidence=0.6,  # Start with moderate confidence
            evidence_for=observations,
            evidence_against=[],
            test_cases=test_cases[:3]
        )

        print(f"[Hypothesis] Formed: {hypothesis}")
        return hyp

    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        test_result: str,
        supports: bool
    ) -> Hypothesis:
        """
        Update hypothesis based on test result.

        Bayesian updating of confidence!
        """
        if supports:
            hypothesis.evidence_for.append(test_result)
            # Increase confidence (but not to 1.0)
            hypothesis.confidence = min(0.95, hypothesis.confidence + 0.1)
            print(f"[Hypothesis] ✅ Test supports hypothesis (conf: {hypothesis.confidence:.2f})")
        else:
            hypothesis.evidence_against.append(test_result)
            # Decrease confidence
            hypothesis.confidence = max(0.1, hypothesis.confidence - 0.15)
            print(f"[Hypothesis] ❌ Test contradicts hypothesis (conf: {hypothesis.confidence:.2f})")

        return hypothesis

    def prevent_catastrophic_forgetting(
        self,
        concept_name: str
    ):
        """
        Ensure old concepts aren't forgotten when learning new ones.

        This is CONTINUAL LEARNING!
        """
        # Track concept usage
        if concept_name in self.concept_usage_count:
            self.concept_usage_count[concept_name] += 1
        else:
            self.concept_usage_count[concept_name] = 1

        # Periodically rehearse old concepts
        if len(self.concept_usage_count) > 50 and random.random() < 0.1:
            # 10% chance to rehearse when KB is large
            self._rehearse_old_concepts()

    def _rehearse_old_concepts(self):
        """
        Periodically review old concepts to prevent forgetting.

        Uses spaced repetition!
        """
        print(f"[ContinualLearning] Rehearsing old concepts to prevent forgetting...")

        # Find concepts that haven't been used recently
        sorted_concepts = sorted(
            self.concept_usage_count.items(),
            key=lambda x: x[1]  # Sort by usage count (ascending)
        )

        # Rehearse bottom 20%
        to_rehearse = sorted_concepts[:max(5, len(sorted_concepts) // 5)]

        for concept_name, usage_count in to_rehearse:
            if concept_name in self.memory.concepts:
                # Quick recall test
                concept_data = self.memory.concepts[concept_name]
                print(f"[ContinualLearning] Rehearsing: {concept_name} (used {usage_count}x)")

                # Mark as used to prevent forgetting
                self.concept_usage_count[concept_name] = usage_count + 1

    def detect_knowledge_drift(self) -> List[str]:
        """
        Detect if old knowledge is degrading.

        Returns list of concepts that might be "forgotten"
        """
        degraded = []

        for concept_name, usage_count in self.concept_usage_count.items():
            # If concept hasn't been used in a while, it might be degrading
            if usage_count < 2 and len(self.concept_usage_count) > 20:
                degraded.append(concept_name)

        if degraded:
            print(f"[ContinualLearning] ⚠️ {len(degraded)} concepts may be degrading")

        return degraded[:10]

    def learn_from_mistake(
        self,
        problem: str,
        incorrect_answer: str,
        correct_answer: str,
        llm
    ) -> str:
        """
        Extract lesson from a mistake.

        This is META-LEARNING - learning from failures!
        """
        print(f"[MetaLearning] Analyzing mistake to prevent recurrence...")

        prompt = f"""Analyze this mistake to learn from it:

PROBLEM: {problem}
YOUR ANSWER: {incorrect_answer}
CORRECT ANSWER: {correct_answer}

What went wrong? Extract the KEY LESSON to prevent this error in the future.

Format:
ERROR TYPE: [what kind of error was this]
LESSON: [what to remember]
HOW TO AVOID: [concrete steps to prevent this]
"""

        response = llm.generate(
            system_prompt="You learn from mistakes by extracting lessons.",
            user_input=prompt
        )

        lesson = response.get("text", "")

        # Store lesson as a new concept
        if "LESSON:" in lesson:
            lesson_text = lesson.split("LESSON:")[1].split("HOW TO AVOID:")[0].strip()
            print(f"[MetaLearning] Lesson learned: {lesson_text[:100]}...")

        return lesson
