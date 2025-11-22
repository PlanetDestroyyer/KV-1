"""
Explainable Reasoning & Active Learning Module

Combines two critical AGI capabilities:
1. Self-Explanation: System explains its reasoning clearly
2. Active Learning: System asks questions when uncertain

Key principle: True intelligence can explain itself AND knows what it doesn't know!
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ReasoningStep:
    """One step in a reasoning chain"""
    step_num: int
    action: str
    reason: str
    confidence: float


@dataclass
class Clarification:
    """A question to clarify ambiguity"""
    question: str
    missing_info: str
    importance: float  # How critical is this info?


class ExplainableReasoner:
    """
    Makes AGI reasoning transparent and interactive.

    "Any intelligent fool can make things bigger and more complex.
    It takes a touch of genius to move in the opposite direction." - E.F. Schumacher
    """

    def __init__(self, llm):
        self.llm = llm
        self.reasoning_history: List[List[ReasoningStep]] = []

    def explain_answer(
        self,
        question: str,
        answer: str,
        concepts_used: List[str],
        confidence: float
    ) -> str:
        """
        Generate human-readable explanation of reasoning.

        Args:
            question: Original question
            answer: Answer provided
            concepts_used: Concepts from knowledge base that were used
            confidence: Confidence in answer

        Returns:
            Clear explanation of reasoning process
        """
        print(f"[Explanation] Generating reasoning trace...")

        prompt = f"""Explain HOW you arrived at this answer step-by-step.

QUESTION: {question}
ANSWER: {answer}
CONCEPTS USED: {', '.join(concepts_used) if concepts_used else 'none'}

Provide a clear, step-by-step explanation:
1. What did you understand from the question?
2. What knowledge did you apply?
3. How did you reach the answer?
4. Why is this answer correct?

Make it clear enough for a student to follow.
"""

        response = self.llm.generate(
            system_prompt="You explain your reasoning clearly and step-by-step.",
            user_input=prompt
        )

        explanation = response.get("text", "")

        # Add confidence caveat if low
        if confidence < 0.7:
            explanation += f"\n\n⚠️ Note: Confidence is {confidence:.0%}. This answer may need verification."

        return explanation

    def identify_ambiguities(
        self,
        question: str
    ) -> List[Clarification]:
        """
        Identify what information is missing or ambiguous.

        This is ACTIVE LEARNING - the system asks questions!
        """
        print(f"[ActiveLearning] Checking for ambiguities...")

        prompt = f"""Analyze this question for missing or ambiguous information:

QUESTION: {question}

What information would you need to answer this accurately?
List any:
- Missing parameters or values
- Ambiguous terms that could mean multiple things
- Unstated assumptions

Format:
MISSING: [what's missing]
IMPORTANCE: [critical/helpful/optional]
QUESTION: [what to ask the user]

Example:
MISSING: Which shape (circle, square, etc.)
IMPORTANCE: critical
QUESTION: "What shape are you asking about?"
"""

        response = self.llm.generate(
            system_prompt="You identify ambiguities and missing information in questions.",
            user_input=prompt
        )

        text = response.get("text", "")

        # Parse clarifications
        clarifications = []
        current = {}

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("MISSING:"):
                if current:
                    clarifications.append(self._build_clarification(current))
                current = {"missing": line.split(":", 1)[1].strip()}
            elif line.startswith("IMPORTANCE:"):
                current["importance"] = line.split(":", 1)[1].strip()
            elif line.startswith("QUESTION:"):
                current["question"] = line.split(":", 1)[1].strip().strip('"')

        if current:
            clarifications.append(self._build_clarification(current))

        if clarifications:
            print(f"[ActiveLearning] Found {len(clarifications)} ambiguities")

        return clarifications

    def _build_clarification(self, data: Dict) -> Clarification:
        """Build Clarification object from parsed data"""
        importance_map = {
            "critical": 1.0,
            "helpful": 0.7,
            "optional": 0.3
        }

        return Clarification(
            question=data.get("question", ""),
            missing_info=data.get("missing", ""),
            importance=importance_map.get(data.get("importance", "helpful").lower(), 0.5)
        )

    def should_ask_clarification(
        self,
        clarifications: List[Clarification],
        threshold: float = 0.7
    ) -> bool:
        """
        Decide if we should ask for clarification before attempting.

        Returns True if any critical information is missing.
        """
        for clarif in clarifications:
            if clarif.importance >= threshold:
                return True
        return False

    def format_clarification_questions(
        self,
        clarifications: List[Clarification]
    ) -> str:
        """Format clarification questions for user"""
        if not clarifications:
            return ""

        questions = ["I need some clarification:", ""]

        for i, clarif in enumerate(clarifications, 1):
            importance_marker = "❗" if clarif.importance >= 0.8 else "❓"
            questions.append(f"{importance_marker} {i}. {clarif.question}")

        return "\n".join(questions)

    def trace_reasoning(
        self,
        question: str,
        answer: str,
        llm
    ) -> List[ReasoningStep]:
        """
        Generate detailed reasoning trace.

        Shows the THOUGHT PROCESS, not just the answer.
        """
        prompt = f"""Break down your reasoning into explicit steps:

QUESTION: {question}
FINAL ANSWER: {answer}

Provide a detailed reasoning trace:
1. [First step] - [Why you did this]
2. [Second step] - [Why you did this]
...

Be explicit about EVERY step in your thinking.
"""

        response = llm.generate(
            system_prompt="You provide detailed reasoning traces showing every step of your thought process.",
            user_input=prompt
        )

        text = response.get("text", "")

        # Parse steps
        steps = []
        for i, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Extract step
                step_text = line.lstrip('0123456789.-) ').strip()
                if ' - ' in step_text:
                    action, reason = step_text.split(' - ', 1)
                    steps.append(ReasoningStep(
                        step_num=i,
                        action=action.strip(),
                        reason=reason.strip(),
                        confidence=0.8  # Default confidence
                    ))

        return steps

    def verify_reasoning(
        self,
        question: str,
        answer: str,
        reasoning_steps: List[ReasoningStep],
        llm
    ) -> Tuple[bool, str]:
        """
        Verify that reasoning is valid.

        Returns (is_valid, critique)
        """
        steps_text = "\n".join([
            f"{s.step_num}. {s.action} - {s.reason}"
            for s in reasoning_steps
        ])

        prompt = f"""Verify this reasoning for logical soundness:

QUESTION: {question}
ANSWER: {answer}

REASONING:
{steps_text}

Is this reasoning valid?
- Are all steps logically sound?
- Are there any gaps or errors?
- Does it reach the correct conclusion?

Respond with:
VALID: [yes/no]
CRITIQUE: [explanation of any issues]
"""

        response = llm.generate(
            system_prompt="You critically evaluate reasoning for logical errors.",
            user_input=prompt
        )

        text = response.get("text", "")

        is_valid = "VALID: yes" in text.lower()
        critique = ""

        if "CRITIQUE:" in text:
            critique = text.split("CRITIQUE:")[1].strip()

        return is_valid, critique
