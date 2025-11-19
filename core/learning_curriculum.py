"""
Concept-based curriculum for guided self-discovery learning.

Defines progression through language ’ numbers ’ mathematics ’ physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Concept:
    """A single concept to be learned."""
    name: str
    search_query: str
    keywords: List[str]  # Keywords to verify understanding
    description: str


@dataclass
class Phase:
    """A learning phase containing related concepts."""
    name: str
    description: str
    concepts: List[Concept]

    def progress(self, learned_concepts: set) -> float:
        """Return progress through this phase (0.0 to 1.0)."""
        if not self.concepts:
            return 1.0
        learned = sum(1 for c in self.concepts if c.name in learned_concepts)
        return learned / len(self.concepts)

    def next_concept(self, learned_concepts: set) -> Optional[Concept]:
        """Get the next concept to learn in this phase."""
        for concept in self.concepts:
            if concept.name not in learned_concepts:
                return concept
        return None


class LearningCurriculum:
    """Manages learning progression through concept phases."""

    def __init__(self):
        self.phases = self._build_phases()
        self.learned_concepts = set()
        self.current_phase_index = 0

    def _build_phases(self) -> List[Phase]:
        """Build the curriculum phases."""

        # Phase 1: Language Fundamentals
        language_phase = Phase(
            name="Language Fundamentals",
            description="Understanding basic linguistic structures",
            concepts=[
                Concept(
                    name="word",
                    search_query="what is a word in language",
                    keywords=["word", "meaning", "language", "unit"],
                    description="Basic unit of language"
                ),
                Concept(
                    name="sentence",
                    search_query="what is a sentence grammar",
                    keywords=["sentence", "subject", "predicate", "complete thought"],
                    description="Complete grammatical unit"
                ),
                Concept(
                    name="paragraph",
                    search_query="what is a paragraph writing",
                    keywords=["paragraph", "sentences", "topic", "idea"],
                    description="Group of related sentences"
                ),
                Concept(
                    name="grammar",
                    search_query="what is grammar in language",
                    keywords=["grammar", "rules", "structure", "syntax"],
                    description="Rules of language structure"
                ),
            ]
        )

        # Phase 2: Number Concepts
        number_phase = Phase(
            name="Number Concepts",
            description="Understanding numbers and counting",
            concepts=[
                Concept(
                    name="number",
                    search_query="what is a number mathematics",
                    keywords=["number", "quantity", "count", "value"],
                    description="Mathematical object representing quantity"
                ),
                Concept(
                    name="counting",
                    search_query="what is counting numbers",
                    keywords=["counting", "sequence", "one", "two", "three"],
                    description="Enumeration of objects"
                ),
                Concept(
                    name="place_value",
                    search_query="what is place value",
                    keywords=["place value", "tens", "hundreds", "digit"],
                    description="Value based on position"
                ),
                Concept(
                    name="operations",
                    search_query="basic arithmetic operations",
                    keywords=["addition", "subtraction", "multiplication", "division"],
                    description="Basic mathematical operations"
                ),
                Concept(
                    name="fractions",
                    search_query="what are fractions",
                    keywords=["fraction", "numerator", "denominator", "part"],
                    description="Parts of a whole"
                ),
            ]
        )

        # Phase 3: Algebra Fundamentals
        algebra_phase = Phase(
            name="Algebra Fundamentals",
            description="Variables, equations, and functions",
            concepts=[
                Concept(
                    name="variable",
                    search_query="what is a variable in algebra",
                    keywords=["variable", "unknown", "symbol", "x"],
                    description="Symbol representing unknown value"
                ),
                Concept(
                    name="equation",
                    search_query="what is an equation algebra",
                    keywords=["equation", "equals", "solve", "solution"],
                    description="Mathematical statement of equality"
                ),
                Concept(
                    name="function",
                    search_query="what is a function mathematics",
                    keywords=["function", "input", "output", "mapping"],
                    description="Relationship between inputs and outputs"
                ),
                Concept(
                    name="linear_equation",
                    search_query="what is a linear equation",
                    keywords=["linear", "slope", "intercept", "line"],
                    description="Equation forming a straight line"
                ),
            ]
        )

        # Phase 4: Calculus Basics
        calculus_phase = Phase(
            name="Calculus Basics",
            description="Derivatives and rates of change",
            concepts=[
                Concept(
                    name="derivative",
                    search_query="what is a derivative calculus",
                    keywords=["derivative", "rate", "change", "slope"],
                    description="Rate of change of a function"
                ),
                Concept(
                    name="limit",
                    search_query="what is a limit in calculus",
                    keywords=["limit", "approach", "infinity", "continuous"],
                    description="Value a function approaches"
                ),
                Concept(
                    name="integral",
                    search_query="what is an integral calculus",
                    keywords=["integral", "area", "accumulation", "antiderivative"],
                    description="Accumulation of quantities"
                ),
            ]
        )

        # Phase 5: Thermodynamics
        thermo_phase = Phase(
            name="Thermodynamics",
            description="Energy, heat, and physical systems",
            concepts=[
                Concept(
                    name="energy",
                    search_query="what is energy physics",
                    keywords=["energy", "work", "power", "conservation"],
                    description="Capacity to do work"
                ),
                Concept(
                    name="temperature",
                    search_query="what is temperature thermodynamics",
                    keywords=["temperature", "heat", "thermal", "kelvin"],
                    description="Measure of thermal energy"
                ),
                Concept(
                    name="first_law",
                    search_query="first law of thermodynamics",
                    keywords=["first law", "energy conservation", "internal energy"],
                    description="Energy conservation in thermodynamics"
                ),
                Concept(
                    name="entropy",
                    search_query="what is entropy thermodynamics",
                    keywords=["entropy", "disorder", "second law", "irreversible"],
                    description="Measure of disorder in a system"
                ),
            ]
        )

        return [language_phase, number_phase, algebra_phase, calculus_phase, thermo_phase]

    @property
    def current_phase(self) -> Optional[Phase]:
        """Get the current learning phase."""
        if self.current_phase_index < len(self.phases):
            return self.phases[self.current_phase_index]
        return None

    def next_concept_to_learn(self) -> Optional[Concept]:
        """Get the next concept to learn across all phases."""
        phase = self.current_phase
        if not phase:
            return None

        concept = phase.next_concept(self.learned_concepts)
        if concept:
            return concept

        # Current phase complete, move to next
        self.current_phase_index += 1
        return self.next_concept_to_learn()

    def mark_learned(self, concept_name: str):
        """Mark a concept as learned."""
        self.learned_concepts.add(concept_name)

    def verify_concept(self, concept_name: str, llm_response: str) -> bool:
        """Verify if LLM understands a concept based on keywords."""
        # Find the concept
        for phase in self.phases:
            for concept in phase.concepts:
                if concept.name == concept_name:
                    # Check if response contains at least 50% of keywords
                    response_lower = llm_response.lower()
                    matches = sum(1 for kw in concept.keywords if kw.lower() in response_lower)
                    return matches >= len(concept.keywords) * 0.5
        return False

    def get_progress(self) -> Dict[str, float]:
        """Get progress through all phases."""
        return {
            phase.name: phase.progress(self.learned_concepts)
            for phase in self.phases
        }

    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_phase is None

    def summary(self) -> Dict:
        """Get curriculum summary."""
        return {
            "current_phase": self.current_phase.name if self.current_phase else "Complete",
            "learned_concepts": len(self.learned_concepts),
            "total_concepts": sum(len(p.concepts) for p in self.phases),
            "progress": self.get_progress(),
        }
