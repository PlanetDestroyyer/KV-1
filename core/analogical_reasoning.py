"""
Analogical Reasoning Module

Enables AGI to solve problems using analogies:
- "A is to B as C is to ?" patterns
- Structural mapping between concepts
- Creative problem solving through analogy

Example: "atom : molecule :: cell : organism"
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Analogy:
    """Represents an A:B :: C:D analogy"""
    a: str
    b: str
    c: str
    d: str
    confidence: float
    relationship_type: str  # "part-whole", "cause-effect", "category", etc.


class AnalogyEngine:
    """
    Reasons by analogy - a key component of human intelligence.

    "Analogy is the core of cognition" - Douglas Hofstadter
    """

    def __init__(self, memory, llm):
        self.memory = memory
        self.llm = llm
        self.analogy_cache: Dict[str, List[Analogy]] = {}

    def solve_by_analogy(
        self,
        a: str,
        b: str,
        c: str
    ) -> List[Tuple[str, float]]:
        """
        Solve A:B :: C:? using analogical reasoning.

        Args:
            a, b: Source pair showing relationship
            c: Target to find analogy for

        Returns:
            List of (candidate, confidence) tuples

        Example:
            solve_by_analogy("atom", "molecule", "cell")
            → [("organism", 0.95), ("tissue", 0.85)]
        """
        print(f"[Analogy] Solving: {a}:{b} :: {c}:?")

        # Use LLM to identify relationship
        relationship = self._identify_relationship(a, b)
        print(f"[Analogy] Relationship: {relationship}")

        # Find concepts that have same relationship with c
        candidates = self._find_analogous_concepts(c, relationship)

        # Rank by structural similarity
        ranked = self._rank_candidates(a, b, c, candidates)

        return ranked[:5]

    def _identify_relationship(self, a: str, b: str) -> str:
        """Identify the relationship between A and B"""
        prompt = f"""What is the relationship between "{a}" and "{b}"?

Examples of relationships:
- part-whole: "wheel" → "car"
- cause-effect: "rain" → "wet ground"
- category-instance: "mammal" → "dog"
- function: "key" → "lock"
- composition: "atoms" → "molecule"

Identify the SPECIFIC relationship type and describe it in 1-2 words.

Respond with ONLY the relationship type (e.g., "part-whole", "cause-effect", etc.)
"""

        response = self.llm.generate(
            system_prompt="You identify relationships between concepts for analogical reasoning.",
            user_input=prompt
        )

        return response.get("text", "unknown").strip().lower()

    def _find_analogous_concepts(
        self,
        c: str,
        relationship: str
    ) -> List[str]:
        """Find concepts that have the same relationship with c"""
        # Search memory for concepts related to c
        candidates = []

        # Use LLM to generate candidates
        prompt = f"""Given the relationship "{relationship}", what concept relates to "{c}" in the same way?

Generate 3-5 concepts that could complete: "{c}" → ?

Think of concepts where the relationship is: {relationship}

List only concept names, one per line.
"""

        response = self.llm.generate(
            system_prompt="You generate analogical mappings.",
            user_input=prompt
        )

        text = response.get("text", "")
        for line in text.split('\n'):
            line = line.strip().lstrip('-').strip()
            if line and len(line) < 50:
                candidates.append(line)

        return candidates

    def _rank_candidates(
        self,
        a: str,
        b: str,
        c: str,
        candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """Rank candidates by structural similarity to the analogy"""
        ranked = []

        for candidate in candidates:
            # Compute confidence based on structural alignment
            confidence = self._compute_analogy_confidence(a, b, c, candidate)
            ranked.append((candidate, confidence))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _compute_analogy_confidence(
        self,
        a: str,
        b: str,
        c: str,
        d: str
    ) -> float:
        """
        Compute confidence that A:B :: C:D is valid.

        Uses structural similarity and semantic coherence.
        """
        # For now, use simple heuristic
        # In production, would use deep structure mapping

        # Check if concepts exist in memory
        a_known = a.lower() in [k.lower() for k in self.memory.concepts.keys()]
        b_known = b.lower() in [k.lower() for k in self.memory.concepts.keys()]
        c_known = c.lower() in [k.lower() for k in self.memory.concepts.keys()]
        d_known = d.lower() in [k.lower() for k in self.memory.concepts.keys()]

        # Base confidence on knowledge availability
        knowledge_score = sum([a_known, b_known, c_known, d_known]) / 4.0

        # Add bonus if we've seen this relationship before
        cache_key = f"{a}:{b}"
        if cache_key in self.analogy_cache:
            knowledge_score += 0.1

        return min(knowledge_score, 1.0)

    def create_analogy_for_explanation(
        self,
        difficult_concept: str
    ) -> Optional[str]:
        """
        Create an analogy to explain a difficult concept.

        Example:
            difficult: "electricity"
            analogy: "Electricity is like water flowing through pipes"
        """
        prompt = f"""Create a simple analogy to explain "{difficult_concept}" to someone who doesn't understand it.

Use the format: "{difficult_concept} is like [familiar concept] because [reason]"

Make it intuitive and easy to understand.
"""

        response = self.llm.generate(
            system_prompt="You create helpful analogies for explaining difficult concepts.",
            user_input=prompt
        )

        analogy = response.get("text", "").strip()

        if analogy:
            print(f"[Analogy] Created explanation: {analogy}")
            return analogy

        return None

    def find_creative_solutions(
        self,
        problem: str
    ) -> List[str]:
        """
        Use analogical reasoning to find creative solutions.

        Looks for similar problems in other domains and transfers solutions.
        """
        print(f"[Analogy] Finding creative solutions via cross-domain analogies...")

        prompt = f"""Find analogies to this problem from OTHER domains:

PROBLEM: {problem}

Think creatively:
1. What similar challenges exist in nature, physics, biology, engineering, etc.?
2. How were those challenges solved?
3. Can we apply those solutions here?

Provide 2-3 analogical solutions from different domains.
"""

        response = self.llm.generate(
            system_prompt="You find creative solutions through cross-domain analogies.",
            user_input=prompt
        )

        solutions = []
        text = response.get("text", "")
        for line in text.split('\n'):
            if line.strip() and len(line) > 20:
                solutions.append(line.strip())

        return solutions[:3]
