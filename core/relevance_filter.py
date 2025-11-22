"""
Prerequisite Relevance Filter

Prevents learning irrelevant concepts by checking if prerequisites
are ACTUALLY needed for the goal.

This solves the problem of learning category theory for prime numbers!
"""

from typing import List, Tuple


class RelevanceFilter:
    """
    Filters out irrelevant prerequisites before learning them.

    Uses LLM to ask: "Do I REALLY need to learn X to understand Y?"
    """

    def __init__(self, llm_bridge):
        self.llm = llm_bridge
        self.relevance_cache = {}  # Cache decisions

    async def is_prerequisite_relevant(
        self,
        prerequisite: str,
        target_concept: str,
        goal: str,
        goal_domain: str
    ) -> Tuple[bool, float, str]:
        """
        Check if a prerequisite is actually needed.

        Returns: (is_relevant: bool, confidence: float, reason: str)
        """
        # Check cache first
        cache_key = f"{prerequisite}|{target_concept}|{goal_domain}"
        if cache_key in self.relevance_cache:
            return self.relevance_cache[cache_key]

        # Obvious irrelevant cases (hard-coded rules)
        if self._is_obviously_irrelevant(prerequisite, goal_domain):
            result = (False, 1.0, f"'{prerequisite}' is clearly not in {goal_domain} domain")
            self.relevance_cache[cache_key] = result
            return result

        # Use LLM for nuanced checking
        check_prompt = f"""Question: To learn "{target_concept}" in order to achieve "{goal}", do I REALLY need to first learn "{prerequisite}"?

Context:
- Main goal: {goal}
- Goal domain: {goal_domain}
- Target concept: {target_concept}
- Proposed prerequisite: {prerequisite}

Think carefully: Is this prerequisite:
1. Absolutely necessary? (Can't understand target without it)
2. Helpful but not required? (Would help but can skip)
3. Completely irrelevant? (Different domain or unrelated)

Answer in this format:
RELEVANT: yes/no
CONFIDENCE: [0.0-1.0]
REASON: [One sentence why]

Examples of IRRELEVANT prerequisites:
- Learning "category theory" for "prime numbers" → NO (way too advanced)
- Learning "immune system factors" for "mathematical factors" → NO (wrong domain)
- Learning "operations research" for "basic math operations" → NO (wrong meaning)

Examples of RELEVANT prerequisites:
- Learning "addition" for "multiplication" → YES (multiplication uses addition)
- Learning "numbers" for "prime numbers" → YES (need to know what numbers are)
- Learning "division" for "factors" → YES (factors involve division)
"""

        response = self.llm.generate(
            system_prompt="You are checking if a prerequisite is truly necessary. Be strict - only say YES if absolutely needed.",
            user_input=check_prompt
        )

        text = response.get("text", "")

        # Parse response
        relevant = False
        confidence = 0.5
        reason = "Could not determine relevance"

        for line in text.split('\n'):
            line_lower = line.lower()
            if "relevant:" in line_lower:
                relevant = "yes" in line_lower.split("relevant:", 1)[1]
            elif "confidence:" in line_lower:
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    confidence = float(conf_str.split()[0])
                except:
                    pass
            elif "reason:" in line_lower:
                reason = line.split(":", 1)[1].strip()

        result = (relevant, confidence, reason)
        self.relevance_cache[cache_key] = result

        # Log decision
        symbol = "✓" if relevant else "✗"
        print(f"    [{symbol}] Prerequisite check: '{prerequisite}'")
        print(f"        → {'RELEVANT' if relevant else 'NOT RELEVANT'} (confidence: {confidence:.2f})")
        print(f"        → {reason}")

        return result

    def _is_obviously_irrelevant(self, prerequisite: str, goal_domain: str) -> bool:
        """
        Hard-coded rules for obviously irrelevant prerequisites.

        This catches common mistakes quickly without needing LLM.
        """
        prereq_lower = prerequisite.lower()

        # Domain mismatches
        domain_keywords = {
            "mathematics": ["immune", "cell", "organism", "protein", "dna", "medical", "biology", "chemistry"],
            "programming": ["photosynthesis", "evolution", "geology", "astronomy"],
            "science": ["syntax", "grammar", "literature", "poetry"]
        }

        if goal_domain in domain_keywords:
            for keyword in domain_keywords[goal_domain]:
                if keyword in prereq_lower:
                    return True

        # Overly advanced concepts for basic goals
        advanced_math = ["category theory", "hom-set", "morphism", "functor", "monad", "topos"]
        if goal_domain == "mathematics":
            for term in advanced_math:
                if term in prereq_lower:
                    # Only relevant if we're actually studying category theory
                    return "category" not in prereq_lower

        # Generic terms that are almost never real prerequisites
        too_generic = ["importance", "impact", "entities", "characteristics"]
        if any(term == prereq_lower for term in too_generic):
            return True

        return False

    async def filter_prerequisites(
        self,
        prerequisites: List[str],
        target_concept: str,
        goal: str,
        goal_domain: str
    ) -> Tuple[List[str], List[str]]:
        """
        Filter a list of prerequisites, keeping only relevant ones.

        Returns: (relevant_prereqs, filtered_out_prereqs)
        """
        relevant = []
        filtered = []

        for prereq in prerequisites:
            is_rel, confidence, reason = await self.is_prerequisite_relevant(
                prereq, target_concept, goal, goal_domain
            )

            if is_rel and confidence > 0.6:
                relevant.append(prereq)
            else:
                filtered.append(prereq)
                print(f"    [🚫] FILTERED OUT: '{prereq}' - {reason}")

        if filtered:
            print(f"\n    [i] Filtered {len(filtered)}/{len(prerequisites)} irrelevant prerequisites")
            print(f"    [i] Learning only {len(relevant)} relevant ones\n")

        return relevant, filtered
