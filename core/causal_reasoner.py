"""
Causal Reasoning Engine

Understands cause-effect relationships, counterfactuals, and interventions.
This is what enables AGI to understand "why" things happen!

Features:
- Causal chain discovery
- Counterfactual reasoning ("what if X didn't happen?")
- Intervention prediction ("if I do X, what happens?")
- Root cause analysis
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CausalRelation:
    """A cause-effect relationship."""
    cause: str
    effect: str
    strength: float  # 0.0-1.0 how strong is the causal link
    mechanism: str  # HOW does cause lead to effect
    confidence: float


@dataclass
class CounterfactualScenario:
    """A 'what if' scenario."""
    intervention: str  # What we change
    predicted_outcome: str  # What we expect to happen
    confidence: float
    reasoning: str


class CausalReasoner:
    """
    Reasons about cause and effect relationships.

    This is AGI's "understanding why" component!
    """

    def __init__(self, llm_bridge):
        self.llm = llm_bridge
        self.causal_graph: Dict[str, List[CausalRelation]] = defaultdict(list)
        self.counterfactuals: List[CounterfactualScenario] = []

    async def discover_causal_chain(
        self,
        observation: str,
        domain: str,
        max_depth: int = 3
    ) -> List[CausalRelation]:
        """
        Discover the causal chain behind an observation.

        Example: "Why are prime numbers important?"
        → "Used in cryptography" → "Factoring is hard" → "Security relies on hard problems"
        """
        print(f"\n[🔗] CAUSAL REASONING: Discovering causal chain for '{observation}'...")

        prompt = f"""Analyze the CAUSAL CHAIN behind this observation:

Observation: {observation}
Domain: {domain}

Find the cause-effect relationships. For each link in the chain:
1. What causes this?
2. What effect does it have?
3. HOW does the cause lead to the effect (mechanism)?

Format:
CAUSE: [what causes this]
EFFECT: [what it causes]
MECHANISM: [how/why the causal link works]
STRENGTH: [0.0-1.0 how strong]

Example for "Prime numbers are hard to find":
CAUSE: Testing primality requires checking many divisors
EFFECT: Finding large primes is computationally expensive
MECHANISM: Each test takes time, and larger numbers need more tests
STRENGTH: 0.9

Generate {max_depth} causal links.
"""

        response = self.llm.generate(
            system_prompt="You analyze causal relationships and mechanisms.",
            user_input=prompt
        )

        text = response.get("text", "")

        relations = []
        current_cause = None
        current_effect = None
        current_mechanism = ""
        current_strength = 0.5

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("CAUSE:"):
                current_cause = line.split("CAUSE:", 1)[1].strip()

            elif line.startswith("EFFECT:"):
                current_effect = line.split("EFFECT:", 1)[1].strip()

            elif line.startswith("MECHANISM:"):
                current_mechanism = line.split("MECHANISM:", 1)[1].strip()

            elif line.startswith("STRENGTH:"):
                try:
                    strength_str = line.split("STRENGTH:", 1)[1].strip()
                    current_strength = float(strength_str.split()[0])
                except:
                    current_strength = 0.5

                # Save the relation
                if current_cause and current_effect:
                    relation = CausalRelation(
                        cause=current_cause,
                        effect=current_effect,
                        strength=current_strength,
                        mechanism=current_mechanism,
                        confidence=0.8
                    )

                    relations.append(relation)
                    self.causal_graph[current_cause].append(relation)

                    print(f"[→] {current_cause}")
                    print(f"    ↓ ({current_strength:.2f}) {current_mechanism}")
                    print(f"    {current_effect}")

                    # Reset
                    current_cause = None
                    current_effect = None
                    current_mechanism = ""

        return relations

    async def counterfactual_reasoning(
        self,
        fact: str,
        intervention: str,
        domain: str
    ) -> CounterfactualScenario:
        """
        Reason about "what if" scenarios.

        Example:
        Fact: "Prime numbers are defined as having exactly 2 divisors"
        Intervention: "What if we allowed 3 divisors?"
        → "Then we'd include numbers like 4 (divisors: 1,2,4) - this would be semi-primes or composite numbers"
        """
        print(f"\n[🤔] CAUSAL REASONING: Counterfactual - What if {intervention}?")

        prompt = f"""Perform counterfactual reasoning:

Current fact: {fact}
Domain: {domain}

COUNTERFACTUAL: {intervention}

What would happen if we made this change? Consider:
1. Direct consequences
2. Cascading effects
3. What would break or change
4. What new properties would emerge

Format:
OUTCOME: [what would happen]
REASONING: [step-by-step why]
CONFIDENCE: [0.0-1.0]

Example:
OUTCOME: Numbers like 4, 6, 9 would become "primes" but this breaks unique factorization
REASONING: If 4 is prime and 2 is prime, then 8 = 2×4 but also 8 = 2×2×2, losing uniqueness
CONFIDENCE: 0.9
"""

        response = self.llm.generate(
            system_prompt="You reason about counterfactual scenarios and their consequences.",
            user_input=prompt
        )

        text = response.get("text", "")

        outcome = "Unknown"
        reasoning = "Could not determine"
        confidence = 0.5

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("OUTCOME:"):
                outcome = line.split("OUTCOME:", 1)[1].strip()

            elif line.startswith("REASONING:"):
                reasoning = line.split("REASONING:", 1)[1].strip()

            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split("CONFIDENCE:", 1)[1].strip()
                    confidence = float(conf_str.split()[0])
                except:
                    confidence = 0.5

        scenario = CounterfactualScenario(
            intervention=intervention,
            predicted_outcome=outcome,
            confidence=confidence,
            reasoning=reasoning
        )

        print(f"[💭] Outcome: {outcome}")
        print(f"[💭] Reasoning: {reasoning}")
        print(f"[💭] Confidence: {confidence:.2f}")

        self.counterfactuals.append(scenario)
        return scenario

    async def predict_intervention(
        self,
        current_state: str,
        intervention: str,
        domain: str
    ) -> Dict[str, any]:
        """
        Predict what happens if we intervene in a system.

        Example:
        State: "System learns prerequisites sequentially"
        Intervention: "Make it learn in parallel"
        → Prediction: "Faster but might miss dependencies, need careful ordering"
        """
        print(f"\n[🔮] CAUSAL REASONING: Predicting intervention '{intervention}'...")

        prompt = f"""Predict the consequences of an intervention:

Current state: {current_state}
Domain: {domain}
Intervention: {intervention}

Analyze:
1. Immediate effects (what changes directly)
2. Side effects (unintended consequences)
3. Risks (what could go wrong)
4. Benefits (what improves)

Format:
IMMEDIATE: [direct effects]
SIDE_EFFECTS: [unintended consequences]
RISKS: [what could go wrong]
BENEFITS: [what improves]
CONFIDENCE: [0.0-1.0]

Example:
IMMEDIATE: Multiple concepts learned simultaneously
SIDE_EFFECTS: Harder to track dependencies
RISKS: Might learn B before its prerequisite A
BENEFITS: 3-5x faster learning
CONFIDENCE: 0.8
"""

        response = self.llm.generate(
            system_prompt="You predict consequences of interventions in systems.",
            user_input=prompt
        )

        text = response.get("text", "")

        prediction = {
            "intervention": intervention,
            "immediate_effects": "",
            "side_effects": "",
            "risks": "",
            "benefits": "",
            "confidence": 0.5
        }

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("IMMEDIATE:"):
                prediction["immediate_effects"] = line.split("IMMEDIATE:", 1)[1].strip()
            elif line.startswith("SIDE_EFFECTS:"):
                prediction["side_effects"] = line.split("SIDE_EFFECTS:", 1)[1].strip()
            elif line.startswith("RISKS:"):
                prediction["risks"] = line.split("RISKS:", 1)[1].strip()
            elif line.startswith("BENEFITS:"):
                prediction["benefits"] = line.split("BENEFITS:", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split("CONFIDENCE:", 1)[1].strip()
                    prediction["confidence"] = float(conf_str.split()[0])
                except:
                    pass

        print(f"[⚡] Immediate: {prediction['immediate_effects']}")
        print(f"[⚠️] Risks: {prediction['risks']}")
        print(f"[✓] Benefits: {prediction['benefits']}")
        print(f"[📊] Confidence: {prediction['confidence']:.2f}")

        return prediction

    async def find_root_cause(
        self,
        problem: str,
        observations: List[str],
        domain: str
    ) -> Tuple[str, float, str]:
        """
        Find the root cause of a problem using causal reasoning.

        Uses "5 whys" technique to dig deeper.
        """
        print(f"\n[🔍] CAUSAL REASONING: Finding root cause of '{problem}'...")

        prompt = f"""Find the ROOT CAUSE of this problem:

Problem: {problem}
Domain: {domain}
Observations: {', '.join(observations)}

Use the "5 Whys" technique:
1. Why does this problem occur?
2. Why does that happen?
3. Why does that happen?
4. ... (continue until you find root cause)

ROOT_CAUSE: [the fundamental underlying cause]
EXPLANATION: [how this root cause leads to the problem]
CONFIDENCE: [0.0-1.0]

Example for "System learns irrelevant concepts":
ROOT_CAUSE: Domain detection fails, leading to wrong web search results
EXPLANATION: "factors" searches return medical results → system learns biology → goes off track
CONFIDENCE: 0.85
"""

        response = self.llm.generate(
            system_prompt="You find root causes using deep causal analysis.",
            user_input=prompt
        )

        text = response.get("text", "")

        root_cause = "Unknown"
        explanation = ""
        confidence = 0.5

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("ROOT_CAUSE:"):
                root_cause = line.split("ROOT_CAUSE:", 1)[1].strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.split("EXPLANATION:", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split("CONFIDENCE:", 1)[1].strip()
                    confidence = float(conf_str.split()[0])
                except:
                    pass

        print(f"[🎯] Root cause: {root_cause}")
        print(f"[📝] Explanation: {explanation}")
        print(f"[📊] Confidence: {confidence:.2f}")

        return root_cause, confidence, explanation

    def get_causal_paths(
        self,
        start: str,
        end: str,
        max_hops: int = 3
    ) -> List[List[CausalRelation]]:
        """
        Find all causal paths from start to end.

        Example: How does "division" causally lead to "cryptography"?
        """
        paths = []

        def dfs(current: str, target: str, path: List[CausalRelation], visited: Set[str], depth: int):
            if depth > max_hops:
                return

            if current == target:
                paths.append(path.copy())
                return

            visited.add(current)

            for relation in self.causal_graph.get(current, []):
                if relation.effect not in visited:
                    path.append(relation)
                    dfs(relation.effect, target, path, visited, depth + 1)
                    path.pop()

            visited.remove(current)

        dfs(start, end, [], set(), 0)
        return paths

    def summarize_causal_knowledge(self) -> str:
        """
        Summarize all causal relationships discovered.
        """
        lines = [
            "\n" + "="*60,
            "🔗 CAUSAL REASONING SUMMARY",
            "="*60
        ]

        total_relations = sum(len(rels) for rels in self.causal_graph.values())
        lines.append(f"\nCausal relations discovered: {total_relations}")

        # Sample strongest relations
        all_relations = []
        for rels in self.causal_graph.values():
            all_relations.extend(rels)

        all_relations.sort(key=lambda r: r.strength, reverse=True)

        if all_relations:
            lines.append("\nStrongest causal links:")
            for i, rel in enumerate(all_relations[:5], 1):
                lines.append(f"  {i}. {rel.cause} → {rel.effect}")
                lines.append(f"     Mechanism: {rel.mechanism}")
                lines.append(f"     Strength: {rel.strength:.2f}")

        # Counterfactuals
        if self.counterfactuals:
            lines.append(f"\nCounterfactuals explored: {len(self.counterfactuals)}")
            for i, cf in enumerate(self.counterfactuals[:3], 1):
                lines.append(f"  {i}. What if: {cf.intervention}")
                lines.append(f"     Outcome: {cf.predicted_outcome}")
                lines.append(f"     Confidence: {cf.confidence:.2f}")

        lines.append("="*60)
        return "\n".join(lines)
