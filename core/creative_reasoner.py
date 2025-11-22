"""
Creative Insight Generation

Generates novel hypotheses, finds hidden patterns, makes creative leaps.
This is what separates AGI from simple retrieval!

Capabilities:
- Analogy generation (X is like Y because...)
- Pattern recognition across domains
- Hypothesis generation
- Counter-intuitive insight discovery
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class CreativeInsight:
    """A creative insight or hypothesis."""
    type: str  # "analogy", "pattern", "hypothesis", "connection"
    content: str
    confidence: float
    novelty: float  # How unexpected/creative is this?
    domains: List[str]


class CreativeReasoner:
    """
    Generates creative insights and novel hypotheses.

    This is AGI's "aha!" moment generator!
    """

    def __init__(self, llm_bridge):
        self.llm = llm_bridge
        self.insights_generated: List[CreativeInsight] = []

    async def generate_analogies(
        self,
        concept: str,
        domain: str,
        known_concepts: List[str]
    ) -> List[CreativeInsight]:
        """
        Generate creative analogies to understand a concept.

        Example: "Prime numbers are like atoms - indivisible building blocks"
        """
        print(f"\n[🎨] CREATIVE REASONING: Generating analogies for '{concept}'...")

        prompt = f"""Generate creative analogies to explain "{concept}" in {domain}.

Known concepts you can reference: {', '.join(known_concepts[:20])}

Create 3-5 analogies that:
1. Connect to familiar concepts
2. Highlight key properties
3. Are memorable and insightful

Format each as:
ANALOGY: [concept] is like [familiar thing] because [reason]
NOVELTY: [0.0-1.0 how creative/unexpected]

Example for "prime numbers":
ANALOGY: Prime numbers are like atoms - indivisible building blocks of all numbers
NOVELTY: 0.7

ANALOGY: Prime numbers are like VIP guests - they only divide evenly with themselves and 1
NOVELTY: 0.6
"""

        response = self.llm.generate(
            system_prompt="You are creative and generate insightful analogies.",
            user_input=prompt
        )

        text = response.get("text", "")

        insights = []
        current_analogy = None
        current_novelty = 0.5

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("ANALOGY:"):
                if current_analogy:
                    # Save previous
                    insights.append(CreativeInsight(
                        type="analogy",
                        content=current_analogy,
                        confidence=0.8,
                        novelty=current_novelty,
                        domains=[domain]
                    ))

                current_analogy = line.split("ANALOGY:", 1)[1].strip()

            elif line.startswith("NOVELTY:"):
                try:
                    novelty_str = line.split("NOVELTY:", 1)[1].strip()
                    current_novelty = float(novelty_str.split()[0])
                except:
                    current_novelty = 0.5

        # Save last one
        if current_analogy:
            insights.append(CreativeInsight(
                type="analogy",
                content=current_analogy,
                confidence=0.8,
                novelty=current_novelty,
                domains=[domain]
            ))

        # Display
        for insight in insights:
            print(f"[💡] {insight.content}")
            print(f"     Novelty: {insight.novelty:.2f}")

        self.insights_generated.extend(insights)
        return insights

    async def find_hidden_patterns(
        self,
        concepts: List[str],
        domain: str
    ) -> List[CreativeInsight]:
        """
        Find hidden patterns across multiple concepts.

        Example: "All these concepts involve transformation: functions transform inputs,
                 equations transform variables, derivatives transform functions"
        """
        if len(concepts) < 3:
            return []

        print(f"\n[🎨] CREATIVE REASONING: Finding patterns across {len(concepts)} concepts...")

        prompt = f"""Analyze these concepts and find hidden patterns or connections:

Concepts: {', '.join(concepts[:15])}
Domain: {domain}

Find 2-3 deep patterns that connect multiple concepts. Look for:
- Common themes or principles
- Unexpected connections
- Meta-patterns (patterns about patterns)
- Unifying frameworks

Format:
PATTERN: [description]
CONNECTS: [concept1, concept2, ...]
NOVELTY: [0.0-1.0]

Example:
PATTERN: All involve hierarchical decomposition - breaking complex things into simpler parts
CONNECTS: division, factorization, prime numbers
NOVELTY: 0.6
"""

        response = self.llm.generate(
            system_prompt="You find deep patterns and connections that aren't obvious.",
            user_input=prompt
        )

        text = response.get("text", "")

        insights = []
        current_pattern = None
        current_connects = []
        current_novelty = 0.5

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("PATTERN:"):
                if current_pattern:
                    insights.append(CreativeInsight(
                        type="pattern",
                        content=current_pattern,
                        confidence=0.7,
                        novelty=current_novelty,
                        domains=[domain]
                    ))

                current_pattern = line.split("PATTERN:", 1)[1].strip()

            elif line.startswith("CONNECTS:"):
                connect_str = line.split("CONNECTS:", 1)[1].strip()
                current_connects = [c.strip() for c in connect_str.split(',')]

            elif line.startswith("NOVELTY:"):
                try:
                    novelty_str = line.split("NOVELTY:", 1)[1].strip()
                    current_novelty = float(novelty_str.split()[0])
                except:
                    current_novelty = 0.5

        # Save last one
        if current_pattern:
            insights.append(CreativeInsight(
                type="pattern",
                content=current_pattern,
                confidence=0.7,
                novelty=current_novelty,
                domains=[domain]
            ))

        for insight in insights:
            print(f"[🔍] Pattern found: {insight.content}")
            print(f"     Novelty: {insight.novelty:.2f}")

        self.insights_generated.extend(insights)
        return insights

    async def generate_hypotheses(
        self,
        goal: str,
        learned_concepts: List[str],
        missing_concepts: List[str]
    ) -> List[CreativeInsight]:
        """
        Generate hypotheses about how to achieve the goal.

        Example: "Hypothesis: We can solve this without learning X if we approach from Y angle"
        """
        print(f"\n[🎨] CREATIVE REASONING: Generating hypotheses...")

        prompt = f"""Generate creative hypotheses about how to achieve this goal:

Goal: {goal}
What we know: {', '.join(learned_concepts[:10])}
What we think we need: {', '.join(missing_concepts[:5])}

Generate 2-3 hypotheses about:
- Alternative approaches we haven't considered
- Shortcuts or simplifications
- Whether we REALLY need all those missing concepts
- Novel problem-solving strategies

Format:
HYPOTHESIS: [hypothesis]
CONFIDENCE: [0.0-1.0]
NOVELTY: [0.0-1.0]

Example:
HYPOTHESIS: We might not need category theory at all - the basics of division and factors are sufficient for understanding prime numbers
CONFIDENCE: 0.8
NOVELTY: 0.7
"""

        response = self.llm.generate(
            system_prompt="You generate bold, creative hypotheses that challenge assumptions.",
            user_input=prompt
        )

        text = response.get("text", "")

        insights = []
        current_hyp = None
        current_conf = 0.5
        current_nov = 0.5

        for line in text.split('\n'):
            line = line.strip()

            if line.startswith("HYPOTHESIS:"):
                if current_hyp:
                    insights.append(CreativeInsight(
                        type="hypothesis",
                        content=current_hyp,
                        confidence=current_conf,
                        novelty=current_nov,
                        domains=["general"]
                    ))

                current_hyp = line.split("HYPOTHESIS:", 1)[1].strip()

            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split("CONFIDENCE:", 1)[1].strip()
                    current_conf = float(conf_str.split()[0])
                except:
                    current_conf = 0.5

            elif line.startswith("NOVELTY:"):
                try:
                    nov_str = line.split("NOVELTY:", 1)[1].strip()
                    current_nov = float(nov_str.split()[0])
                except:
                    current_nov = 0.5

        # Save last one
        if current_hyp:
            insights.append(CreativeInsight(
                type="hypothesis",
                content=current_hyp,
                confidence=current_conf,
                novelty=current_nov,
                domains=["general"]
            ))

        for insight in insights:
            print(f"[💭] Hypothesis: {insight.content}")
            print(f"     Confidence: {insight.confidence:.2f}, Novelty: {insight.novelty:.2f}")

        self.insights_generated.extend(insights)
        return insights

    async def cross_domain_transfer(
        self,
        concept: str,
        source_domain: str,
        target_domain: str,
        known_concepts: Dict[str, List[str]]  # domain -> concepts
    ) -> Optional[CreativeInsight]:
        """
        Transfer knowledge from one domain to another.

        Example: "Sorting in programming is like organizing in real life - both need comparison criteria"
        """
        source_concepts = known_concepts.get(source_domain, [])
        target_concepts = known_concepts.get(target_domain, [])

        if not source_concepts:
            return None

        print(f"\n[🎨] CREATIVE REASONING: Cross-domain transfer {source_domain} → {target_domain}...")

        prompt = f"""Can we understand "{concept}" in {target_domain} by transferring knowledge from {source_domain}?

Known concepts in {source_domain}: {', '.join(source_concepts[:10])}
Known concepts in {target_domain}: {', '.join(target_concepts[:10])}

Find a creative transfer or analogy:

TRANSFER: [How concept in source domain maps to target domain]
CONFIDENCE: [0.0-1.0]
NOVELTY: [0.0-1.0]

Example:
TRANSFER: Sorting in programming is like alphabetizing books - both need a comparison rule to decide order
CONFIDENCE: 0.8
NOVELTY: 0.6
"""

        response = self.llm.generate(
            system_prompt="You creatively transfer knowledge across domains.",
            user_input=prompt
        )

        text = response.get("text", "")

        # Parse
        transfer = None
        confidence = 0.5
        novelty = 0.5

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("TRANSFER:"):
                transfer = line.split("TRANSFER:", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split("CONFIDENCE:", 1)[1].strip().split()[0])
                except:
                    pass
            elif line.startswith("NOVELTY:"):
                try:
                    novelty = float(line.split("NOVELTY:", 1)[1].strip().split()[0])
                except:
                    pass

        if transfer:
            insight = CreativeInsight(
                type="connection",
                content=transfer,
                confidence=confidence,
                novelty=novelty,
                domains=[source_domain, target_domain]
            )

            print(f"[🌉] Cross-domain insight: {transfer}")
            print(f"     Confidence: {confidence:.2f}, Novelty: {novelty:.2f}")

            self.insights_generated.append(insight)
            return insight

        return None

    def get_most_novel_insights(self, top_k: int = 5) -> List[CreativeInsight]:
        """
        Get the most creative/novel insights generated.
        """
        sorted_insights = sorted(
            self.insights_generated,
            key=lambda x: x.novelty,
            reverse=True
        )
        return sorted_insights[:top_k]

    def summarize_insights(self) -> str:
        """
        Create a summary of all creative insights generated.
        """
        if not self.insights_generated:
            return "No insights generated yet."

        lines = [
            "\n" + "="*60,
            "🎨 CREATIVE INSIGHTS SUMMARY",
            "="*60
        ]

        by_type = {}
        for insight in self.insights_generated:
            if insight.type not in by_type:
                by_type[insight.type] = []
            by_type[insight.type].append(insight)

        for itype, insights in by_type.items():
            lines.append(f"\n{itype.upper()}S ({len(insights)}):")
            for i, insight in enumerate(insights[:3], 1):  # Top 3 per type
                lines.append(f"  {i}. {insight.content}")
                lines.append(f"     Novelty: {insight.novelty:.2f}")

        avg_novelty = sum(i.novelty for i in self.insights_generated) / len(self.insights_generated)
        lines.append(f"\nTotal insights: {len(self.insights_generated)}")
        lines.append(f"Average novelty: {avg_novelty:.2f}")
        lines.append("="*60)

        return "\n".join(lines)
