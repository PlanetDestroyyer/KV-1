"""
Self-Discovery Learning Orchestrator

Goal-driven autonomous learning system that discovers knowledge through need.
Unlike curriculum-based learning, this system:
- Starts with a goal
- Identifies what it doesn't know by attempting and failing
- Recursively learns prerequisites
- Saves learned knowledge to persistent LTM
- Shows the discovery journey (like human history)
"""

import asyncio
import json
import os
import sys
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsokv'))

from core.llm import LLMBridge
from core.web_researcher import WebResearcher


@dataclass
class LearningEntry:
    """A single entry in the learning journal."""
    concept: str
    definition: str
    learned_at: str
    needed_for: str
    source: str  # "web", "primitive", "inference"


@dataclass
class GoalAttempt:
    """Result of attempting a goal."""
    success: bool
    result: str
    missing_concepts: List[str]
    error_message: Optional[str] = None


class PersistentLTM:
    """Persistent long-term memory with disk storage."""

    def __init__(self, storage_path: str = "./ltm_memory.json"):
        self.storage_path = storage_path
        self.knowledge: Dict[str, LearningEntry] = {}
        self.load()

    def load(self):
        """Load LTM from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge = {
                        k: LearningEntry(**v) for k, v in data.items()
                    }
                print(f"[+] Loaded {len(self.knowledge)} concepts from LTM")
            except Exception as e:
                print(f"[!] Failed to load LTM: {e}")
                self.knowledge = {}
        else:
            print("[i] No existing LTM found, starting fresh")
            self.knowledge = {}

    def save(self):
        """Save LTM to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                data = {k: asdict(v) for k, v in self.knowledge.items()}
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[+] Saved {len(self.knowledge)} concepts to LTM")
        except Exception as e:
            print(f"[!] Failed to save LTM: {e}")

    def has(self, concept: str) -> bool:
        """Check if concept is known."""
        return concept.lower() in self.knowledge

    def get(self, concept: str) -> Optional[LearningEntry]:
        """Retrieve concept from memory."""
        return self.knowledge.get(concept.lower())

    def add(self, entry: LearningEntry):
        """Add new concept to memory."""
        self.knowledge[entry.concept.lower()] = entry
        self.save()

    def get_all_concepts(self) -> List[str]:
        """Get list of all known concepts."""
        return list(self.knowledge.keys())


class SelfDiscoveryOrchestrator:
    """
    Autonomous goal-driven learning system.

    Pursues goals by:
    1. Attempting goal with current knowledge
    2. Identifying missing concepts from failures
    3. Recursively learning prerequisites
    4. Retrying until success
    """

    def __init__(
        self,
        goal: str,
        ltm_path: str = "./ltm_memory.json",
        data_dir: str = "./self_discovery_data",
        max_depth: int = 10
    ):
        self.goal = goal
        self.max_learning_depth = max_depth
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Initialize components
        self.ltm = PersistentLTM(ltm_path)
        self.llm = LLMBridge(provider="ollama", default_model="gemma3:4b")
        self.web_researcher = WebResearcher(
            cache_dir=os.path.join(data_dir, "web_cache"),
            daily_cap=100
        )

        # Learning journal
        self.journal: List[Dict] = []
        self.current_depth = 0
        self.attempts = 0

    def _get_knowledge_summary(self) -> str:
        """Generate summary of current knowledge for LLM context."""
        if not self.ltm.knowledge:
            return "You have no prior knowledge. You are starting from zero."

        concepts = self.ltm.get_all_concepts()
        summary = f"You currently know these {len(concepts)} concepts:\n"
        for concept in sorted(concepts)[:20]:  # Limit to avoid context overflow
            entry = self.ltm.get(concept)
            summary += f"- {concept}: {entry.definition[:100]}...\n"

        if len(concepts) > 20:
            summary += f"... and {len(concepts) - 20} more concepts\n"

        return summary

    async def attempt_goal(self) -> GoalAttempt:
        """
        Attempt to achieve the goal with current knowledge.
        Returns success status and what concepts are missing.
        """
        self.attempts += 1
        print(f"\n{'='*60}")
        print(f"[Attempt {self.attempts}] Trying to achieve goal")
        print(f"{'='*60}")
        print(f"Goal: {self.goal}")
        print(f"Known concepts: {len(self.ltm.knowledge)}")

        # Build prompt with current knowledge
        knowledge_summary = self._get_knowledge_summary()

        prompt = f"""You are an AI attempting to achieve a goal using only what you know.

CURRENT KNOWLEDGE:
{knowledge_summary}

GOAL: {self.goal}

INSTRUCTIONS:
1. Try to achieve the goal using ONLY the concepts you know
2. If you can complete it, provide the answer
3. If you cannot, identify EXACTLY what concepts you don't understand

Respond in this format:
SUCCESS: [yes/no]
ANSWER: [your answer if successful, or "cannot complete" if not]
MISSING: [comma-separated list of concepts you need to learn, or "none" if successful]
REASONING: [brief explanation]"""

        response = self.llm.generate(
            system_prompt="You are a self-learning AI that honestly assesses what you know and don't know.",
            user_input=prompt
        )

        text = response.get("text", "")
        print(f"\n[i] Response:\n{text}\n")

        # Parse response
        lines = text.split('\n')
        success = False
        result = ""
        missing = []

        for line in lines:
            if line.startswith("SUCCESS:"):
                success = "yes" in line.lower()
            elif line.startswith("ANSWER:"):
                result = line.split(":", 1)[1].strip()
            elif line.startswith("MISSING:"):
                missing_str = line.split(":", 1)[1].strip()
                if missing_str.lower() not in ["none", "n/a", ""]:
                    missing = [m.strip() for m in missing_str.split(",")]

        return GoalAttempt(
            success=success,
            result=result,
            missing_concepts=missing
        )

    def _is_valid_concept(self, concept: str) -> bool:
        """Check if concept is valid and worth learning."""
        if not concept or len(concept) < 2:
            return False

        # Skip if too long (probably garbage)
        if len(concept) > 60:
            return False

        concept_lower = concept.lower().strip()

        # Skip meta-concepts
        skip_patterns = [
            "none",
            "n/a",
            "null",
            "undefined",
            "this definition",
            "prior knowledge",
            "general understanding",
            "foundational",
            "doesn't require",
            "beyond that",
            ")",  # Truncated prerequisite
            "(",  # Truncated prerequisite
        ]

        for pattern in skip_patterns:
            if pattern in concept_lower:
                return False

        return True

    async def discover_concept(self, concept: str, needed_for: str) -> bool:
        """
        Autonomously discover and learn a concept.
        Returns True if successfully learned.
        """
        # Validate concept first
        if not self._is_valid_concept(concept):
            print(f"[!] Skipping invalid concept: {concept}")
            return False

        self.current_depth += 1

        if self.current_depth > self.max_learning_depth:
            print(f"[!] Max learning depth reached, stopping at {concept}")
            self.current_depth -= 1
            return False

        indent = "  " * self.current_depth
        print(f"\n{indent}[L] Discovering: {concept}")
        print(f"{indent}    Needed for: {needed_for}")

        # Check if already known
        if self.ltm.has(concept):
            print(f"{indent}    [i] Already in LTM, skipping")
            self.current_depth -= 1
            return True

        # Check if it's a primitive concept (letters, numbers)
        primitive = self._try_primitive_learning(concept)
        if primitive:
            print(f"{indent}    [OK] Learned as primitive")
            self.journal.append({
                "type": "primitive_learned",
                "concept": concept,
                "needed_for": needed_for,
                "depth": self.current_depth
            })
            self.current_depth -= 1
            return True

        # Search web for concept
        print(f"{indent}    [i] Searching web...")
        search_query = f"what is {concept}"
        result = self.web_researcher.fetch(search_query, mode="scrape")

        if not result or not result.text:
            print(f"{indent}    [X] No web content found")
            self.current_depth -= 1
            return False

        web_content = result.text[:2000]
        print(f"{indent}    [+] Retrieved {len(result.text)} chars from web")

        # Ask LLM to understand and explain
        understanding_prompt = f"""Based on this information about '{concept}':

{web_content}

Please provide:
1. A concise definition (1-2 sentences)
2. What concrete prerequisite concepts are needed

Format:
DEFINITION: [your definition]
PREREQUISITES: [comma-separated list of concrete concepts, or "none"]

IMPORTANT:
- Only list actual concepts (like "addition", "variables", "numbers")
- Do NOT include meta-concepts like "basic understanding" or "familiarity"
- Do NOT include explanations or parenthetical notes
- Keep each prerequisite to 1-3 words maximum
"""

        response = self.llm.generate(
            system_prompt="You are learning a new concept. Extract the key definition and identify only concrete prerequisite concepts, not meta-descriptions.",
            user_input=understanding_prompt
        )

        text = response.get("text", "")

        # Parse response
        definition = ""
        prerequisites = []

        for line in text.split('\n'):
            if line.startswith("DEFINITION:"):
                definition = line.split(":", 1)[1].strip()
            elif line.startswith("PREREQUISITES:"):
                prereq_str = line.split(":", 1)[1].strip()
                if prereq_str.lower() not in ["none", "n/a", ""]:
                    # Split and clean prerequisites
                    raw_prereqs = [p.strip() for p in prereq_str.split(",")]
                    # Filter out invalid concepts
                    prerequisites = [p for p in raw_prereqs if self._is_valid_concept(p)]

        if not definition:
            # Fallback: use first substantial line
            definition = text.strip().split('\n')[0]

        print(f"{indent}    [i] Definition: {definition[:100]}...")

        # Recursively learn prerequisites
        if prerequisites:
            # Show only valid prerequisites
            valid_prereqs = prerequisites[:3]  # Limit to top 3
            print(f"{indent}    [i] Prerequisites: {', '.join(valid_prereqs)}")
            for prereq in valid_prereqs:
                if not self.ltm.has(prereq):
                    await self.discover_concept(prereq, needed_for=concept)

        # Store in LTM
        entry = LearningEntry(
            concept=concept,
            definition=definition,
            learned_at=datetime.now().isoformat(),
            needed_for=needed_for,
            source="web"
        )

        self.ltm.add(entry)
        print(f"{indent}    [OK] Stored in LTM")

        # Log in journal
        self.journal.append({
            "type": "concept_learned",
            "concept": concept,
            "definition": definition,
            "needed_for": needed_for,
            "prerequisites": prerequisites,
            "depth": self.current_depth
        })

        self.current_depth -= 1
        return True

    def _try_primitive_learning(self, concept: str) -> bool:
        """
        Try to learn concept as a primitive (A-Z, 0-9, basic symbols).
        Returns True if it's a primitive concept.
        """
        concept_lower = concept.lower()

        # Single letters
        if len(concept) == 1 and concept.isalpha():
            definition = f"The letter '{concept.upper()}', a basic symbol of written language"
            entry = LearningEntry(
                concept=concept_lower,
                definition=definition,
                learned_at=datetime.now().isoformat(),
                needed_for="primitives",
                source="primitive"
            )
            self.ltm.add(entry)
            return True

        # Single digits
        if len(concept) == 1 and concept.isdigit():
            definition = f"The number {concept}, representing quantity"
            entry = LearningEntry(
                concept=concept_lower,
                definition=definition,
                learned_at=datetime.now().isoformat(),
                needed_for="primitives",
                source="primitive"
            )
            self.ltm.add(entry)
            return True

        # Basic concepts
        primitives = {
            "alphabet": "The letters A-Z used in written English",
            "numbers": "The digits 0-9 used to represent quantities",
            "letter": "A symbol representing a sound in written language",
            "digit": "A symbol representing a number (0-9)",
        }

        if concept_lower in primitives:
            entry = LearningEntry(
                concept=concept_lower,
                definition=primitives[concept_lower],
                learned_at=datetime.now().isoformat(),
                needed_for="primitives",
                source="primitive"
            )
            self.ltm.add(entry)
            return True

        return False

    async def pursue_goal(self, max_attempts: int = 10) -> bool:
        """
        Autonomously pursue the goal through self-discovery.
        Returns True if goal achieved.
        """
        print("\n" + "="*60)
        print("[G] SELF-DISCOVERY LEARNING")
        print("="*60)
        print(f"Goal: {self.goal}")
        print(f"Starting LTM size: {len(self.ltm.knowledge)} concepts")
        print("="*60)

        for attempt_num in range(max_attempts):
            # Attempt goal
            attempt = await self.attempt_goal()

            if attempt.success:
                print("\n" + "="*60)
                print("[OK] GOAL ACHIEVED!")
                print("="*60)
                print(f"Answer: {attempt.result}")
                print(f"Attempts: {self.attempts}")
                print(f"Concepts learned: {len(self.journal)}")
                return True

            # Goal failed - learn missing concepts
            if not attempt.missing_concepts:
                print("\n[!] Goal failed but no missing concepts identified")
                print("[!] LLM may not understand the goal or is confused")
                return False

            print(f"\n[i] Missing concepts: {', '.join(attempt.missing_concepts)}")

            # Learn each missing concept
            for concept in attempt.missing_concepts:
                learned = await self.discover_concept(concept, needed_for=self.goal)
                if not learned:
                    print(f"[!] Failed to learn {concept}, continuing anyway...")

            # Brief delay before retry
            await asyncio.sleep(1)

        print("\n[X] Max attempts reached without achieving goal")
        return False

    def print_learning_journal(self):
        """Print the learning journey."""
        print("\n" + "="*60)
        print("[#] LEARNING JOURNAL - Discovery Path")
        print("="*60)

        if not self.journal:
            print("No learning occurred")
            return

        for i, entry in enumerate(self.journal, 1):
            if entry["type"] == "concept_learned":
                indent = "  " * entry.get("depth", 0)
                print(f"\n{i}. {indent}Learned: {entry['concept']}")
                print(f"   {indent}Needed for: {entry['needed_for']}")
                print(f"   {indent}Definition: {entry['definition'][:80]}...")
                if entry.get("prerequisites"):
                    print(f"   {indent}Prerequisites: {', '.join(entry['prerequisites'])}")
            elif entry["type"] == "primitive_learned":
                indent = "  " * entry.get("depth", 0)
                print(f"\n{i}. {indent}Learned primitive: {entry['concept']}")
                print(f"   {indent}Needed for: {entry['needed_for']}")

        print("\n" + "="*60)
        print(f"Total concepts learned: {len(self.journal)}")
        print(f"Final LTM size: {len(self.ltm.knowledge)}")
        print("="*60)


async def main_self_discovery(goal: str, ltm_path: str = "./ltm_memory.json"):
    """Run self-discovery learning experiment."""
    orchestrator = SelfDiscoveryOrchestrator(
        goal=goal,
        ltm_path=ltm_path
    )

    success = await orchestrator.pursue_goal(max_attempts=10)

    # Print learning journal
    orchestrator.print_learning_journal()

    # Print final LTM summary
    print("\n" + "="*60)
    print("[#] LONG-TERM MEMORY")
    print("="*60)

    concepts = sorted(orchestrator.ltm.get_all_concepts())
    for concept in concepts:
        entry = orchestrator.ltm.get(concept)
        print(f"\n{concept}:")
        print(f"  {entry.definition}")
        print(f"  (learned for: {entry.needed_for})")

    print("\n" + "="*60)
    print(f"[{'OK' if success else 'X'}] Goal {'achieved' if success else 'not achieved'}")
    print(f"LTM saved to: {orchestrator.ltm.storage_path}")
    print("="*60)

    return success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python self_discovery_orchestrator.py '<goal>'")
        print("\nExample goals:")
        print("  'Count from 1 to 5'")
        print("  'Solve 2x + 5 = 15'")
        print("  'Calculate area of circle with radius 3'")
        print("  'Explain why ice floats'")
        sys.exit(1)

    goal = sys.argv[1]
    asyncio.run(main_self_discovery(goal))
