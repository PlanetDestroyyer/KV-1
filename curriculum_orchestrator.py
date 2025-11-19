"""
Curriculum-based orchestrator for guided self-discovery learning.

Uses LearningCurriculum to systematically progress through:
Language -> Numbers -> Algebra -> Calculus -> Thermodynamics
"""

import asyncio
import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add hsokv to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsokv'))

from core.learning_curriculum import LearningCurriculum, Concept
from core.llm import LLMBridge
from core.three_stage_learner import ThreeStageLearner
from core.web_researcher import WebResearcher

try:
    from hsokv import DualMemorySystem, SentenceBERTEmbedder
    HSOKV_AVAILABLE = True
except ImportError:
    HSOKV_AVAILABLE = False
    print("[Warning] HSOKV not available. Install with: cd hsokv && pip install -e .")
    # Create simple mock memory for basic functionality
    class MockMemory:
        def __init__(self, **kwargs):
            self.stm = []
            self.ltm = []
        def learn(self, word: str, definition: str, **kwargs):
            self.ltm.append((word, definition))
    DualMemorySystem = MockMemory
    SentenceBERTEmbedder = None


class CurriculumOrchestrator:
    """Orchestrator that guides learning through a structured curriculum."""

    def __init__(self, data_dir: str = "./curriculum_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Initialize curriculum
        self.curriculum = LearningCurriculum()

        # Initialize LLM
        self.llm = LLMBridge(provider="ollama", default_model="gemma3:4b")

        # Initialize memory with HSOKV
        if HSOKV_AVAILABLE and SentenceBERTEmbedder:
            embedder = SentenceBERTEmbedder()
            self.memory = DualMemorySystem(
                embedder=embedder,
                stm_capacity=7,  # Miller's magic number
                stm_decay_seconds=30.0
            )
            print("[+] HSOKV memory initialized (STM: 7, LTM: 0)")
        else:
            # Use mock memory
            self.memory = DualMemorySystem()
            print("[!] Using mock memory (HSOKV not available)")

        # Initialize web researcher (before three_stage needs it)
        self.web_researcher = WebResearcher(
            cache_dir=os.path.join(data_dir, "web_cache"),
            daily_cap=50,  # Allow more requests for curriculum learning
        )

        # Initialize three-stage learner
        self.three_stage = ThreeStageLearner(
            self,
            researcher=self.web_researcher
        )

        self.iteration_count = 0
        self.events_log = []

    def generate_with_llm(self, user_input: str, system_prompt: str = None, execute: bool = True) -> dict:
        """Generate with LLM using curriculum-aware system prompt."""
        prompt = system_prompt or self._get_system_prompt()
        return self.llm.generate(prompt, user_input, execute=execute)

    def _get_system_prompt(self) -> str:
        """System prompt that guides curriculum-based learning."""
        phase = self.curriculum.current_phase
        if not phase:
            return "You are a self-learning AI that has mastered the curriculum."

        learned = list(self.curriculum.learned_concepts)
        progress = self.curriculum.get_progress()

        prompt = f"""You are a self-learning intelligence system progressing through a structured curriculum.

CURRENT LEARNING PHASE: {phase.name}
Phase Description: {phase.description}

CONCEPTS YOU'VE MASTERED: {', '.join(learned) if learned else 'None yet'}

PHASE PROGRESS:
"""
        for phase_name, prog in progress.items():
            bar = '�' * int(prog * 10) + '�' * (10 - int(prog * 10))
            prompt += f"  {phase_name}: [{bar}] {prog*100:.0f}%\n"

        prompt += """
YOUR TASK:
- Learn concepts systematically, one at a time
- When explaining a concept, use what you've already learned
- Build knowledge incrementally
- Focus on understanding, not memorization

When asked to explain a concept, provide:
1. Clear definition using plain language
2. Examples that illustrate the concept
3. How it relates to what you've already learned
"""
        return prompt

    async def learn_concept(self, concept: Concept) -> bool:
        """Learn a single concept through web research and LLM explanation."""
        print(f"\n=� Learning: {concept.name}")
        print(f"   Query: {concept.search_query}")

        # Web research
        research_result = self.web_researcher.fetch(concept.search_query, mode="scrape")
        if not research_result or not research_result.text:
            print(f"   �  No web content found")
            return False

        # Extract clean snippet
        snippet = research_result.text[:2000]  # First 2000 chars
        print(f"    Retrieved {len(research_result.text)} chars from web")

        # Feed to 3-stage learner as surprise
        self.three_stage.on_surprise(
            concept.name,
            {
                "source": "curriculum",
                "description": concept.description,
                "web_content": snippet,
                "keywords": concept.keywords,
            }
        )

        # Ask LLM to explain the concept
        prompt = f"""Based on this information about '{concept.name}':

{snippet}

Please explain what '{concept.name}' means in 2-3 clear sentences. Focus on the core idea."""

        response = self.generate_with_llm(prompt)
        explanation = response.get("text", "")

        if explanation:
            print(f"   =� Explanation: {explanation[:200]}...")

            # Verify understanding using keywords
            understood = self.curriculum.verify_concept(concept.name, explanation)

            if understood:
                self.curriculum.mark_learned(concept.name)
                print(f"    Concept mastered!")

                # Store explanation in memory
                self.memory.learn(concept.name, explanation[:500])
                return True
            else:
                print(f"   �  Understanding incomplete (missing key concepts)")
                return False
        else:
            print(f"   L Failed to generate explanation")
            return False

    async def learning_cycle(self):
        """Execute one curriculum-based learning cycle."""
        self.iteration_count += 1

        print(f"\n{'='*60}")
        print(f"[Iteration {self.iteration_count}] Curriculum Learning Cycle")
        print(f"{'='*60}")

        # Get next concept to learn
        concept = self.curriculum.next_concept_to_learn()

        if not concept:
            print("<� Curriculum complete!")
            return False  # No more concepts to learn

        # Learn the concept
        learned = await self.learn_concept(concept)

        # Trigger 3-stage learning processes
        print(f"\n>� Processing working memory...")
        await self.three_stage._self_probe()
        print(f"    STM: {len(self.three_stage.episodes)} active episodes")

        # Periodic consolidation
        if self.iteration_count % 5 == 0:
            print(f"\n=� Consolidating memories...")
            await self.three_stage._consolidate()

        # Show progress
        print(f"\n=� Curriculum Progress:")
        summary = self.curriculum.summary()
        print(f"   Phase: {summary['current_phase']}")
        print(f"   Learned: {summary['learned_concepts']}/{summary['total_concepts']} concepts")

        for phase_name, prog in summary['progress'].items():
            bar = '�' * int(prog * 20) + '�' * (20 - int(prog * 20))
            print(f"   {phase_name:25} [{bar}] {prog*100:.0f}%")

        return True  # Continue learning

    def log_event(self, event_type: str, data: dict):
        """Log an event."""
        self.events_log.append({"type": event_type, "data": data})


async def main_curriculum_experiment(iterations: int = 30):
    """Run curriculum-based learning experiment."""
    print("="*60)
    print("<� CURRICULUM-BASED LEARNING EXPERIMENT")
    print("="*60)
    print(f"Iterations: {iterations}")
    print("Curriculum: Language � Numbers � Algebra � Calculus � Thermodynamics")
    print("="*60)

    orchestrator = CurriculumOrchestrator()

    for i in range(iterations):
        should_continue = await orchestrator.learning_cycle()

        if not should_continue:
            print("\n Curriculum complete!")
            break

        # Short delay between iterations
        await asyncio.sleep(2)

    # Final summary
    print("\n" + "="*60)
    print("=� FINAL SUMMARY")
    print("="*60)
    summary = orchestrator.curriculum.summary()
    print(f"Learned Concepts: {summary['learned_concepts']}/{summary['total_concepts']}")
    print(f"LTM Size: {len(orchestrator.memory.ltm) if orchestrator.memory else 0}")
    print(f"Learned: {', '.join(sorted(orchestrator.curriculum.learned_concepts))}")
    print("="*60)


if __name__ == "__main__":
    import sys
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(main_curriculum_experiment(iterations))
