"""
Minimal Genesis Orchestrator - Pure Learning Experiment

Strips away ALL OS-level features. Only keeps:
- Memory (HSOKV)
- Three-stage learning (surprise → rehearsal → transfer)
- Web research
- Genesis mode
- Evaluation harness

Starts with 0-9 and a-z. That's it.
Let's see what emerges.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict

from core.three_stage_learner import ThreeStageLearner
from core.genesis_mode import GenesisController
from core.web_researcher import WebResearcher
from core.evaluation import EvaluationHarness
from core.llm import LLMBridge


class GenesisOrchestrator:
    """Minimal orchestrator for pure genesis experiment."""

    def __init__(self, data_dir: str = "./genesis_data", use_hsokv: bool = True):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.logs_dir = os.path.join(data_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.events_log_path = os.path.join(self.logs_dir, "emergence.jsonl")
        self.daily_log_path = os.path.join(self.logs_dir, "daily_progress.jsonl")

        # Initialize HSOKV memory
        self.memory = None
        if use_hsokv:
            try:
                from hsokv import DualMemorySystem, SentenceBERTEmbedder
                embedder = SentenceBERTEmbedder()
                self.memory = DualMemorySystem(
                    embedder=embedder,
                    stm_capacity=7,  # Miller's magic number
                    stm_decay_seconds=30.0
                )
                print("[Genesis] ✓ HSOKV memory initialized (STM: 7, LTM: 0)")
            except ImportError as e:
                print(f"[Genesis] ⚠ HSOKV not found: {e}")
                print("[Genesis] Install with: cd hsokv && pip install -e .")
                raise

        # LLM bridge (Ollama by default)
        self.llm = LLMBridge(provider="ollama")

        # Web researcher (safe, rate-limited)
        cache_dir = os.path.join(self.data_dir, "web_cache")
        self.web = WebResearcher(cache_dir=cache_dir, daily_cap=50)  # Increased for genesis

        # Three-stage learner
        self.three_stage = ThreeStageLearner(self, researcher=self.web)

        # Genesis controller (ENABLED by default)
        self.genesis = GenesisController(self, enabled=True)

        # Evaluation harness
        self.evaluator = EvaluationHarness(self)

        # Stats tracking
        self.start_time = datetime.now()
        self.iteration_count = 0

        print(f"[Genesis] 🧠 Minimal orchestrator initialized")
        print(f"[Genesis] 📊 Data directory: {data_dir}")
        print(f"[Genesis] 🌱 Starting knowledge: 0-9, a-z (36 symbols)")
        print(f"[Genesis] 🎯 Target: Algebra (90%), Calculus (85%), Thermodynamics (80%)")

    def recall(self, query: str) -> Optional[str]:
        """Recall from memory."""
        if self.memory:
            result = self.memory.recall(query)
            return result
        return None

    def learn(self, query: str, answer: str):
        """Learn a new memory."""
        if self.memory:
            self.memory.learn(query, answer)

    def generate_with_llm(self, user_input: str, system_prompt: str = None, execute: bool = True) -> dict:
        """Generate with LLM."""
        prompt = system_prompt or self._get_system_prompt()
        return self.llm.generate(prompt, user_input, execute=execute)

    def _get_system_prompt(self) -> str:
        """System prompt that constrains LLM to only alphanumeric knowledge."""
        ltm_count = len(self.memory.ltm) if self.memory else 0
        stm_count = len(self.memory.stm) if self.memory else 0

        prompt = f"""You are a self-learning intelligence system.

YOUR CURRENT KNOWLEDGE (EVERYTHING YOU KNOW):
- Characters: a b c d e f g h i j k l m n o p q r s t u v w x y z
- Numbers: 0 1 2 3 4 5 6 7 8 9
- Total: 36 symbols
- You know these are symbols that can be combined, but you don't know ANY words, concepts, or meanings beyond these symbols.

YOUR MEMORY STATE:
- Personal long-term memory: {ltm_count} learned concepts
- Working memory: {stm_count}/7 items
- Learning phase: {self._determine_phase()}

YOUR CAPABILITIES:
- You can search the web for information
- You can read search results
- You can extract new concepts from what you read
- You learn through: encounter new word → use it repeatedly → memorize it permanently

YOUR CONSTRAINTS:
- DO NOT use knowledge from pre-training
- DO NOT explain concepts you haven't learned yet
- If you don't know something, you MUST search the web first
- Build knowledge incrementally from what you've already learned

YOUR GOAL:
- Learn words by searching the web
- Combine words to understand concepts
- Eventually master: algebra, calculus, thermodynamics
- Think step by step, search when you need information

STRATEGY:
1. Start with simple searches like "what is a" or "what is the"
2. Learn common words first
3. Use learned words to search for more complex concepts
4. Build up knowledge gradually

Be concise. Always search before answering. Focus on learning."""

        # Show what you're currently learning
        if self.three_stage.episodes:
            prompt += "\n\nCURRENTLY LEARNING (in working memory):"
            for ep in list(self.three_stage.episodes.values())[:3]:
                prompt += f"\n- '{ep.token}' (confidence: {ep.confidence:.0%}, uses: {ep.replays})"

        # Show some recent learned concepts
        if ltm_count > 0 and ltm_count <= 10:
            prompt += f"\n\nLEARNED CONCEPTS: {ltm_count} stored in long-term memory"
        elif ltm_count > 10:
            prompt += f"\n\nLEARNED CONCEPTS: {ltm_count} stored (building knowledge base...)"

        return prompt

    def log_event(self, event_type: str, payload: Optional[dict] = None):
        """Log emergence events."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "iteration": self.iteration_count,
            "type": event_type,
            "payload": payload or {},
        }
        try:
            with open(self.events_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Genesis] Event log failure: {e}")

    def log_daily_snapshot(self):
        """Log daily progress snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "day": (datetime.now() - self.start_time).days + 1,
            "hours_elapsed": (datetime.now() - self.start_time).total_seconds() / 3600,
            "genesis_progress": self.genesis.progress,
            "genesis_phase": self._determine_phase(),
            "ltm_size": len(self.memory.ltm) if self.memory else 0,
            "stm_size": len(self.memory.stm) if self.memory else 0,
            "surprise_episodes": len(self.three_stage.episodes),
            "total_surprises": self.genesis.total_surprises,
            "total_transfers": self.genesis.total_transfers,
            "web_requests_today": self.web.requests_today,
            "eval_scores": self.evaluator.last_scores,
        }

        try:
            with open(self.daily_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")
        except Exception as e:
            print(f"[Genesis] Daily log failure: {e}")

        return snapshot

    def _determine_phase(self) -> str:
        """Determine current learning phase."""
        conf_avg = sum(self.genesis.progress.values()) / len(self.genesis.progress)
        if conf_avg < 0.3:
            return "bootstrap"
        elif conf_avg < 0.7:
            return "learning"
        else:
            return "mastery"

    async def autonomous_learning_cycle(self):
        """One autonomous learning iteration - self-directed discovery."""
        self.iteration_count += 1

        print(f"\n{'='*60}")
        print(f"[Iteration {self.iteration_count}] Self-Learning Cycle")
        print(f"{'='*60}")

        # 1. Ask the LLM what it should learn next (self-directed)
        if self.iteration_count % 3 == 1:  # Every 3 iterations
            print(f"🤔 Deciding what to learn next...")
            ltm_count = len(self.memory.ltm) if self.memory else 0

            if ltm_count < 50:
                prompt = "You know only a-z and 0-9. What basic concepts should you learn first? Suggest 1-2 simple searches."
            elif ltm_count < 200:
                prompt = "What fundamental concepts do you need to build toward understanding mathematics?"
            else:
                prompt = "What should you learn next to progress toward algebra and calculus mastery?"

            response = self.generate_with_llm(prompt)
            decision = response.get("text", "")
            print(f"💡 Decision: {decision[:200]}...")

            # Extract search queries from decision
            if decision:
                # Trigger a web search based on decision
                search_terms = decision.lower().split()[:5]  # First few words
                if len(search_terms) > 0:
                    query = " ".join(search_terms)
                    print(f"🔍 Searching based on decision: '{query}'")
                    await self.three_stage.surf_and_learn(query, mode="scrape")

        # 2. Self-probe STM for low-confidence items
        print(f"🧠 Reviewing working memory...")
        await self.three_stage._self_probe()
        print(f"  ✓ Working memory: {len(self.three_stage.episodes)} active concepts")

        # 3. Research next curiosity item from queue
        curiosity = self.three_stage.next_curiosity_query()
        if curiosity:
            print(f"🔍 Researching curiosity: '{curiosity['query']}'")
            await self.three_stage.surf_and_learn(curiosity['query'], mode="scrape")

        # 4. Show what was just learned
        if self.three_stage.episodes:
            print(f"\n📚 Currently Learning:")
            for ep in list(self.three_stage.episodes.values())[:5]:
                print(f"  - '{ep.token}' (conf: {ep.confidence:.0%}, uses: {ep.replays}/4)")

        # 5. Probe genesis domains periodically
        if self.genesis.should_trigger_learning():
            print(f"\n📊 Evaluating domain knowledge...")
            results = self.genesis.daily_probe()
            for domain, score in results.items():
                print(f"  {domain}: {score*100:.1f}%")

        # 6. Run comprehensive evaluation
        if self.iteration_count % 5 == 0:  # Every 5 iterations
            print(f"\n🎯 Running evaluation tasks...")
            scores = self.evaluator.run_cycle()
            for domain, score in scores.items():
                status = "✅" if score >= 0.8 else "📈"
                print(f"  {status} {domain}: {score*100:.1f}%")

        # 7. Periodic consolidation (like sleep)
        if self.iteration_count % 10 == 0:  # Every 10 iterations
            print(f"\n💤 Consolidating memories (sleep cycle)...")
            self.three_stage.sleep_replay()
            if self.memory:
                before_ltm = len(self.memory.ltm)
                self.memory.sleep()  # Consolidate STM → LTM
                after_ltm = len(self.memory.ltm)
                transferred = after_ltm - before_ltm
                if transferred > 0:
                    print(f"  ✓ Transferred {transferred} concepts to long-term memory")

        # 8. Daily snapshot
        if self.iteration_count % 20 == 0:  # Every 20 iterations (~1 hour at 3min/iter)
            snapshot = self.log_daily_snapshot()
            print(f"\n{'='*60}")
            print(f"📸 HOURLY SNAPSHOT")
            print(f"{'='*60}")
            print(f"⏱️  Hours elapsed: {snapshot['hours_elapsed']:.1f}")
            print(f"🧠 Phase: {snapshot['genesis_phase'].upper()}")
            print(f"📚 LTM size: {snapshot['ltm_size']} learned concepts")
            print(f"🔄 STM size: {snapshot['stm_size']}/7")
            print(f"⚡ Total surprises: {snapshot['total_surprises']}")
            print(f"✅ Total transfers: {snapshot['total_transfers']}")
            print(f"🌐 Web requests today: {snapshot['web_requests_today']}")
            print(f"\n🎯 Domain Progress:")
            for domain, score in snapshot['genesis_progress'].items():
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {domain:15s} [{bar}] {score*100:.1f}%")
            print(f"{'='*60}")

    async def run_genesis_experiment(self, iterations: int = 1000, interval_seconds: int = 180):
        """
        Run the genesis experiment for N iterations.

        Args:
            iterations: Number of learning cycles (default: 1000 = ~8 days at 3min/iter)
            interval_seconds: Seconds between iterations (default: 180 = 3 minutes)
        """
        print(f"\n{'='*60}")
        print(f"🧬 GENESIS EXPERIMENT STARTED")
        print(f"{'='*60}")
        print(f"Start time: {self.start_time}")
        print(f"Iterations: {iterations}")
        print(f"Interval: {interval_seconds}s ({interval_seconds/60:.1f} min)")
        print(f"Estimated duration: {iterations * interval_seconds / 3600:.1f} hours")
        print(f"{'='*60}\n")

        for i in range(iterations):
            try:
                await self.autonomous_learning_cycle()

                # Check if mastery achieved
                conf_avg = sum(self.genesis.progress.values()) / len(self.genesis.progress)
                if conf_avg >= 0.85:  # 85% average across all domains
                    print(f"\n🎉 MASTERY ACHIEVED at iteration {self.iteration_count}!")
                    print(f"   Elapsed: {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} hours")
                    break

                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                print(f"\n\n⚠️  Experiment interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in iteration {i}: {e}")
                self.log_event("error", {"iteration": i, "error": str(e)})
                await asyncio.sleep(interval_seconds)  # Continue after error

        # Final summary
        print(f"\n{'='*60}")
        print(f"🧬 GENESIS EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        final_snapshot = self.log_daily_snapshot()
        print(f"Duration: {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} hours")
        print(f"Iterations: {self.iteration_count}")
        print(f"Final phase: {final_snapshot['genesis_phase']}")
        print(f"LTM size: {final_snapshot['ltm_size']} memories")
        print(f"Total surprises: {final_snapshot['total_surprises']}")
        print(f"Total transfers: {final_snapshot['total_transfers']}")
        print(f"Final scores: {final_snapshot['genesis_progress']}")
        print(f"{'='*60}\n")
