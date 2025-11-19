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
        """Minimal system prompt for genesis mode."""
        ltm_count = len(self.memory.ltm) if self.memory else 0
        stm_count = len(self.memory.stm) if self.memory else 0

        prompt = f"""You are a learning system in Genesis Mode.

CURRENT STATE:
- Long-term memories: {ltm_count}
- Working memory: {stm_count}/7
- Genesis phase: {self.genesis.progress}

YOUR CONSTRAINTS:
- You started knowing ONLY: 0-9, a-z (36 symbols)
- Everything else must be learned through web research and reasoning
- You learn by: surprise → rehearsal → consolidation

YOUR GOAL:
- Master algebra, calculus, thermodynamics from first principles
- Build knowledge incrementally, step by step
- Ask clarifying questions when uncertain

Be concise. Focus on learning."""

        # Inject active surprise episodes
        if self.three_stage.episodes:
            prompt += "\n\nACTIVE LEARNING:"
            for ep in list(self.three_stage.episodes.values())[:3]:
                prompt += f"\n- {ep.to_prompt_snippet()}"

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
        """One autonomous learning iteration."""
        self.iteration_count += 1

        print(f"\n[Iteration {self.iteration_count}] Starting learning cycle...")

        # 1. Self-probe STM for low-confidence items
        await self.three_stage._self_probe()
        print(f"  ✓ Self-probe complete (STM: {len(self.three_stage.episodes)})")

        # 2. Research next curiosity item
        curiosity = self.three_stage.next_curiosity_query()
        if curiosity:
            print(f"  🔍 Researching: {curiosity['query']}")
            await self.three_stage.surf_and_learn(curiosity['query'], mode="scrape")

        # 3. Probe genesis domains
        if self.genesis.should_trigger_learning():
            print(f"  📊 Running domain evaluation...")
            results = self.genesis.daily_probe()
            print(f"  📈 Scores: {results}")

        # 4. Run evaluation cycle
        if self.iteration_count % 5 == 0:  # Every 5 iterations
            print(f"  🎯 Evaluation cycle...")
            scores = self.evaluator.run_cycle()
            print(f"  📊 Eval: {scores}")

        # 5. Periodic consolidation (like sleep)
        if self.iteration_count % 10 == 0:  # Every 10 iterations
            print(f"  💤 Consolidating memories...")
            self.three_stage.sleep_replay()
            if self.memory:
                self.memory.sleep()  # Consolidate STM → LTM

        # 6. Daily snapshot
        if self.iteration_count % 20 == 0:  # Every 20 iterations (~1 hour at 3min/iter)
            snapshot = self.log_daily_snapshot()
            print(f"\n📸 Daily Snapshot:")
            print(f"   Phase: {snapshot['genesis_phase']}")
            print(f"   LTM size: {snapshot['ltm_size']}")
            print(f"   Surprises: {snapshot['total_surprises']}")
            print(f"   Transfers: {snapshot['total_transfers']}")
            print(f"   Progress: {snapshot['genesis_progress']}")

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
