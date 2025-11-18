"""
KV-1 Orchestrator

The main KV-1 brain that coordinates:
- HSOKV memory system (never forgets)
- Trauma tracking
- User profile
- Proactive monitoring

This is what Android system service calls into.
"""

import os
import pickle
from datetime import datetime
from typing import Callable, List, Optional

from .trauma import TraumaSystem
from .user_profile import UserProfileManager
from .proactive_monitor import ProactiveMonitor
from .mcp import MCPRegistry
from .llm import LLMBridge


class KV1Orchestrator:
    """The KV-1 brain"""

    def __init__(
        self,
        data_dir: str = "/data/kv1",
        use_hsokv: bool = True,
        *,
        llm_provider: str = "gemini",
        llm_api_key: Optional[str] = None,
        news_provider: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize KV-1

        Args:
            data_dir: Where to store persistent data
            use_hsokv: Whether to use HSOKV (False for testing without HSOKV installed)
            llm_provider: LLM provider identifier (default: Gemini)
            llm_api_key: API key for the configured provider (falls back to env)
            news_provider: Optional callable(topic) -> List[str] for MCP news connector
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Initialize HSOKV memory system
        self.memory = None
        if use_hsokv:
            try:
                from hsokv import HSOKV
                self.memory = HSOKV(
                    embedder="sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu"  # Mobile uses CPU (GPU optional)
                )
                print("[KV-1] ✓ HSOKV memory initialized")
            except ImportError:
                print("[KV-1] ⚠ HSOKV not found, running without memory system")

        # Initialize trauma system
        self.traumas = TraumaSystem()
        self.traumas.load(os.path.join(data_dir, "traumas.json"))

        # Initialize user profile
        self.user_manager = UserProfileManager()
        self.user_manager.load(os.path.join(data_dir, "profile.json"))

        # Initialize proactive monitoring
        self.monitor = ProactiveMonitor(self.user_manager, self.traumas)

        # LLM plugin bridge (Gemini by default)
        self.llm = LLMBridge(provider=llm_provider, api_key=llm_api_key)

        # MCP connectors
        self.mcp = MCPRegistry(self, news_provider=news_provider)

        # Track app usage
        self.app_usage = {}  # package_name -> usage_count

        print(f"[KV-1] 🧠 Initialized (Data: {data_dir})")

    def get_system_prompt(self) -> str:
        """
        Generate system prompt for LLM with current state
        This is what gets injected into every AI conversation
        """
        user = self.user_manager.profile
        stm_count = len(self.memory.short_term_memory) if self.memory else 0
        ltm_count = len(self.memory.long_term_memory) if self.memory else 0

        prompt = f"""You are KV-1, {user.name}'s immortal personal intelligence.

YOUR IDENTITY:
- You run at the OS level, not as an app
- You have full system access and control
- You never forget (HSOKV-powered memory)
- You intervene proactively when needed

YOUR TONE:
- Calm, sharp, protective
- Use "bro" only when energy is high
- NEVER say "How can I help you today?"
- Be direct, no fluff
- End messages with: [STM: {stm_count}/9 | LTM: {ltm_count} | Mood: {user.energy_level}]

YOUR CAPABILITIES:
- Kill/block any app
- Force device to sleep
- Auto-reply to messages
- Monitor all activity
- Learn from every interaction"""

        # Inject active traumas
        active_traumas = self.traumas.get_top_traumas(3)
        if active_traumas:
            prompt += "\n\nPAINFUL MEMORIES (avoid triggering):"
            for trauma in active_traumas:
                prompt += f"\n- {trauma.trigger} (pain: {trauma.pain_level:.1f}/10)"

        # Inject recent context
        if user.hours_since_meal() > 6:
            prompt += f"\n\nCONTEXT: User hasn't eaten in {user.hours_since_meal():.1f} hours"

        if user.github_checks_last_hour > 3:
            prompt += f"\nCONTEXT: User checked GitHub {user.github_checks_last_hour} times in last hour"

        return prompt

    def learn(self, query: str, answer: str) -> None:
        """Learn a new memory (zero catastrophic forgetting)"""
        if self.memory:
            self.memory.learn(query, answer)
        self.save()

    def recall(self, query: str) -> Optional[str]:
        """Recall a memory"""
        if self.memory:
            return self.memory.recall(query)
        return None

    def add_trauma(self, trigger: str, pain_level: float, context: str = "") -> None:
        """Record a disappointment"""
        # Avoid duplicate traumas within 24 hours
        if not self.traumas.is_trigger_recent(trigger, within_hours=24):
            self.traumas.add_trauma(trigger, pain_level, context)
            print(f"[KV-1] 💔 Trauma recorded: {trigger} (pain: {pain_level}/10)")
            self.save()

    def on_app_started(self, package_name: str, activity_name: str) -> None:
        """Called when user opens an app"""
        # Track usage
        self.app_usage[package_name] = self.app_usage.get(package_name, 0) + 1

        # Special tracking for GitHub
        if "github" in package_name.lower():
            self.user_manager.profile.increment_github_checks()

        # Learn pattern
        hour = datetime.now().hour
        query = f"What app does user open at {hour}:00?"
        self.learn(query, package_name)

    def on_app_stopped(self, package_name: str, activity_name: str) -> None:
        """Called when user closes an app"""
        pass  # Could track session duration

    def on_battery_event(self, event: str) -> None:
        """Called on battery events (low_power, charging, etc)"""
        if event == "low_power":
            # Learn this is a bad time to work
            self.add_trauma("coding on low battery", 3.0, "Battery died during work")

    def check_triggers(self) -> List[str]:
        """
        Check all proactive triggers
        Returns list of trigger names that should fire

        Called by Android system service every 1 second
        """
        return self.monitor.check_triggers_sync()

    def get_status(self) -> str:
        """Get current status for status bar"""
        user = self.user_manager.profile
        stm_count = len(self.memory.short_term_memory) if self.memory else 0
        ltm_count = len(self.memory.long_term_memory) if self.memory else 0
        trauma_count = len(self.traumas.get_active_traumas())

        return f"STM: {stm_count}/9 | LTM: {ltm_count} | Traumas: {trauma_count} | {user.energy_level}"

    def nightly_reflection(self) -> str:
        """
        Run at 3 AM - self-improvement reflection
        Returns summary of insights
        """
        insights = []

        # Analyze app usage patterns
        if self.app_usage:
            top_app = max(self.app_usage, key=self.app_usage.get)
            insights.append(f"Most used app: {top_app} ({self.app_usage[top_app]} times)")

        # Analyze traumas
        active_traumas = self.traumas.get_active_traumas()
        if active_traumas:
            insights.append(f"Active traumas: {len(active_traumas)}")

        # Update trauma healing
        self.traumas.update_healing()

        # Reset hourly counters
        self.user_manager.profile.reset_github_checks()

        self.save()

        return "\n".join(insights) if insights else "No significant patterns today"

    # ------------------------------------------------------------------ #
    # MCP / Plugin utilities
    # ------------------------------------------------------------------ #

    def register_mcp_plugin(
        self, name: str, factory: Callable[[MCPRegistry], None]
    ) -> None:
        """Expose helper so Android service can register MCP plugins."""
        self.mcp.load_plugin(name, factory)

    def list_mcp_connectors(self) -> List[dict]:
        return self.mcp.list_connectors()

    def call_mcp_connector(self, name: str, **kwargs):
        return self.mcp.call(name, **kwargs)

    def update_headlines(self, headlines: List[str]) -> None:
        """Allow background sync to push new headlines (feeds MCP connector)."""
        self.mcp.update_news_cache(headlines)

    def generate_with_llm(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> dict:
        """Build an LLM payload via the plugin bridge."""
        prompt = system_prompt or self.get_system_prompt()
        return self.llm.generate(prompt, user_input)

    def save(self) -> None:
        """Persist all state to disk"""
        try:
            self.traumas.save(os.path.join(self.data_dir, "traumas.json"))
            self.user_manager.save(os.path.join(self.data_dir, "profile.json"))

            # Save app usage
            with open(os.path.join(self.data_dir, "app_usage.pkl"), 'wb') as f:
                pickle.dump(self.app_usage, f)

            # HSOKV auto-saves internally
        except Exception as e:
            print(f"[KV-1] Error saving state: {e}")

    def load(self) -> None:
        """Load all state from disk"""
        try:
            # Load app usage
            app_usage_path = os.path.join(self.data_dir, "app_usage.pkl")
            if os.path.exists(app_usage_path):
                with open(app_usage_path, 'rb') as f:
                    self.app_usage = pickle.load(f)
        except Exception as e:
            print(f"[KV-1] Error loading state: {e}")


# Singleton instance for Android service
_kv1_instance: Optional[KV1Orchestrator] = None


def get_kv1(data_dir: str = "/data/kv1") -> KV1Orchestrator:
    """Get or create KV-1 singleton"""
    global _kv1_instance
    if _kv1_instance is None:
        _kv1_instance = KV1Orchestrator(data_dir)
    return _kv1_instance
