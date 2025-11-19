"""
Model Context Protocol connectors for KV-1.

Provides a registry pattern that lets KV-1 expose rich, structured
context to external LLM tooling (Ollama, MCP, etc). The registry ships
with default connectors (news, user snapshot, traumas, proactive alerts)
and can be extended at runtime via plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MCPConnector:
    """A single MCP connector entry."""

    name: str
    description: str
    handler: Callable[..., Any]
    plugin_name: Optional[str] = None

    def __call__(self, **kwargs) -> Any:
        return self.handler(**kwargs)


class MCPRegistry:
    """Registry for MCP connectors exposed by KV-1."""

    def __init__(
        self,
        orchestrator: "KV1Orchestrator",
        news_provider: Optional[Callable[[str], List[str]]] = None,
    ):
        self.orchestrator = orchestrator
        self.news_provider = news_provider
        self.connectors: Dict[str, MCPConnector] = {}
        self.plugins: Dict[str, Callable[["MCPRegistry"], None]] = {}
        self._latest_headlines: List[str] = []
        self._register_core_connectors()

    def _register_core_connectors(self) -> None:
        """Wire up core connectors that ship with KV-1."""
        self.register(
            "latest_news",
            "Return the latest headlines for a topic",
            self._connector_latest_news,
        )
        self.register(
            "user_snapshot",
            "Summarize current user profile state",
            self._connector_user_snapshot,
        )
        self.register(
            "trauma_focus",
            "Surface top pain points the AI should avoid triggering",
            self._connector_trauma_focus,
        )
        self.register(
            "app_usage",
            "Summarize which apps have been opened recently",
            self._connector_app_usage,
        )
        self.register(
            "proactive_alerts",
            "Run proactive monitor synchronously and return fired triggers",
            self._connector_proactive_alerts,
        )
        self.register(
            "system_prompt",
            "Return the dynamic system prompt KV-1 would feed to an LLM",
            lambda: self.orchestrator.get_system_prompt(),
        )
        self.register(
            "llm_generate",
            "Invoke the configured LLM provider via the plugin bridge",
            self._connector_llm_generate,
        )

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        plugin_name: Optional[str] = None,
    ) -> None:
        """Register a connector."""
        self.connectors[name] = MCPConnector(
            name=name,
            description=description,
            handler=handler,
            plugin_name=plugin_name,
        )

    def list_connectors(self) -> List[Dict[str, str]]:
        """Return metadata for all connectors."""
        return [
            {
                "name": connector.name,
                "description": connector.description,
                "plugin": connector.plugin_name or "core",
            }
            for connector in self.connectors.values()
        ]

    def call(self, name: str, **kwargs) -> Any:
        """Execute a connector."""
        if name not in self.connectors:
            raise ValueError(f"MCP connector '{name}' not registered")
        return self.connectors[name](**kwargs)

    def load_plugin(self, plugin_name: str, factory: Callable[["MCPRegistry"], None]):
        """
        Load a plugin that can register multiple connectors.

        Example:
            def register_gmail(registry):
                registry.register(
                    \"gmail.inbox\",
                    \"List unread Gmail messages\",
                    handler=...
                    plugin_name=\"gmail\",
                )
            registry.load_plugin(\"gmail\", register_gmail)
        """
        factory(self)
        self.plugins[plugin_name] = factory

    def update_news_cache(self, headlines: List[str]) -> None:
        """Allow background sync jobs to push latest headlines."""
        self._latest_headlines = headlines[-10:]

    # --- Core connector handlers -------------------------------------------------

    def _connector_latest_news(self, topic: str = "tech") -> Dict[str, Any]:
        if self.news_provider:
            try:
                headlines = self.news_provider(topic)
                self._latest_headlines = headlines or self._latest_headlines
            except Exception as exc:
                return {"error": f"News provider error: {exc}", "topic": topic}
        return {
            "topic": topic,
            "headlines": self._latest_headlines
            or ["No news provider configured. Register one via MCPRegistry."],
        }

    def _connector_user_snapshot(self) -> Dict[str, Any]:
        profile = self.orchestrator.user_manager.profile
        return {
            "name": profile.name,
            "energy": profile.energy_level,
            "hours_since_meal": round(profile.hours_since_meal(), 2),
            "github_checks_last_hour": profile.github_checks_last_hour,
            "work_mode": profile.work_mode,
            "focus_apps": profile.focus_apps,
            "blocked_apps": profile.blocked_apps,
        }

    def _connector_trauma_focus(self) -> Dict[str, Any]:
        traumas = [
            {
                "trigger": trauma.trigger,
                "pain_level": round(trauma.pain_level, 2),
                "timestamp": trauma.timestamp.isoformat(),
                "context": trauma.context,
            }
            for trauma in self.orchestrator.traumas.get_top_traumas(5)
        ]
        return {"active_traumas": traumas}

    def _connector_app_usage(self) -> Dict[str, Any]:
        usage = self.orchestrator.app_usage
        top = sorted(usage.items(), key=lambda item: item[1], reverse=True)[:5]
        return {"app_usage": [{"package": pkg, "count": count} for pkg, count in top]}

    def _connector_proactive_alerts(self) -> Dict[str, Any]:
        fired = self.orchestrator.monitor.check_triggers_sync()
        return {"fired_triggers": fired}

    def _connector_llm_generate(
        self, prompt: Optional[str] = None, user_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fan a request out to the configured LLM provider.

        Returns metadata describing the HTTP payload to send so integrators
        can forward it from their plugin host.
        """
        return self.orchestrator.llm.generate(
            system_prompt=prompt or self.orchestrator.get_system_prompt(),
            user_input=user_input or "",
        )


# Avoid circular import in type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import KV1Orchestrator
