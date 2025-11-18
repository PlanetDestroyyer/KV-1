"""
LLM plugin bridge for KV-1.

Provides a thin abstraction that knows how to format payloads for the
configured provider (starting with Gemini) while keeping API keys out
of the main orchestrator logic. The bridge does not make HTTP requests
directly so it can run inside offline/unit-test environments; instead
it returns the payload metadata that a plugin host can forward.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class LLMBridge:
    """Small helper that holds provider information and API keys."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        default_model: str = "gemini-1.5-flash-latest",
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.default_model = default_model

    def configure(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Update runtime configuration (useful for plugin hot-swap)."""
        if provider:
            self.provider = provider
        if api_key:
            self.api_key = api_key
        if default_model:
            self.default_model = default_model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> Dict[str, Any]:
        """Surface metadata for MCP connectors."""
        return {
            "provider": self.provider,
            "model": self.default_model,
            "configured": self.is_configured(),
        }

    def generate(self, system_prompt: str, user_input: str) -> Dict[str, Any]:
        """
        Build a payload for the configured provider.

        The host environment is responsible for actually sending the HTTP
        request. Returning metadata keeps this class deterministic in the
        OSS repo while giving integrators a plug-in ready structure.
        """
        if not self.is_configured():
            return {
                "error": "LLM provider not configured. Set GEMINI_API_KEY or pass api_key.",
                "provider": self.provider,
            }

        if self.provider.lower() == "gemini":
            return self._build_gemini_payload(system_prompt, user_input)

        return {
            "error": f"Provider '{self.provider}' not supported yet.",
            "provider": self.provider,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_gemini_payload(
        self,
        system_prompt: str,
        user_input: str,
    ) -> Dict[str, Any]:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.default_model}:generateContent"
        )
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {"text": user_input},
                    ]
                }
            ],
        }
        return {
            "provider": "gemini",
            "endpoint": endpoint,
            "headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            "body": body,
        }
