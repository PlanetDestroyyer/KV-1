"""
LLM plugin bridge for KV-1.

Integrates with Ollama (Gemma3) so KV-1 can issue live calls or run in
fallback mode when the local daemon is unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None

DEFAULT_OLLAMA_MODEL = "gemma3:4b"


class LLMBridge:
    """Handles provider configuration and live API calls (Ollama by default)."""

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_OLLAMA_MODEL,
        *,
        host: Optional[str] = None,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ):
        self.provider = provider
        self.api_key = api_key  # unused for Ollama but kept for API parity
        self.default_model = default_model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.logger = logging.getLogger("kv1.llm")
        self.client = None
        self._build_client()

    def configure(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        if provider:
            self.provider = provider
        if api_key:
            self.api_key = api_key
        if default_model:
            self.default_model = default_model
        if host:
            self.host = host
        self._build_client()

    def is_configured(self) -> bool:
        return self.client is not None

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.default_model,
            "configured": self.client is not None,
        }

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        execute: bool = True,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        result: Dict[str, Any] = {
            "provider": self.provider,
            "request": {
                "model": self.default_model,
                "messages": [m["content"] for m in messages],
            },
            "executed": False,
            "response": None,
            "text": None,
        }

        if execute and self.client:
            try:
                start = time.time()
                response = self._invoke_with_retry(messages)
                text = response.get("message", {}).get("content") if isinstance(response, dict) else str(response)
                result["response"] = response
                result["text"] = text
                result["executed"] = True
                elapsed = time.time() - start
                self.logger.info(
                    "LLM call success",
                    extra={
                        "provider": self.provider,
                        "model": self.default_model,
                        "duration": elapsed,
                    },
                )
            except Exception as exc:
                result["error"] = f"LLM call failed: {exc}"
                result["text"] = f"[offline fallback] {user_input}"
                self.logger.error(
                    "LLM call failed",
                    exc_info=True,
                    extra={"provider": self.provider, "model": self.default_model},
                )
        else:
            result["error"] = "Ollama client unavailable"
            result["text"] = f"[offline fallback] {user_input}"

        return result

    def _build_client(self):
        if self.provider.lower() == "ollama" and OllamaClient is not None:
            try:
                self.client = OllamaClient(host=self.host)
            except Exception:
                self.client = None
        else:
            self.client = None

    def _invoke_with_retry(self, messages):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat(model=self.default_model, messages=messages)
            except Exception as exc:
                last_exc = exc
                wait = self.backoff_seconds * attempt
                self.logger.warning(
                    "LLM call attempt failed",
                    exc_info=True,
                    extra={"attempt": attempt, "backoff": wait},
                )
                time.sleep(wait)
        raise last_exc if last_exc else RuntimeError("LLM call failed without exception")
