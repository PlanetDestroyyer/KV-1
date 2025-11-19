"""
LLM plugin bridge for KV-1.

Integrates with Gemini via LangChain's ChatGoogleGenerativeAI wrapper
so that KV-1 can issue live calls or dry-run payloads.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    from langchain.schema import HumanMessage, SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_KEY = "AIzaSyCZxKH24ZN8zfSxdt444p2J4eUUymFhYZ4"


class LLMBridge:
    """Handles provider configuration and live API calls (Gemini by default)."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_GEMINI_MODEL,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_KEY
        self.default_model = default_model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.logger = logging.getLogger("kv1.llm")
        self.client: Optional[ChatGoogleGenerativeAI] = None
        self._build_client()

    def configure(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        if provider:
            self.provider = provider
        if api_key:
            self.api_key = api_key
        if default_model:
            self.default_model = default_model
        self._build_client()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.default_model,
            "configured": self.is_configured(),
        }

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """Call the configured LLM (or just build payload if execute=False)."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
        result: Dict[str, Any] = {
            "provider": self.provider,
            "request": {
                "model": self.default_model,
                "messages": [msg.content for msg in messages],
            },
            "executed": False,
            "response": None,
            "text": None,
        }

        if execute and self.is_configured() and self.client:
            try:
                start = time.time()
                response = self._invoke_with_retry(messages)
                result["response"] = {"content": response.content}
                result["text"] = response.content
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
                self.logger.error(
                    "LLM call failed",
                    exc_info=True,
                    extra={"provider": self.provider, "model": self.default_model},
                )
        else:
            if not self.is_configured():
                result["error"] = "LLM provider not configured. Set GEMINI_API_KEY or pass api_key."

        return result

    def _build_client(self):
        if self.provider.lower() == "gemini" and self.api_key:
            self.client = ChatGoogleGenerativeAI(
                model=self.default_model,
                google_api_key=self.api_key,
            )
        else:
            self.client = None

    def _invoke_with_retry(self, messages):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.invoke(messages)
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
