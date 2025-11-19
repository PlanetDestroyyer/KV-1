"""
LLM plugin bridge for KV-1.

Provides a thin abstraction that knows how to format payloads for the
configured provider (starting with Gemini) while keeping API keys out
of the main orchestrator logic. The bridge does not make HTTP requests
directly so it can run inside offline/unit-test environments; instead
it returns the payload metadata that a plugin host can forward.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests


class LLMBridge:
    """Handles provider configuration and live API calls (Gemini by default)."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        default_model: str = "gemini-1.5-flash-latest",
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        session: Optional[requests.Session] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.default_model = default_model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()
        self.logger = logging.getLogger("kv1.llm")

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
        request = self._build_request(system_prompt, user_input)
        result: Dict[str, Any] = {
            "provider": self.provider,
            "request": request,
            "executed": False,
            "response": None,
            "text": None,
        }

        if execute and self.is_configured() and request:
            try:
                start = time.time()
                data = self._execute_with_retry(request)
                result["response"] = data
                result["text"] = self._extract_text(data)
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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_request(self, system_prompt: str, user_input: str) -> Optional[Dict[str, Any]]:
        if self.provider.lower() != "gemini":
            return None
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
            "endpoint": endpoint,
            "headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key or "",
            },
            "body": body,
        }

    def _extract_text(self, response: Dict[str, Any]) -> Optional[str]:
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return None
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            texts = [part.get("text", "") for part in parts if part.get("text")]
            return "\n".join(texts) if texts else None
        except Exception:
            return None

    def _execute_with_retry(self, request: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    request["endpoint"],
                    headers=request["headers"],
                    json=request["body"],
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
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
