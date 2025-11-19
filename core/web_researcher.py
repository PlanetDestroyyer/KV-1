"""Dedicated safe web research helper for KV-1."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup


@dataclass
class ResearchResult:
    query: str
    mode: str
    text: str
    source: str


class WebResearcher:
    def __init__(
        self,
        *,
        cache_dir: str,
        user_agent: str = "KV1/WebResearcher",
        allow_domains: Optional[list] = None,
        daily_cap: int = 20,
        session: Optional[requests.Session] = None,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.user_agent = user_agent
        self.allow_domains = allow_domains or ["wikipedia.org", "khanacademy.org", "arxiv.org", "nasa.gov"]
        self.daily_cap = daily_cap
        self.session = session or requests.Session()
        self.logger = logging.getLogger("kv1.web")
        self.requests_today = 0

    def fetch(self, query: str, mode: str = "scrape") -> Optional[ResearchResult]:
        if self.requests_today >= self.daily_cap:
            self.logger.warning("Daily web cap reached", extra={"query": query})
            return None
        cached = self._cache_path(query, mode)
        if os.path.exists(cached):
            with open(cached, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return ResearchResult(**payload)
        text = self._scrape(query) if mode == "scrape" else self._wiki(query)
        if not text:
            return None
        result = ResearchResult(query=query, mode=mode, text=text, source="web")
        with open(cached, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f)
        self.requests_today += 1
        return result

    def _scrape(self, query: str) -> str:
        # First try Wikipedia API for better educational content
        wiki_result = self._wiki(query)
        if wiki_result and len(wiki_result) > 100:  # Got meaningful Wikipedia content
            return wiki_result

        # Fallback to DuckDuckGo scraping with better filtering
        url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=web"
        try:
            resp = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                # Filter out common garbage
                if self._is_clean_content(text):
                    paragraphs.append(text)

            return "\n".join(paragraphs)[:5000]
        except Exception as exc:
            self.logger.error("Scrape failed", exc_info=True, extra={"query": query})
            return ""

    def _is_clean_content(self, text: str) -> bool:
        """Filter out garbage content like JavaScript warnings, redirects, etc."""
        if not text or len(text) < 20:  # Too short to be meaningful
            return False

        # Filter out common garbage patterns
        garbage_patterns = [
            "javascript",
            "redirected",
            "cookies",
            "enable cookies",
            "browser",
            "click here",
            "non-javascript",
            "tool_code",
            "```",
            "loading...",
            "please wait",
            "error",
            "404",
            "not found",
        ]

        text_lower = text.lower()
        for pattern in garbage_patterns:
            if pattern in text_lower:
                return False

        # Must contain some educational keywords
        educational_patterns = [
            "is", "are", "means", "refers", "defined",
            "example", "such as", "called", "known",
            "number", "word", "sentence", "concept",
        ]

        has_educational_content = any(pattern in text_lower for pattern in educational_patterns)
        return has_educational_content

    def _wiki(self, query: str) -> str:
        api = "https://en.wikipedia.org/api/rest_v1/page/summary/"

        # Extract main concept from question-style queries
        # e.g., "what is a word in language" -> "word"
        query_lower = query.lower().strip()

        # Save context clues for disambiguation
        context_qualifier = None
        if "grammar" in query_lower or "sentence" in query_lower or "paragraph" in query_lower:
            context_qualifier = "linguistics"
        elif "number" in query_lower or "counting" in query_lower or "arithmetic" in query_lower:
            context_qualifier = "mathematics"

        # Remove common question words
        for prefix in ["what is a ", "what is an ", "what is the ", "what is ",
                      "what are ", "explain ", "define "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
                break

        # Remove trailing qualifiers
        for suffix in [" in language", " in mathematics", " in math", " in physics",
                       " in algebra", " in calculus", " grammar", " writing", " mathematics"]:
            if query_lower.endswith(suffix):
                query_lower = query_lower[:-len(suffix)]
                break

        # Clean up and use first significant word if it's still a phrase
        words = query_lower.strip().split()
        if len(words) > 3:
            # For long phrases, take the main noun (usually after "a/an/the")
            query_lower = words[0] if words[0] not in ["a", "an", "the"] else words[1] if len(words) > 1 else words[0]

        # Try multiple slugs in order
        slugs_to_try = [query_lower.strip().replace(" ", "_")]

        # Add disambiguation variant if we have context
        if context_qualifier:
            slugs_to_try.append(f"{query_lower.strip().replace(' ', '_')}_({context_qualifier})")

        for slug in slugs_to_try:
            try:
                resp = self.session.get(
                    api + slug,
                    headers={"User-Agent": self.user_agent},
                    timeout=10,
                )
                if resp.ok:
                    extract = resp.json().get("extract", "")
                    # Check if this is a real article (not a disambiguation page)
                    if extract and len(extract) > 100:
                        return extract[:5000]
            except Exception as exc:
                continue  # Try next slug

        self.logger.error("Wiki fetch failed for all attempts", extra={"query": query, "slugs": slugs_to_try})
        return ""

    def _cache_path(self, query: str, mode: str) -> str:
        safe = query.replace("/", "_")
        return os.path.join(self.cache_dir, f"{mode}_{safe}.json")
