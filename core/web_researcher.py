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
        # Try multiple sources in order
        sources = [
            ("Wikipedia", self._wiki),
            ("Simple Wikipedia", self._simple_wiki),
            ("Britannica", self._britannica),
            ("Direct web search", self._web_search),
        ]

        for source_name, source_func in sources:
            try:
                result = source_func(query)
                if result and len(result) > 100:
                    self.logger.info(f"Content found from {source_name}", extra={"query": query})
                    return result
            except Exception as exc:
                self.logger.debug(f"{source_name} failed", exc_info=True, extra={"query": query})
                continue

        self.logger.warning("All sources failed", extra={"query": query})
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
        elif ("number" in query_lower or "counting" in query_lower or "arithmetic" in query_lower or
              "algebra" in query_lower or "mathematics" in query_lower or "math" in query_lower or
              "calculus" in query_lower or "equation" in query_lower or "variable" in query_lower or
              "derivative" in query_lower or "integral" in query_lower or "function" in query_lower):
            context_qualifier = "mathematics"
        elif ("energy" in query_lower or "temperature" in query_lower or "thermodynamics" in query_lower or
              "entropy" in query_lower or "heat" in query_lower or "physics" in query_lower):
            context_qualifier = "physics"

        # Remove common question words
        for prefix in ["what is a ", "what is an ", "what is the ", "what is ",
                      "what are ", "explain ", "define "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
                break

        # Remove trailing qualifiers (try longer patterns first)
        for suffix in [" in language", " in mathematics", " in math", " in physics",
                       " in algebra", " in calculus", " of thermodynamics",
                       " thermodynamics", " algebra", " calculus", " mathematics",
                       " grammar", " writing", " physics"]:
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

    def _simple_wiki(self, query: str) -> str:
        """Try Simple English Wikipedia for easier-to-understand content."""
        api = "https://simple.wikipedia.org/api/rest_v1/page/summary/"

        # Use same query extraction logic
        query_lower = query.lower().strip()

        # Remove question words
        for prefix in ["what is a ", "what is an ", "what is the ", "what is ", "what are "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
                break

        # Remove qualifiers
        for suffix in [" in language", " in mathematics", " in math", " in physics",
                       " in algebra", " in calculus", " of thermodynamics",
                       " thermodynamics", " algebra", " calculus", " mathematics"]:
            if query_lower.endswith(suffix):
                query_lower = query_lower[:-len(suffix)]
                break

        slug = query_lower.strip().replace(" ", "_")

        try:
            resp = self.session.get(
                api + slug,
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            if resp.ok:
                extract = resp.json().get("extract", "")
                if extract and len(extract) > 100:
                    return extract[:5000]
        except Exception:
            pass
        return ""

    def _britannica(self, query: str) -> str:
        """Try Britannica for educational content."""
        # Extract main concept
        query_lower = query.lower().strip()

        for prefix in ["what is a ", "what is an ", "what is the ", "what is ", "what are "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
                break

        for suffix in [" in language", " in mathematics", " in math", " in physics",
                       " in algebra", " in calculus", " of thermodynamics",
                       " thermodynamics", " algebra", " calculus", " mathematics"]:
            if query_lower.endswith(suffix):
                query_lower = query_lower[:-len(suffix)]
                break

        # Try Britannica URL
        slug = query_lower.strip().replace(" ", "-")
        url = f"https://www.britannica.com/topic/{slug}"

        try:
            resp = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            if resp.ok:
                soup = BeautifulSoup(resp.text, "html.parser")

                # Find first paragraph
                paragraphs = []
                for p in soup.find_all("p", limit=5):
                    text = p.get_text(strip=True)
                    if len(text) > 50 and self._is_clean_content(text):
                        paragraphs.append(text)

                if paragraphs:
                    return "\n".join(paragraphs)[:3000]
        except Exception:
            pass
        return ""

    def _web_search(self, query: str) -> str:
        """Fallback to general web search."""
        # Try DuckDuckGo instant answer API first
        try:
            url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1"
            resp = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            if resp.ok:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                if abstract and len(abstract) > 100:
                    return abstract[:3000]

                # Try related topics
                topics = data.get("RelatedTopics", [])
                texts = []
                for topic in topics[:3]:
                    if isinstance(topic, dict) and "Text" in topic:
                        text = topic["Text"]
                        if len(text) > 50:
                            texts.append(text)
                if texts:
                    return "\n".join(texts)[:3000]
        except Exception:
            pass

        # Last resort: scrape DuckDuckGo results page
        try:
            url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=web"
            resp = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if self._is_clean_content(text):
                    paragraphs.append(text)

            return "\n".join(paragraphs)[:5000]
        except Exception:
            pass

        return ""

    def _cache_path(self, query: str, mode: str) -> str:
        safe = query.replace("/", "_")
        return os.path.join(self.cache_dir, f"{mode}_{safe}.json")
