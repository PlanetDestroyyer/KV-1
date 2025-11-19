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
        url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=web"
        try:
            resp = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if any(domain in text.lower() for domain in self.allow_domains):
                    paragraphs.append(text)
            if not paragraphs:
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            return "\n".join(paragraphs)[:5000]
        except Exception as exc:
            self.logger.error("Scrape failed", exc_info=True, extra={"query": query})
            return ""

    def _wiki(self, query: str) -> str:
        api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        slug = query.strip().replace(" ", "_")
        try:
            resp = self.session.get(
                api + slug,
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            if resp.ok:
                return resp.json().get("extract", "")[:5000]
        except Exception as exc:
            self.logger.error("Wiki fetch failed", exc_info=True, extra={"query": query})
        return ""

    def _cache_path(self, query: str, mode: str) -> str:
        safe = query.replace("/", "_")
        return os.path.join(self.cache_dir, f"{mode}_{safe}.json")
