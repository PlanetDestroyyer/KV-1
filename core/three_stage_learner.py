"""
Three-stage biological learning loop for KV-1.

Implements surprise episodes (stage 1), rehearsal (stage 2), and
cortical transfer (stage 3) with web-scraping fuel and an autonomous
self-learning loop. This module is intentionally self-contained so it
can run both on Android deployments and on Kaggle-style notebooks.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


def _now_ts() -> float:
    return time.time()


@dataclass
class SurpriseEpisode:
    """Container for a surprise event stored in STM."""

    episode_id: str
    token: str
    timestamp: float
    surprise: float
    context: Dict[str, Any]
    guessed_meaning: str
    confidence: float
    replays: int = 0

    def to_prompt_snippet(self) -> str:
        return (
            f"{self.token} ≈ {self.guessed_meaning} "
            f"(conf={self.confidence:.2f}, surprise={self.surprise:.2f})"
        )


class ThreeStageLearner:
    """
    Biological learning inspired engine:
    1. Surprise episodes are cached in STM.
    2. Rehearsal reinforces usage.
    3. Transfer locks knowledge into LTM.
    """

    def __init__(
        self,
        orchestrator,
        *,
        surprise_threshold: float = 0.6,
        max_capacity: int = 9,
        web_daily_cap: int = 20,
    ):
        self.orchestrator = orchestrator
        self.surprise_threshold = surprise_threshold
        self.max_capacity = max_capacity
        self.episodes: "OrderedDict[str, SurpriseEpisode]" = OrderedDict()
        self.web_samples_today = 0
        self.web_daily_cap = web_daily_cap

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def on_surprise(self, unknown_token: str, context: Optional[Dict[str, Any]] = None):
        """Register a surprise and store it in STM."""
        similarity = self._ltm_similarity(unknown_token)
        surprise = max(0.0, 1.0 - similarity)
        if surprise < self.surprise_threshold:
            return None

        context = context or {}
        guessed_meaning = context.get("guess") or self._guess_meaning(unknown_token, context)
        confidence = min(1.0, 0.6 + surprise * 0.4)
        episode = SurpriseEpisode(
            episode_id=str(uuid.uuid4()),
            token=unknown_token,
            timestamp=_now_ts(),
            surprise=surprise,
            context=context,
            guessed_meaning=guessed_meaning,
            confidence=confidence,
        )
        self._store_episode(episode)
        return episode

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def on_usage(self, token: str):
        """Increase rehearsal counters when the concept is used."""
        for episode in self.episodes.values():
            if episode.token == token:
                episode.replays += 1
                episode.confidence = min(1.0, episode.confidence + 0.15)
                if episode.replays >= 4 and episode.confidence >= 0.94:
                    self._transfer_to_ltm(episode)
                break

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    def _transfer_to_ltm(self, episode: SurpriseEpisode):
        """Consolidate STM episode into LTM."""
        if not self.orchestrator.memory:
            return
        concept_text = (
            f"{episode.token} => {episode.guessed_meaning} "
            f"(learned via 3-stage loop, tone={episode.context.get('mood', 'calm')})"
        )
        self.orchestrator.memory.learn(episode.token, concept_text)
        if episode.episode_id in self.episodes:
            del self.episodes[episode.episode_id]

    # ------------------------------------------------------------------
    # Self learning loop primitives
    # ------------------------------------------------------------------
    async def self_learning_loop(self, interval_seconds: int = 300):
        """Continuously mines STM gaps and rehearses them."""
        while True:
            await self._self_probe()
            await asyncio.sleep(interval_seconds)

    async def _self_probe(self):
        """Probe STM for low-confidence items and reinforce them."""
        pending = sorted(
            self.episodes.values(), key=lambda ep: (ep.confidence, -ep.surprise)
        )[:2]
        for episode in pending:
            query = f"Explain {episode.token} in detail with examples."
            payload = self.orchestrator.generate_with_llm(query)
            explanation = payload.get("body", {}).get("contents", [{}])[0].get("parts", [{}])[-1].get("text", "")
            if explanation:
                episode.guessed_meaning = explanation
                episode.confidence = min(1.0, episode.confidence + 0.05)
                self.on_usage(episode.token)

    def sleep_replay(self):
        """Replay top episodes during nightly reflections."""
        top_items = sorted(
            self.episodes.values(),
            key=lambda ep: (ep.replays, ep.surprise),
            reverse=True,
        )[:10]
        for _ in range(2):
            for episode in top_items:
                self.on_usage(episode.token)

    # ------------------------------------------------------------------
    # Web surfing
    # ------------------------------------------------------------------
    async def surf_and_learn(self, query: str, mode: str = "scrape"):
        """Use the web as endless experience fuel."""
        if self.web_samples_today >= self.web_daily_cap:
            return []
        text = ""
        if mode == "api":
            text = self._wiki_lookup(query)
        else:
            text = self._scrape(query)
        if not text:
            return []
        tokens = self._extract_unknown_tokens(text)
        episodes = []
        for token in tokens[:5]:
            episodes.append(
                self.on_surprise(
                    token,
                    {
                        "source": "web",
                        "query": query,
                        "mood": "curious",
                        "snippet": text[:280],
                    },
                )
            )
        self.web_samples_today += 1
        return [ep for ep in episodes if ep]

    def _scrape(self, query: str) -> str:
        url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=web"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "KV1/three-stage"},
                timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            return "\n".join(paragraphs)[:5000]
        except Exception:
            return ""

    def _wiki_lookup(self, query: str) -> str:
        api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        slug = query.strip().replace(" ", "_")
        try:
            resp = requests.get(
                api + slug,
                headers={"User-Agent": "KV1/three-stage"},
                timeout=10,
            )
            if resp.ok:
                return resp.json().get("extract", "")
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _store_episode(self, episode: SurpriseEpisode):
        self.episodes[episode.episode_id] = episode
        # Capacity control (7±2 items)
        while len(self.episodes) > self.max_capacity:
            self.episodes.popitem(last=False)

    def _ltm_similarity(self, token: str) -> float:
        if not self.orchestrator.memory:
            return 0.0
        try:
            recalled = self.orchestrator.recall(token)
            return 1.0 if recalled else 0.0
        except Exception:
            return 0.0

    def _guess_meaning(self, token: str, context: Dict[str, Any]) -> str:
        prompt = (
            f"You encountered '{token}'. Context: {context.get('snippet', '')}. "
            "Guess the meaning succinctly."
        )
        payload = self.orchestrator.generate_with_llm(prompt)
        return payload.get("body", {}).get("contents", [{}])[0].get("parts", [{}])[-1].get(
            "text", f"Hypothesis about {token}"
        )

    def _extract_unknown_tokens(self, text: str) -> List[str]:
        words = {word.strip(".,():") for word in text.split() if len(word) > 5}
        shuffled = list(words)
        random.shuffle(shuffled)
        return shuffled
