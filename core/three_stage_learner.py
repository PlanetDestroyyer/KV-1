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
import heapq


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
        researcher=None,
    ):
        self.orchestrator = orchestrator
        self.surprise_threshold = surprise_threshold
        self.max_capacity = max_capacity
        self.episodes: "OrderedDict[str, SurpriseEpisode]" = OrderedDict()
        self.researcher = researcher
        self.curiosity_queue = CuriosityQueue()

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
        self._log_episode("surprise_episode", episode)
        self.curiosity_queue.record_episode(episode)
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
                self._log_episode(
                    "episode_rehearsal",
                    episode,
                    extra={"replays": episode.replays},
                )
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
        self._log_episode("episode_transfer", episode)
        self.curiosity_queue.resolve(episode.token)

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
            explanation = payload.get("text")
            if explanation:
                episode.guessed_meaning = explanation
                episode.confidence = min(1.0, episode.confidence + 0.05)
                self.on_usage(episode.token)
                self._log_episode("episode_probe", episode, extra={"query": query})

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

    def summarize_active_episodes(self) -> List[str]:
        """Provide summaries for reflection/goals."""
        return [ep.to_prompt_snippet() for ep in self.episodes.values()]

    # ------------------------------------------------------------------
    # Web surfing
    # ------------------------------------------------------------------
    async def surf_and_learn(self, query: str, mode: str = "scrape"):
        """Use the web as endless experience fuel."""
        if not self.researcher:
            return []
        result = self.researcher.fetch(query, mode=mode)
        if not result or not result.text:
            return []
        tokens = self._extract_unknown_tokens(result.text)
        episodes = []
        for token in tokens[:5]:
            episodes.append(
                self.on_surprise(
                    token,
                    {
                        "source": "web",
                        "query": query,
                        "mood": "curious",
                        "snippet": result.text[:280],
                    },
                )
            )
        self.orchestrator.log_event(
            "web_research",
            {
                "query": query,
                "mode": mode,
                "tokens": [ep.token for ep in episodes if ep],
            },
        )
        return [ep for ep in episodes if ep]

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
        return payload.get("text") or f"Hypothesis about {token}"

    def _extract_unknown_tokens(self, text: str) -> List[str]:
        words = {word.strip(".,():") for word in text.split() if len(word) > 5}
        shuffled = list(words)
        random.shuffle(shuffled)
        return shuffled

    def _log_episode(self, event_type: str, episode: SurpriseEpisode, extra: Optional[Dict[str, Any]] = None):
        data = {
            "episode_id": episode.episode_id,
            "token": episode.token,
            "surprise": episode.surprise,
            "confidence": episode.confidence,
            "replays": episode.replays,
            "context": episode.context,
        }
        if extra:
            data.update(extra)
        self.orchestrator.log_event(event_type, data)

    def next_curiosity_query(self) -> Optional[Dict[str, str]]:
        return self.curiosity_queue.next_item()

    def add_curiosity_item(self, token: str, query: str):
        self.curiosity_queue.add_manual(token, query)


class CuriosityQueue:
    """Priority queue for unknown concepts to research next."""

    def __init__(self):
        self.heap = []
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.counter = 0

    def record_episode(self, episode: SurpriseEpisode):
        priority = episode.surprise + (1.0 - episode.confidence)
        data = {"token": episode.token, "query": episode.context.get("query") or episode.token}
        self.entries[episode.token] = data
        heapq.heappush(self.heap, (-priority, self.counter, episode.token))
        self.counter += 1

    def resolve(self, token: str):
        self.entries.pop(token, None)

    def next_item(self) -> Optional[Dict[str, str]]:
        while self.heap:
            _, _, token = heapq.heappop(self.heap)
            if token in self.entries:
                data = self.entries.pop(token)
                return data
        return None

    def add_manual(self, token: str, query: str):
        data = {"token": token, "query": query}
        self.entries[token] = data
        heapq.heappush(self.heap, (-1.0, self.counter, token))
        self.counter += 1
