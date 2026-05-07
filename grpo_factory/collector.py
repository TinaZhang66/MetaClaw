"""
Rollout collector: samples G completions per prompt from an LLM.

Architecture inspired by MetaClaw's AsyncRolloutWorker + MetaClawAPIServer:
- Async-first, concurrent sampling via asyncio + httpx
- Captures per-token logprobs for KL penalty computation
- Returns structured TrajectoryGroup objects ready for reward scoring
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Optional

import httpx

from .config import CollectorConfig
from .trajectory import Trajectory, TrajectoryGroup, TrajectoryStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message type alias matching OpenAI chat format
# ---------------------------------------------------------------------------
Message = dict[str, Any]  # {"role": ..., "content": ...}
PromptItem = dict[str, Any]  # {"messages": [...], "metadata": {...}}


class RolloutCollector:
    """
    Async rollout collector that samples G completions per prompt.

    Usage::

        collector = RolloutCollector(config, api_base, api_key, model)
        async for group in collector.collect(prompts):
            # group is a TrajectoryGroup with G trajectories (unscored)
            process(group)
    """

    def __init__(
        self,
        config: CollectorConfig,
        api_base: str,
        api_key: str,
        model: str,
    ) -> None:
        self.config = config
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._semaphore = asyncio.Semaphore(config.concurrency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect(
        self,
        prompts: list[PromptItem],
        extra_body: Optional[dict] = None,
    ) -> AsyncIterator[TrajectoryGroup]:
        """
        Yield one TrajectoryGroup per prompt.

        Each group contains config.group_size Trajectory objects.
        Trajectories carry per-token logprobs when the LLM supports them.
        """
        tasks = [
            asyncio.create_task(self._sample_group(item, extra_body or {}))
            for item in prompts
        ]
        for coro in asyncio.as_completed(tasks):
            group = await coro
            yield group

    async def collect_all(
        self,
        prompts: list[PromptItem],
        extra_body: Optional[dict] = None,
    ) -> list[TrajectoryGroup]:
        """Collect all groups eagerly (non-streaming)."""
        groups: list[TrajectoryGroup] = []
        async for g in self.collect(prompts, extra_body):
            groups.append(g)
        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _sample_group(
        self,
        item: PromptItem,
        extra_body: dict,
    ) -> TrajectoryGroup:
        """Sample G completions for a single prompt item."""
        messages: list[Message] = item["messages"]
        metadata: dict = item.get("metadata", {})
        group_id = str(uuid.uuid4())

        # Build prompt text for display/logging
        prompt_text = _messages_to_text(messages)

        # Fire G concurrent completion requests
        tasks = [
            asyncio.create_task(
                self._sample_one(messages, group_id, i, extra_body)
            )
            for i in range(self.config.group_size)
        ]
        trajectories = await asyncio.gather(*tasks, return_exceptions=False)

        return TrajectoryGroup(
            group_id=group_id,
            prompt=prompt_text,
            trajectories=list(trajectories),
            metadata=metadata,
        )

    async def _sample_one(
        self,
        messages: list[Message],
        group_id: str,
        sample_idx: int,
        extra_body: dict,
    ) -> Trajectory:
        """Call the LLM once, parse the response into a Trajectory."""
        async with self._semaphore:
            response = await self._call_llm_with_retry(messages, extra_body)

        choice = response["choices"][0]
        assistant_content = choice["message"]["content"] or ""

        # Parse per-token logprobs if provided
        logprobs_data = choice.get("logprobs") or {}
        token_logprobs: list[float] = []
        token_ids: list[int] = []

        content_lps = logprobs_data.get("content") or []
        for tok in content_lps:
            token_logprobs.append(tok.get("logprob", 0.0))
            # token id may not always be present; use 0 as sentinel
            token_ids.append(tok.get("token_id", 0))

        # Build TrajectorySteps
        prompt_steps = _messages_to_steps(messages)
        response_step = TrajectoryStep(
            role="assistant",
            content=assistant_content,
            token_ids=token_ids,
            logprobs=token_logprobs,
            loss_mask=[1] * len(token_ids),
        )

        traj = Trajectory(
            id=str(uuid.uuid4()),
            group_id=group_id,
            steps=prompt_steps + [response_step],
            timestamp=time.time(),
            metadata={"sample_idx": sample_idx, "model": self.model},
        )
        return traj

    async def _call_llm_with_retry(
        self,
        messages: list[Message],
        extra_body: dict,
    ) -> dict:
        """POST to /v1/chat/completions with exponential backoff retries."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "logprobs": True,
            "top_logprobs": 1,
            **extra_body,
        }
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/chat/completions"

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    logger.warning("LLM call failed (attempt %d/%d), retry in %ds: %s",
                                   attempt + 1, self.config.max_retries, wait, exc)
                    await asyncio.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries") from last_exc


# ---------------------------------------------------------------------------
# Custom sampling function adapter
# ---------------------------------------------------------------------------

class FunctionCollector:
    """
    Wraps a user-supplied async sampling function instead of an HTTP endpoint.

    The function receives (messages, temperature, max_tokens) and must return
    a dict: {"content": str, "token_ids": list[int], "logprobs": list[float]}

    Useful for in-process vLLM / HuggingFace model integration.
    """

    def __init__(
        self,
        config: CollectorConfig,
        sample_fn: Callable,
    ) -> None:
        self.config = config
        self.sample_fn = sample_fn
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def collect_all(
        self,
        prompts: list[PromptItem],
    ) -> list[TrajectoryGroup]:
        tasks = [asyncio.create_task(self._sample_group(item)) for item in prompts]
        return await asyncio.gather(*tasks)

    async def _sample_group(self, item: PromptItem) -> TrajectoryGroup:
        messages = item["messages"]
        metadata = item.get("metadata", {})
        group_id = str(uuid.uuid4())
        prompt_text = _messages_to_text(messages)

        tasks = [
            asyncio.create_task(self._sample_one(messages, group_id, i))
            for i in range(self.config.group_size)
        ]
        trajectories = await asyncio.gather(*tasks)

        return TrajectoryGroup(
            group_id=group_id,
            prompt=prompt_text,
            trajectories=list(trajectories),
            metadata=metadata,
        )

    async def _sample_one(self, messages, group_id, idx) -> Trajectory:
        async with self._semaphore:
            result = await self.sample_fn(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
            )

        token_ids = result.get("token_ids", [])
        logprobs = result.get("logprobs", [])
        content = result.get("content", "")

        prompt_steps = _messages_to_steps(messages)
        response_step = TrajectoryStep(
            role="assistant",
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            loss_mask=[1] * len(token_ids),
        )
        return Trajectory(
            id=str(uuid.uuid4()),
            group_id=group_id,
            steps=prompt_steps + [response_step],
            timestamp=time.time(),
            metadata={"sample_idx": idx},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _messages_to_steps(messages: list[Message]) -> list[TrajectoryStep]:
    steps = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        # Prompt steps have no logprobs / loss_mask
        steps.append(TrajectoryStep(role=role, content=content))
    return steps
