"""
Process Reward Model (PRM) scorer backed by an LLM judge.

Design directly mirrors MetaClaw's PRMScorer:
- Majority-vote ensemble of M independent judge calls
- Scores: {-1.0, 0.0, 1.0} (unhelpful / unclear / helpful)
- Async, concurrent evaluation with retry logic
- Supports OpenAI-compatible endpoints and custom judge prompts
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import httpx

from .base import BaseReward
from ..config import RewardConfig
from ..trajectory import Trajectory

logger = logging.getLogger(__name__)

# Default judge prompt template (mirrors MetaClaw's judge_prompt)
_DEFAULT_SYSTEM_PROMPT = (
    "You are an objective evaluator. "
    "Given an instruction and a response, score the response as follows:\n"
    "+1 if the response is helpful and correctly addresses the instruction\n"
    " 0 if it is unclear, partially correct, or ambiguous\n"
    "-1 if it is unhelpful, incorrect, or harmful\n"
    "Reply with ONLY: Score: <number>"
)

_DEFAULT_USER_TEMPLATE = (
    "Instruction:\n{instruction}\n\n"
    "Response:\n{response}\n\n"
    "Score:"
)

_SCORE_PATTERN = re.compile(r"Score:\s*([+-]?\d+(?:\.\d+)?)")
_BOXED_PATTERN = re.compile(r"\\boxed\{([+-]?\d+(?:\.\d+)?)\}")


class PRMScorer(BaseReward):
    """
    Async LLM-as-judge reward scorer with majority voting.

    score_async() fires self.m independent judge calls concurrently
    and returns the majority vote.  Ties resolve to 0.0.
    """

    name = "prm"

    def __init__(
        self,
        config: RewardConfig,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        self.config = config
        self.model = config.prm_model
        self.api_base = config.prm_api_base.rstrip("/")
        self.api_key = config.prm_api_key
        self.m = config.prm_m
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._user_template = user_template or _DEFAULT_USER_TEMPLATE
        self._semaphore = asyncio.Semaphore(32)

    def score(self, trajectory: Trajectory) -> float:
        """Sync wrapper — runs the async scorer in a new event loop if needed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Inside an async context — caller should use score_async
                raise RuntimeError("Use score_async() inside async contexts")
            return loop.run_until_complete(self.score_async(trajectory))
        except RuntimeError:
            return asyncio.run(self.score_async(trajectory))

    async def score_async(self, trajectory: Trajectory) -> float:
        """Fire M judge calls, return majority vote."""
        instruction = trajectory.prompt_text
        response = trajectory.response_text

        tasks = [
            asyncio.create_task(self._judge_once(instruction, response))
            for _ in range(self.m)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes: list[float] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Judge call failed: %s", r)
            else:
                votes.append(r)

        if not votes:
            return 0.0

        return _majority_vote(votes)

    async def score_group(self, group) -> list[float]:
        """Score all trajectories in a group concurrently."""
        tasks = [self.score_async(t) for t in group.trajectories]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _judge_once(self, instruction: str, response: str) -> float:
        """Single LLM judge call. Returns parsed score."""
        # Sanitize to avoid content filter issues (mirrors MetaClaw _sanitize)
        instruction = _sanitize(instruction)
        response = _sanitize(response)

        user_msg = self._user_template.format(
            instruction=instruction, response=response
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 16,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/chat/completions"

        async with self._semaphore:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(url, json=payload, headers=headers)
                        resp.raise_for_status()
                        data = resp.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    return _parse_score(text)
                except Exception as exc:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise exc
        return 0.0


class StepwisePRMScorer(PRMScorer):
    """
    Stepwise PRM: scores each reasoning step individually.

    Useful for chain-of-thought trajectories where intermediate
    steps can be evaluated independently.

    Splits the response on numbered steps (1. / 2. / Step N:) and
    averages the per-step scores, weighted toward later steps.
    """

    name = "stepwise_prm"

    _STEP_PATTERN = re.compile(r"(?:^|\n)(?:Step\s+)?\d+[.)]\s+", re.MULTILINE)

    async def score_async(self, trajectory: Trajectory) -> float:
        response = trajectory.response_text
        instruction = trajectory.prompt_text

        steps = self._split_steps(response)
        if len(steps) <= 1:
            # Fall back to whole-response scoring
            return await super().score_async(trajectory)

        # Score each step; later steps get higher weight
        tasks = [
            asyncio.create_task(self._judge_once(instruction, step))
            for step in steps
        ]
        scores = list(await asyncio.gather(*tasks, return_exceptions=True))
        valid = [s for s in scores if isinstance(s, float)]

        if not valid:
            return 0.0

        # Linear increasing weights: last step matters most
        weights = list(range(1, len(valid) + 1))
        total_w = sum(weights)
        return sum(s * w for s, w in zip(valid, weights)) / total_w

    def _split_steps(self, text: str) -> list[str]:
        parts = self._STEP_PATTERN.split(text)
        return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_score(text: str) -> float:
    """Extract numeric score from judge response."""
    m = _SCORE_PATTERN.search(text)
    if m:
        return _clamp(float(m.group(1)))
    m = _BOXED_PATTERN.search(text)
    if m:
        return _clamp(float(m.group(1)))
    # Last-resort: look for standalone integer
    digits = re.findall(r"[+-]?\d+", text)
    if digits:
        return _clamp(float(digits[-1]))
    return 0.0


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _majority_vote(votes: list[float]) -> float:
    """Return the most frequent vote; ties → 0.0."""
    from collections import Counter
    counts = Counter(votes)
    top = counts.most_common(2)
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return 0.0


def _sanitize(text: str) -> str:
    """Strip XML-like tags to reduce content filter false positives."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# RewardPipeline: combine multiple reward sources
# ---------------------------------------------------------------------------

class RewardPipeline:
    """
    Weighted combination of multiple BaseReward instances.

    Applies each scorer, multiplies by its weight, and sums.
    """

    def __init__(self, scorers: list[tuple[BaseReward, float]]) -> None:
        self._scorers = scorers  # [(scorer, weight), ...]

    async def score_group(self, group) -> list[float]:
        """Score all trajectories in a group across all reward sources."""
        import asyncio

        # Run all scorers concurrently across all trajectories
        scorer_tasks = [scorer.score_group(group) for scorer, _ in self._scorers]
        all_scores = await asyncio.gather(*scorer_tasks)

        # Weighted sum per trajectory
        total_w = sum(w for _, w in self._scorers)
        if total_w == 0:
            return [0.0] * len(group.trajectories)

        combined = []
        for i in range(len(group.trajectories)):
            s = sum(scores[i] * w for scores, (_, w) in zip(all_scores, self._scorers))
            combined.append(s / total_w)
        return combined

    async def score_and_assign(self, group, clip: float | None = None) -> None:
        """Score all trajectories and assign .reward in place."""
        scores = await self.score_group(group)
        for traj, score in zip(group.trajectories, scores):
            if clip is not None:
                score = max(-clip, min(clip, score))
            traj.reward = score
