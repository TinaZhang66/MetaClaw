"""
Rule-based reward functions — fast, deterministic, no LLM calls.

These are cheap signal sources that complement the PRM judge.
Each returns a scalar in [-1, 1] or [0, 1] by convention.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Pattern

from .base import BaseReward
from ..trajectory import Trajectory


class LengthReward(BaseReward):
    """
    Reward based on response token length.

    Gives +1 when length is within [min_tokens, max_tokens],
    linearly decays to 0 outside, and clips at -1 for extreme over/under.
    """

    name = "length"

    def __init__(self, min_tokens: int = 50, max_tokens: int = 1024) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def score(self, trajectory: Trajectory) -> float:
        n = trajectory.total_response_tokens
        if self.min_tokens <= n <= self.max_tokens:
            return 1.0
        elif n < self.min_tokens:
            ratio = n / max(self.min_tokens, 1)
            return max(-1.0, 2 * ratio - 1)
        else:
            overshoot = (n - self.max_tokens) / max(self.max_tokens, 1)
            return max(-1.0, 1.0 - overshoot)


class FormatReward(BaseReward):
    """
    Reward for matching a required output format.

    Provide a regex pattern; +1 if matched, -1 if not.
    Common use: check for \boxed{}, JSON structure, markdown code blocks, etc.
    """

    name = "format"

    def __init__(
        self,
        pattern: str,
        reward_match: float = 1.0,
        reward_no_match: float = -1.0,
        flags: int = re.DOTALL,
    ) -> None:
        self._pattern: Pattern = re.compile(pattern, flags)
        self.reward_match = reward_match
        self.reward_no_match = reward_no_match

    def score(self, trajectory: Trajectory) -> float:
        text = trajectory.response_text
        return self.reward_match if self._pattern.search(text) else self.reward_no_match


class ExactMatchReward(BaseReward):
    """
    Reward for exact answer match (useful for math / MCQ tasks).

    Extracts the answer from the response using a regex group named 'answer'
    and compares to the ground-truth stored in trajectory.metadata["answer"].
    Falls back to full text comparison if no capture group.
    """

    name = "exact_match"

    def __init__(
        self,
        extract_pattern: str = r"\\boxed\{(?P<answer>[^}]+)\}",
        normalize_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._pattern: Pattern = re.compile(extract_pattern, re.DOTALL)
        self._normalize = normalize_fn or (lambda x: x.strip().lower())

    def score(self, trajectory: Trajectory) -> float:
        gt = trajectory.metadata.get("answer")
        if gt is None:
            return 0.0

        text = trajectory.response_text
        m = self._pattern.search(text)
        pred = m.group("answer") if m else text

        return 1.0 if self._normalize(pred) == self._normalize(str(gt)) else -1.0


class MultiChoiceReward(BaseReward):
    """
    F1-based reward for multi-select answers.

    Mirrors MetaClaw's benchmark scoring logic:
    - Extracts selected options from \\bbox{A,B,C} or \\boxed{A,B}
    - Computes: score = 1 - (false_pos + false_neg) / total_options
    """

    name = "multi_choice"

    _BBOX_PATTERN = re.compile(r"\\bbox\{([^}]+)\}")
    _BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")

    def score(self, trajectory: Trajectory) -> float:
        gt_raw = trajectory.metadata.get("answer", [])
        if not gt_raw:
            return 0.0

        gt: set[str] = {x.strip().upper() for x in (gt_raw if isinstance(gt_raw, list) else [gt_raw])}
        total_options: int = trajectory.metadata.get("total_options", max(len(gt), 4))

        text = trajectory.response_text
        pred = self._extract(text)

        false_pos = len(pred - gt)
        false_neg = len(gt - pred)
        score = 1.0 - (false_pos + false_neg) / max(total_options, 1)
        # Map [0,1] → [-1,1]
        return 2 * score - 1.0

    def _extract(self, text: str) -> set[str]:
        for pat in (self._BBOX_PATTERN, self._BOXED_PATTERN):
            m = pat.search(text)
            if m:
                return {x.strip().upper() for x in m.group(1).split(",")}
        return set()


class CompositeRuleReward(BaseReward):
    """
    Weighted sum of multiple rule-based rewards.
    """

    name = "composite_rule"

    def __init__(self, rewards: list[tuple[BaseReward, float]]) -> None:
        self._rewards = rewards  # [(reward_fn, weight), ...]

    def score(self, trajectory: Trajectory) -> float:
        total_w = sum(w for _, w in self._rewards)
        if total_w == 0:
            return 0.0
        return sum(r.score(trajectory) * w for r, w in self._rewards) / total_w


class FunctionReward(BaseReward):
    """
    Wraps any user-supplied scoring function.

    fn(trajectory: Trajectory) -> float
    """

    name = "function"

    def __init__(self, fn: Callable[[Trajectory], float], name: str = "function") -> None:
        self._fn = fn
        self.name = name

    def score(self, trajectory: Trajectory) -> float:
        return self._fn(trajectory)
