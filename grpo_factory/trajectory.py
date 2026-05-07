"""
Core data structures for GRPO trajectory collection and analysis.

Design mirrors MetaClaw's ConversationSample but extends it to the full
GRPO group abstraction: G completions per prompt, with per-token logprobs,
loss masks, rewards, and computed advantages stored together.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrajectoryStep:
    """
    One turn in a (possibly multi-turn) conversation.

    For assistant turns: token_ids, logprobs, and loss_mask are populated.
    For user/system turns: token_ids may be present but logprobs/loss_mask
    are left empty — they contribute to the prompt context only.
    """

    role: str  # "system" | "user" | "assistant"
    content: str

    # Token-level data (populated for assistant turns after sampling)
    token_ids: list[int] = field(default_factory=list)

    # Per-token log-probabilities under the sampling model (p_theta)
    # Shape: [seq_len], same length as token_ids
    logprobs: list[float] = field(default_factory=list)

    # Per-token log-probabilities under a reference/teacher model (p_ref)
    # Used for KL penalty computation; empty if no ref model is used
    ref_logprobs: list[float] = field(default_factory=list)

    # 1 = include in loss, 0 = skip (prompt tokens always 0)
    loss_mask: list[int] = field(default_factory=list)

    def num_tokens(self) -> int:
        return len(self.token_ids)

    def response_logprob_sum(self) -> float:
        """Sum of logprobs for loss-masked tokens (approximate sequence logprob)."""
        return sum(lp * m for lp, m in zip(self.logprobs, self.loss_mask))

    def kl_from_ref(self) -> Optional[float]:
        """
        Per-token average KL divergence from reference model.
        KL(p_theta || p_ref) ≈ log p_theta - log p_ref averaged over masked tokens.
        Returns None if ref_logprobs are not populated.
        """
        if not self.ref_logprobs:
            return None
        masked = [
            (lp - rl) * m
            for lp, rl, m in zip(self.logprobs, self.ref_logprobs, self.loss_mask)
        ]
        n = sum(self.loss_mask)
        return sum(masked) / n if n > 0 else 0.0


@dataclass
class Trajectory:
    """
    A single rollout: one completion for a given prompt.

    Multiple Trajectory objects sharing the same group_id form a GRPO group
    (G completions for the same prompt) used for advantage normalisation.
    """

    # Unique trajectory identifier
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Shared across all G completions for the same prompt
    group_id: str = ""

    # Ordered conversation turns
    steps: list[TrajectoryStep] = field(default_factory=list)

    # Scalar reward assigned by the reward pipeline (None = not yet scored)
    reward: Optional[float] = None

    # GRPO advantage: (reward - group_mean) / (group_std + eps)
    advantage: Optional[float] = None

    # Unix timestamp of when the rollout was collected
    timestamp: float = field(default_factory=time.time)

    # Free-form metadata (task type, dataset source, eval results, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Convenience accessors ---

    @property
    def prompt_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.role != "assistant"]

    @property
    def response_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.role == "assistant"]

    @property
    def prompt_text(self) -> str:
        return "\n".join(s.content for s in self.prompt_steps)

    @property
    def response_text(self) -> str:
        return "\n".join(s.content for s in self.response_steps)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.num_tokens() for s in self.prompt_steps)

    @property
    def total_response_tokens(self) -> int:
        return sum(s.num_tokens() for s in self.response_steps)

    def flat_token_ids(self) -> list[int]:
        return [tid for s in self.steps for tid in s.token_ids]

    def flat_logprobs(self) -> list[float]:
        return [lp for s in self.steps for lp in s.logprobs]

    def flat_loss_mask(self) -> list[int]:
        return [m for s in self.steps for m in s.loss_mask]

    def sequence_reward_signal(self, kl_coef: float = 0.0) -> Optional[float]:
        """
        Effective reward after optional KL penalty (mirrors MetaClaw trainer logic).
        r_eff = reward - kl_coef * mean_kl_over_response_tokens
        """
        if self.reward is None:
            return None
        if kl_coef == 0.0:
            return self.reward
        kl_vals = [s.kl_from_ref() for s in self.response_steps if s.kl_from_ref() is not None]
        mean_kl = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0
        return self.reward - kl_coef * mean_kl


@dataclass
class TrajectoryGroup:
    """
    G trajectories sampled for the same prompt.

    This is the unit of GRPO advantage computation: rewards are normalised
    within the group before being written to training data.
    """

    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Original prompt text (before any templating)
    prompt: str = ""

    # The G sampled completions
    trajectories: list[Trajectory] = field(default_factory=list)

    # Populated by AdvantageComputer
    group_mean_reward: Optional[float] = None
    group_std_reward: Optional[float] = None

    # Metadata forwarded from the dataset (e.g. ground-truth answer, task_id)
    metadata: dict[str, Any] = field(default_factory=dict)

    def rewards(self) -> list[Optional[float]]:
        return [t.reward for t in self.trajectories]

    def advantages(self) -> list[Optional[float]]:
        return [t.advantage for t in self.trajectories]

    def is_fully_scored(self) -> bool:
        return all(r is not None for r in self.rewards())

    def is_fully_advantaged(self) -> bool:
        return all(a is not None for a in self.advantages())

    def reward_range(self) -> Optional[float]:
        rs = [r for r in self.rewards() if r is not None]
        return (max(rs) - min(rs)) if len(rs) >= 2 else None

    def has_learning_signal(self) -> bool:
        """True if not all rewards are identical (group provides gradient signal)."""
        rs = [r for r in self.rewards() if r is not None]
        return len(set(rs)) > 1 if rs else False


@dataclass
class AnalysisReport:
    """
    Summary statistics produced by TrajectoryAnalyzer over a list of groups.
    Useful for dataset quality checks and logging.
    """

    total_groups: int = 0
    total_trajectories: int = 0

    # Reward statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    # Advantage statistics
    mean_advantage: float = 0.0
    std_advantage: float = 0.0

    # Token statistics
    mean_prompt_tokens: float = 0.0
    mean_response_tokens: float = 0.0
    max_response_tokens: int = 0

    # Filtering statistics
    groups_dropped_uniform: int = 0
    trajectories_dropped_min_reward: int = 0
    trajectories_dropped_length: int = 0
    trajectories_kept: int = 0

    # Diversity
    mean_group_reward_range: float = 0.0
    groups_with_signal: int = 0

    def summary(self) -> str:
        lines = [
            "=== GRPO Trajectory Analysis Report ===",
            f"Groups:        {self.total_groups} total, {self.groups_with_signal} with signal",
            f"Trajectories:  {self.total_trajectories} sampled → {self.trajectories_kept} kept",
            f"Reward:        mean={self.mean_reward:.3f} std={self.std_reward:.3f} "
            f"[{self.min_reward:.3f}, {self.max_reward:.3f}]",
            f"Advantage:     mean={self.mean_advantage:.3f} std={self.std_advantage:.3f}",
            f"Tokens(resp):  mean={self.mean_response_tokens:.0f} max={self.max_response_tokens}",
            f"Dropped:       {self.groups_dropped_uniform} uniform-groups, "
            f"{self.trajectories_dropped_min_reward} low-reward, "
            f"{self.trajectories_dropped_length} bad-length",
        ]
        return "\n".join(lines)
