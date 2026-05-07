"""
GRPO advantage computation.

Core formula (from DeepSeekMath / GRPO paper):
    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

Extended with:
- Optional global running baseline (EMA across groups)
- KL penalty subtraction (mirrors MetaClaw's kl_penalty_coef)
- Per-token advantage broadcasting for sequence-level training
"""

from __future__ import annotations

import math
from typing import Optional

from .config import AdvantageConfig
from .trajectory import Trajectory, TrajectoryGroup


class GRPOAdvantageComputer:
    """
    Computes and assigns GRPO advantages to all trajectories in a group.

    Usage::
        computer = GRPOAdvantageComputer(config)
        computer.compute(group)   # mutates group.trajectories[i].advantage
    """

    def __init__(self, config: AdvantageConfig) -> None:
        self.config = config
        self._global_baseline: float = 0.0
        self._baseline_initialized: bool = False

    def compute(self, group: TrajectoryGroup, kl_coef: Optional[float] = None) -> None:
        """
        Compute and assign advantages for all trajectories in the group.

        Args:
            group: TrajectoryGroup with all rewards already assigned.
            kl_coef: Override for KL penalty coefficient (uses config default if None).
        """
        if not group.is_fully_scored():
            raise ValueError(
                f"Group {group.group_id} has unscored trajectories. "
                "Run the reward pipeline first."
            )

        kl_coef = kl_coef if kl_coef is not None else self.config.kl_penalty_coef
        rewards = [t.reward for t in group.trajectories]  # all non-None at this point

        # Effective rewards: subtract KL penalty per trajectory
        eff_rewards = [
            _effective_reward(t, kl_coef)
            for t in group.trajectories
        ]

        mean_r = _mean(eff_rewards)
        std_r = _std(eff_rewards, mean_r)

        # Optional global baseline subtraction
        if self.config.use_global_baseline:
            mean_r = mean_r - self._update_baseline(mean_r)

        group.group_mean_reward = mean_r
        group.group_std_reward = std_r

        for traj, r_eff in zip(group.trajectories, eff_rewards):
            raw_adv = (r_eff - mean_r) / (std_r + self.config.eps)
            traj.advantage = raw_adv

    def compute_batch(self, groups: list[TrajectoryGroup]) -> None:
        """Compute advantages for a list of groups."""
        for g in groups:
            self.compute(g)

    def _update_baseline(self, group_mean: float) -> float:
        """EMA update of global baseline; returns the baseline BEFORE update."""
        if not self._baseline_initialized:
            self._global_baseline = group_mean
            self._baseline_initialized = True
            return 0.0
        old = self._global_baseline
        self._global_baseline = (
            self.config.baseline_ema * self._global_baseline
            + (1 - self.config.baseline_ema) * group_mean
        )
        return old


# ---------------------------------------------------------------------------
# Per-token advantage broadcasting
# ---------------------------------------------------------------------------

def broadcast_advantage_to_tokens(
    trajectory: Trajectory,
    advantage: Optional[float] = None,
) -> list[float]:
    """
    Broadcast the trajectory-level advantage to per-token values.

    For response tokens with loss_mask=1, the advantage is the trajectory
    advantage (or provided override).  For prompt tokens, advantage is 0.

    Returns a flat list of per-token advantages matching flat_loss_mask().
    """
    adv = advantage if advantage is not None else trajectory.advantage
    if adv is None:
        raise ValueError("Trajectory has no advantage assigned.")

    result = []
    for step in trajectory.steps:
        if step.role == "assistant":
            result.extend(adv * m for m in step.loss_mask)
        else:
            result.extend(0.0 for _ in step.token_ids)
    return result


def compute_token_level_advantages(
    trajectory: Trajectory,
    kl_coef: float = 0.0,
) -> list[float]:
    """
    Per-token advantage = trajectory_advantage - kl_coef * (log p_theta - log p_ref).

    This mirrors MetaClaw's sample_to_datum logic where the per-token KL
    is subtracted from the advantage before passing to the training backend.
    """
    if trajectory.advantage is None:
        raise ValueError("Trajectory has no advantage assigned.")

    token_advs = []
    for step in trajectory.steps:
        if step.role != "assistant":
            token_advs.extend(0.0 for _ in step.token_ids)
            continue

        for i, (lp, mask) in enumerate(zip(step.logprobs, step.loss_mask)):
            if mask == 0:
                token_advs.append(0.0)
                continue

            kl_penalty = 0.0
            if kl_coef > 0 and step.ref_logprobs and i < len(step.ref_logprobs):
                kl_penalty = kl_coef * (lp - step.ref_logprobs[i])

            token_advs.append(trajectory.advantage - kl_penalty)

    return token_advs


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], mean: Optional[float] = None) -> float:
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _effective_reward(trajectory: Trajectory, kl_coef: float) -> float:
    """r_eff = reward - kl_coef * mean_kl_over_response (sequence level)."""
    r = trajectory.reward  # non-None guaranteed by caller
    if kl_coef == 0.0:
        return r

    kl_vals = []
    for step in trajectory.response_steps:
        k = step.kl_from_ref()
        if k is not None:
            kl_vals.append(k)

    mean_kl = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0
    return r - kl_coef * mean_kl
