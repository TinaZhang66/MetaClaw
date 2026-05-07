"""
Trajectory analyzer: quality filtering, statistics, and diversity metrics.

Inspired by MetaClaw's approach to filtering samples with no learning signal
(uniform reward groups) and enforcing token length constraints.

The analyzer operates on fully scored + advantaged TrajectoryGroups and
produces an AnalysisReport alongside filtered groups ready for formatting.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .config import AnalyzerConfig
from .trajectory import AnalysisReport, Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """
    Filters and analyses a list of TrajectoryGroups.

    Usage::
        analyzer = TrajectoryAnalyzer(config)
        kept_groups, report = analyzer.analyze(groups)
        print(report.summary())
    """

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

    def analyze(
        self, groups: list[TrajectoryGroup]
    ) -> tuple[list[TrajectoryGroup], AnalysisReport]:
        """
        Filter groups/trajectories and compute statistics.

        Returns:
            (kept_groups, report) where kept_groups contains only the
            trajectories that passed all filters.
        """
        report = AnalysisReport()
        report.total_groups = len(groups)
        report.total_trajectories = sum(len(g.trajectories) for g in groups)

        kept_groups: list[TrajectoryGroup] = []

        for group in groups:
            # --- Group-level filter: uniform reward ---
            if self.config.drop_uniform_reward_groups and not group.has_learning_signal():
                report.groups_dropped_uniform += 1
                logger.debug("Dropped uniform-reward group %s", group.group_id)
                continue

            # --- Trajectory-level filters ---
            kept_trajs: list[Trajectory] = []
            for traj in group.trajectories:
                drop_reason = self._check_trajectory(traj)
                if drop_reason == "min_reward":
                    report.trajectories_dropped_min_reward += 1
                elif drop_reason == "length":
                    report.trajectories_dropped_length += 1
                else:
                    kept_trajs.append(traj)

            if not kept_trajs:
                logger.debug("Group %s has no kept trajectories after filtering", group.group_id)
                continue

            # Rebuild group with only kept trajectories
            filtered_group = TrajectoryGroup(
                group_id=group.group_id,
                prompt=group.prompt,
                trajectories=kept_trajs,
                group_mean_reward=group.group_mean_reward,
                group_std_reward=group.group_std_reward,
                metadata=group.metadata,
            )
            kept_groups.append(filtered_group)

        # --- Compute aggregate statistics ---
        self._fill_report_stats(kept_groups, report)

        # --- Diversity warnings ---
        self._check_diversity(kept_groups, report)

        return kept_groups, report

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _check_trajectory(self, traj: Trajectory) -> Optional[str]:
        """Return drop reason string, or None to keep."""
        # Reward threshold
        if (
            self.config.min_reward_threshold is not None
            and traj.reward is not None
            and traj.reward < self.config.min_reward_threshold
        ):
            return "min_reward"

        # Token length
        n_resp = traj.total_response_tokens
        if n_resp < self.config.min_response_tokens:
            return "length"
        if n_resp > self.config.max_response_tokens:
            return "length"

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _fill_report_stats(
        self, groups: list[TrajectoryGroup], report: AnalysisReport
    ) -> None:
        all_rewards: list[float] = []
        all_advantages: list[float] = []
        prompt_tokens: list[int] = []
        response_tokens: list[int] = []
        reward_ranges: list[float] = []

        for g in groups:
            rr = g.reward_range()
            if rr is not None:
                reward_ranges.append(rr)
            if g.has_learning_signal():
                report.groups_with_signal += 1

            for t in g.trajectories:
                if t.reward is not None:
                    all_rewards.append(t.reward)
                if t.advantage is not None:
                    all_advantages.append(t.advantage)
                prompt_tokens.append(t.total_prompt_tokens)
                response_tokens.append(t.total_response_tokens)

        report.trajectories_kept = sum(len(g.trajectories) for g in groups)

        if all_rewards:
            report.mean_reward = _mean(all_rewards)
            report.std_reward = _std(all_rewards)
            report.min_reward = min(all_rewards)
            report.max_reward = max(all_rewards)

        if all_advantages:
            report.mean_advantage = _mean(all_advantages)
            report.std_advantage = _std(all_advantages)

        if response_tokens:
            report.mean_response_tokens = _mean(response_tokens)
            report.max_response_tokens = max(response_tokens)

        if prompt_tokens:
            report.mean_prompt_tokens = _mean(prompt_tokens)

        if reward_ranges:
            report.mean_group_reward_range = _mean(reward_ranges)

    def _check_diversity(
        self, groups: list[TrajectoryGroup], report: AnalysisReport
    ) -> None:
        """Warn about groups where all responses are near-identical."""
        threshold = self.config.diversity_warn_threshold
        low_diversity_count = 0

        for g in groups:
            if len(g.trajectories) < 2:
                continue
            texts = [t.response_text for t in g.trajectories]
            sim = _max_pairwise_jaccard(texts)
            if sim > threshold:
                low_diversity_count += 1

        if low_diversity_count > 0:
            logger.warning(
                "%d/%d groups have near-identical responses (Jaccard > %.2f). "
                "Consider increasing sampling temperature.",
                low_diversity_count, len(groups), threshold,
            )


# ---------------------------------------------------------------------------
# Per-trajectory introspection utilities
# ---------------------------------------------------------------------------

def trajectory_quality_score(traj: Trajectory) -> float:
    """
    Heuristic quality score combining reward, response length, and logprob confidence.

    Returns a value in [0, 1]; higher = better training example.
    """
    # Reward component (normalised to [0,1] from [-1,1])
    reward_norm = (traj.reward + 1.0) / 2.0 if traj.reward is not None else 0.5

    # Length component: prefer responses in [100, 512] tokens
    n = traj.total_response_tokens
    if 100 <= n <= 512:
        length_score = 1.0
    elif n < 100:
        length_score = n / 100.0
    else:
        length_score = max(0.0, 1.0 - (n - 512) / 512.0)

    # Logprob confidence: avoid near-zero probability completions
    steps = traj.response_steps
    if steps:
        all_lps = [lp for s in steps for lp, m in zip(s.logprobs, s.loss_mask) if m]
        avg_lp = sum(all_lps) / len(all_lps) if all_lps else -10.0
        # avg_lp typically in [-10, 0]; normalize to [0,1]
        lp_score = max(0.0, min(1.0, (avg_lp + 10.0) / 10.0))
    else:
        lp_score = 0.5

    return 0.5 * reward_norm + 0.3 * length_score + 0.2 * lp_score


def group_diversity_stats(group: TrajectoryGroup) -> dict:
    """Return diversity metrics for a group."""
    texts = [t.response_text for t in group.trajectories]
    rewards = [t.reward for t in group.trajectories if t.reward is not None]

    return {
        "group_id": group.group_id,
        "size": len(group.trajectories),
        "reward_mean": _mean(rewards),
        "reward_std": _std(rewards),
        "reward_range": max(rewards) - min(rewards) if len(rewards) >= 2 else 0.0,
        "max_jaccard_sim": _max_pairwise_jaccard(texts),
        "unique_responses": len(set(texts)),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _jaccard(a: str, b: str, ngram: int = 3) -> float:
    """Character n-gram Jaccard similarity between two strings."""
    def ngrams(s: str) -> set[str]:
        return {s[i:i+ngram] for i in range(len(s) - ngram + 1)}

    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def _max_pairwise_jaccard(texts: list[str]) -> float:
    """Maximum pairwise Jaccard similarity among a list of texts."""
    if len(texts) < 2:
        return 0.0
    max_sim = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = _jaccard(texts[i], texts[j])
            if sim > max_sim:
                max_sim = sim
    return max_sim
