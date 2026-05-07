"""Abstract base class for all reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..trajectory import Trajectory, TrajectoryGroup


class BaseReward(ABC):
    """
    All reward implementations inherit from this.

    score() is called once per Trajectory.  Implementations may be
    synchronous (override score) or async (override score_async).
    The pipeline calls score_async, which by default wraps score().
    """

    name: str = "base"

    @abstractmethod
    def score(self, trajectory: Trajectory) -> float:
        """Return a scalar reward for a single trajectory. Sync."""

    async def score_async(self, trajectory: Trajectory) -> float:
        """Async wrapper — override for truly async reward functions."""
        return self.score(trajectory)

    async def score_group(self, group: TrajectoryGroup) -> list[float]:
        """Score all trajectories in a group. Override for batched efficiency."""
        import asyncio
        tasks = [self.score_async(t) for t in group.trajectories]
        return list(await asyncio.gather(*tasks))

    def weight(self) -> float:
        """Default weight when combined in RewardPipeline."""
        return 1.0
