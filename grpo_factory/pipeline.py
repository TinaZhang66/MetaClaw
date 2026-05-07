"""
End-to-end GRPO data production pipeline.

Wires together: Collector → RewardPipeline → AdvantageComputer → Analyzer → Formatter

Mirrors the MetaClaw trainer loop structure but focused purely on offline
trajectory analysis and training data generation (no online training backend).

Usage::

    from grpo_factory import GRPOFactory, GRPOFactoryConfig
    from grpo_factory.rewards import ExactMatchReward, PRMScorer, RewardPipeline

    config = GRPOFactoryConfig(model_name="gpt-4o-mini", llm_api_key="sk-...")
    factory = GRPOFactory(config)

    # Add reward functions
    factory.add_reward(ExactMatchReward(), weight=2.0)
    factory.add_reward(PRMScorer(config.reward), weight=1.0)

    # Run
    report = await factory.run(prompts, output_path="train_data.jsonl")
    print(report.summary())
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .advantage import GRPOAdvantageComputer
from .analyzer import TrajectoryAnalyzer, group_diversity_stats
from .collector import FunctionCollector, RolloutCollector
from .config import GRPOFactoryConfig
from .formatter import TrainingDataFormatter
from .rewards.base import BaseReward
from .rewards.prm import RewardPipeline
from .trajectory import AnalysisReport, TrajectoryGroup

logger = logging.getLogger(__name__)

PromptItem = dict[str, Any]


class GRPOFactory:
    """
    Orchestrates the full GRPO data production pipeline.

    Typical workflow:
        1. Provide prompts as list of {"messages": [...], "metadata": {...}}
        2. Collector samples G completions per prompt
        3. RewardPipeline scores each trajectory
        4. AdvantageComputer normalises rewards within each group
        5. Analyzer filters low-quality groups and computes statistics
        6. Formatter writes training data to disk
    """

    def __init__(self, config: GRPOFactoryConfig) -> None:
        self.config = config
        self._reward_scorers: list[tuple[BaseReward, float]] = []
        self._advantage_computer = GRPOAdvantageComputer(config.advantage)
        self._analyzer = TrajectoryAnalyzer(config.analyzer)
        self._formatter = TrainingDataFormatter(config.formatter)
        self._collector: Optional[RolloutCollector] = None

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def add_reward(self, scorer: BaseReward, weight: float = 1.0) -> "GRPOFactory":
        """Register a reward function. Call multiple times to combine rewards."""
        self._reward_scorers.append((scorer, weight))
        return self

    def set_collector(self, collector) -> "GRPOFactory":
        """Override the default HTTP collector with a custom one (e.g. FunctionCollector)."""
        self._collector = collector
        return self

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    async def run(
        self,
        prompts: list[PromptItem],
        output_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> AnalysisReport:
        """
        Full pipeline: collect → reward → advantage → analyze → format → write.

        Args:
            prompts: List of prompt items with "messages" and optional "metadata".
            output_path: Override output path from config.
            dry_run: If True, skip writing to disk (useful for testing).

        Returns:
            AnalysisReport with statistics about the produced dataset.
        """
        start_t = time.time()
        logger.info("Starting GRPO data production for %d prompts", len(prompts))

        # --- Step 1: Collect rollouts ---
        groups = await self._collect(prompts)
        logger.info("Collected %d groups (%d trajectories)",
                    len(groups), sum(len(g.trajectories) for g in groups))

        # --- Step 2: Score with reward pipeline ---
        groups = await self._score(groups)
        logger.info("Scoring complete")

        # --- Step 3: Compute GRPO advantages ---
        self._compute_advantages(groups)
        logger.info("Advantages computed")

        # --- Step 4: Analyze and filter ---
        kept_groups, report = self._analyzer.analyze(groups)
        logger.info("Analysis complete: %d groups kept", len(kept_groups))

        # --- Step 5: Write training data ---
        if not dry_run and kept_groups:
            out_path = self._formatter.write(kept_groups, output_path)
            logger.info("Training data written to %s", out_path)
        elif dry_run:
            logger.info("[dry_run] Skipping write")

        elapsed = time.time() - start_t
        logger.info("Pipeline complete in %.1fs | %s", elapsed, report.summary())

        return report

    async def collect_only(self, prompts: list[PromptItem]) -> list[TrajectoryGroup]:
        """Only collect rollouts without scoring or formatting."""
        return await self._collect(prompts)

    async def score_groups(
        self, groups: list[TrajectoryGroup]
    ) -> list[TrajectoryGroup]:
        """Score pre-collected groups (useful when loading from checkpoint)."""
        return await self._score(groups)

    def analyze_groups(
        self, groups: list[TrajectoryGroup]
    ) -> tuple[list[TrajectoryGroup], AnalysisReport]:
        """Analyze and filter groups that are already scored + advantaged."""
        return self._analyzer.analyze(groups)

    def write_groups(
        self,
        groups: list[TrajectoryGroup],
        output_path: Optional[str] = None,
    ) -> Path:
        """Write groups to disk. Groups must be fully processed."""
        return self._formatter.write(groups, output_path)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _collect(self, prompts: list[PromptItem]) -> list[TrajectoryGroup]:
        collector = self._get_collector()
        if isinstance(collector, FunctionCollector):
            return await collector.collect_all(prompts)
        else:
            return await collector.collect_all(prompts)

    def _get_collector(self) -> RolloutCollector | FunctionCollector:
        if self._collector is not None:
            return self._collector
        return RolloutCollector(
            config=self.config.collector,
            api_base=self.config.llm_api_base,
            api_key=self.config.llm_api_key,
            model=self.config.model_name,
        )

    async def _score(self, groups: list[TrajectoryGroup]) -> list[TrajectoryGroup]:
        if not self._reward_scorers:
            logger.warning(
                "No reward functions registered. "
                "All trajectories will have reward=0.0. "
                "Use factory.add_reward(...) before calling run()."
            )
            for g in groups:
                for t in g.trajectories:
                    t.reward = 0.0
            return groups

        pipeline = RewardPipeline(self._reward_scorers)
        clip = self.config.reward.reward_clip

        tasks = [
            asyncio.create_task(pipeline.score_and_assign(group, clip=clip))
            for group in groups
        ]
        await asyncio.gather(*tasks)
        return groups

    def _compute_advantages(self, groups: list[TrajectoryGroup]) -> None:
        kl_coef = self.config.advantage.kl_penalty_coef
        for group in groups:
            if not group.is_fully_scored():
                logger.warning("Group %s has unscored trajectories — skipping", group.group_id)
                continue
            self._advantage_computer.compute(group, kl_coef=kl_coef)


# ---------------------------------------------------------------------------
# Convenience: run from a dataset file
# ---------------------------------------------------------------------------

async def produce_from_dataset(
    dataset_path: str,
    config: GRPOFactoryConfig,
    reward_fns: Optional[list[tuple[BaseReward, float]]] = None,
    output_path: Optional[str] = None,
) -> AnalysisReport:
    """
    Load prompts from a JSONL dataset file and run the full pipeline.

    Each line in the dataset must be a JSON object with at least:
    {"messages": [...]}  and optionally {"metadata": {...}}
    """
    import json

    prompts: list[PromptItem] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Normalise: if "prompt" key exists, convert to messages format
            if "prompt" in item and "messages" not in item:
                item["messages"] = [{"role": "user", "content": item.pop("prompt")}]
            prompts.append(item)

    factory = GRPOFactory(config)
    if reward_fns:
        for scorer, weight in reward_fns:
            factory.add_reward(scorer, weight)

    return await factory.run(prompts, output_path=output_path)


# ---------------------------------------------------------------------------
# Convenience: quick diversity report
# ---------------------------------------------------------------------------

def diversity_report(groups: list[TrajectoryGroup]) -> list[dict]:
    """Return per-group diversity stats for analysis / debugging."""
    return [group_diversity_stats(g) for g in groups]
