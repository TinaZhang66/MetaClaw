"""
Training data formatter: converts TrajectoryGroups into serialisable training records.

Output schema mirrors what GRPO training backends expect:
- prompt_tokens / response_tokens (token ids)
- logprobs (per-token, for PPO-clip or KL divergence)
- advantages (per-token, broadcast from trajectory-level)
- loss_mask (1 = compute loss, 0 = skip)
- reward / advantage scalars

Supported output formats: JSONL, Parquet, HuggingFace Dataset directory.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .advantage import broadcast_advantage_to_tokens, compute_token_level_advantages
from .config import FormatterConfig
from .trajectory import Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training record schema
# ---------------------------------------------------------------------------

@dataclass
class TrainingRecord:
    """
    One training example produced from a single Trajectory.

    This is the canonical output unit — one record per trajectory,
    not per group.  The group_id links records that share the same prompt.
    """

    # Identifiers
    record_id: str = ""
    group_id: str = ""
    prompt: str = ""

    # Text fields (optional, for readability / debugging)
    prompt_text: str = ""
    response_text: str = ""

    # Token sequences
    prompt_tokens: list[int] = field(default_factory=list)
    response_tokens: list[int] = field(default_factory=list)

    # Full flat sequence (prompt + response concatenated)
    input_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)

    # Per-token logprobs under the sampling policy (p_theta)
    logprobs: list[float] = field(default_factory=list)

    # Per-token advantages (trajectory advantage broadcast to token level)
    token_advantages: list[float] = field(default_factory=list)

    # Scalar reward and advantage
    reward: float = 0.0
    advantage: float = 0.0

    # Group-level normalisation stats
    group_mean_reward: Optional[float] = None
    group_std_reward: Optional[float] = None

    # Free-form metadata from the original prompt/trajectory
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_logprobs: bool = True, include_text: bool = True) -> dict:
        d = asdict(self)
        if not include_logprobs:
            d.pop("logprobs", None)
            d.pop("token_advantages", None)
        if not include_text:
            d.pop("prompt_text", None)
            d.pop("response_text", None)
            d.pop("prompt", None)
        return d


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class TrainingDataFormatter:
    """
    Converts fully-processed TrajectoryGroups → TrainingRecords → files.

    Usage::
        formatter = TrainingDataFormatter(config)
        formatter.write(groups, output_path)
    """

    def __init__(self, config: FormatterConfig) -> None:
        self.config = config

    def to_records(self, groups: list[TrajectoryGroup]) -> list[TrainingRecord]:
        """Convert all groups to a flat list of TrainingRecords."""
        records = []
        for group in groups:
            for traj in group.trajectories:
                record = self._trajectory_to_record(traj, group)
                if record is not None:
                    records.append(record)
        return records

    def iter_records(
        self, groups: list[TrajectoryGroup]
    ) -> Iterator[TrainingRecord]:
        """Lazy iterator over records (memory-efficient for large datasets)."""
        for group in groups:
            for traj in group.trajectories:
                record = self._trajectory_to_record(traj, group)
                if record is not None:
                    yield record

    def write(
        self,
        groups: list[TrajectoryGroup],
        output_path: Optional[str] = None,
    ) -> Path:
        """Write training data to disk in the configured format."""
        path = Path(output_path or self.config.output_path)

        fmt = self.config.output_format
        if fmt == "jsonl":
            return self._write_jsonl(groups, path)
        elif fmt == "parquet":
            return self._write_parquet(groups, path)
        elif fmt == "hf_dataset":
            return self._write_hf_dataset(groups, path)
        else:
            raise ValueError(f"Unknown output format: {fmt}")

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def _trajectory_to_record(
        self, traj: Trajectory, group: TrajectoryGroup
    ) -> Optional[TrainingRecord]:
        if traj.reward is None or traj.advantage is None:
            logger.warning("Trajectory %s missing reward/advantage — skipped", traj.id)
            return None

        # Flat token sequences
        input_ids = traj.flat_token_ids()
        logprobs = traj.flat_logprobs()
        loss_mask = traj.flat_loss_mask()

        # Per-token advantages
        kl_coef = 0.0  # token-level KL handled in advantage.py if needed
        try:
            token_advantages = compute_token_level_advantages(traj, kl_coef=kl_coef)
        except ValueError:
            token_advantages = broadcast_advantage_to_tokens(traj)

        # Prompt / response token ids
        prompt_tokens = [tid for s in traj.prompt_steps for tid in s.token_ids]
        response_tokens = [tid for s in traj.response_steps for tid in s.token_ids]

        return TrainingRecord(
            record_id=traj.id,
            group_id=group.group_id,
            prompt=group.prompt,
            prompt_text=traj.prompt_text,
            response_text=traj.response_text,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs if self.config.include_logprobs else [],
            token_advantages=token_advantages if self.config.include_logprobs else [],
            reward=float(traj.reward),
            advantage=float(traj.advantage),
            group_mean_reward=group.group_mean_reward,
            group_std_reward=group.group_std_reward,
            metadata={**group.metadata, **traj.metadata},
        )

    # ------------------------------------------------------------------
    # JSONL output
    # ------------------------------------------------------------------

    def _write_jsonl(self, groups: list[TrajectoryGroup], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        shard_size = self.config.shard_size
        if shard_size <= 0:
            return self._write_jsonl_single(groups, path)

        shard_idx = 0
        buffer: list[TrainingRecord] = []
        written_paths = []

        for record in self.iter_records(groups):
            buffer.append(record)
            if len(buffer) >= shard_size:
                shard_path = _shard_path(path, shard_idx)
                _write_jsonl_file(
                    buffer, shard_path,
                    self.config.include_logprobs,
                    self.config.include_text,
                )
                written_paths.append(shard_path)
                buffer = []
                shard_idx += 1

        if buffer:
            shard_path = _shard_path(path, shard_idx)
            _write_jsonl_file(
                buffer, shard_path,
                self.config.include_logprobs,
                self.config.include_text,
            )
            written_paths.append(shard_path)

        logger.info("Wrote %d shards to %s", len(written_paths), path.parent)
        return path.parent

    def _write_jsonl_single(self, groups: list[TrajectoryGroup], path: Path) -> Path:
        _write_jsonl_file(
            list(self.iter_records(groups)),
            path,
            self.config.include_logprobs,
            self.config.include_text,
        )
        logger.info("Wrote %d records to %s", sum(len(g.trajectories) for g in groups), path)
        return path

    # ------------------------------------------------------------------
    # Parquet output
    # ------------------------------------------------------------------

    def _write_parquet(self, groups: list[TrajectoryGroup], path: Path) -> Path:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet output: pip install pyarrow")

        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records(groups)
        rows = [r.to_dict(self.config.include_logprobs, self.config.include_text) for r in records]

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(path))
        logger.info("Wrote %d records to %s (Parquet)", len(records), path)
        return path

    # ------------------------------------------------------------------
    # HuggingFace Dataset output
    # ------------------------------------------------------------------

    def _write_hf_dataset(self, groups: list[TrajectoryGroup], path: Path) -> Path:
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets is required for HF output: pip install datasets")

        records = self.to_records(groups)
        rows = [r.to_dict(self.config.include_logprobs, self.config.include_text) for r in records]

        dataset = Dataset.from_list(rows)
        path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(path))
        logger.info("Saved HF Dataset (%d records) to %s", len(records), path)
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl_file(
    records: list[TrainingRecord],
    path: Path,
    include_logprobs: bool,
    include_text: bool,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            line = json.dumps(r.to_dict(include_logprobs, include_text), ensure_ascii=False)
            f.write(line + "\n")


def _shard_path(base: Path, idx: int) -> Path:
    stem = base.stem
    suffix = base.suffix or ".jsonl"
    return base.parent / f"{stem}_{idx:05d}{suffix}"


# ---------------------------------------------------------------------------
# Convenience: load records back from JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file produced by TrainingDataFormatter."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
