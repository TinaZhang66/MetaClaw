"""
Configuration dataclasses for the GRPO trajectory factory.
Mirrors MetaClaw's MetaClawConfig pattern but scoped to GRPO data production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CollectorConfig:
    """Controls how rollouts are sampled from the LLM."""

    # Number of completions to sample per prompt (GRPO group size G)
    group_size: int = 8

    # Sampling temperature; higher = more diverse rollouts
    temperature: float = 1.0

    # Max tokens per completion
    max_new_tokens: int = 1024

    # Stop sequences forwarded to the LLM
    stop_sequences: list[str] = field(default_factory=list)

    # Number of concurrent async calls to the LLM
    concurrency: int = 16

    # Retry budget per completion on transient errors
    max_retries: int = 3


@dataclass
class RewardConfig:
    """Controls the reward pipeline."""

    # Which rewards to activate (applied in order, scores summed)
    reward_types: list[str] = field(default_factory=lambda: ["rule_based"])

    # PRM judge model (OpenAI-compatible endpoint)
    prm_model: str = "gpt-4o-mini"
    prm_api_base: str = "https://api.openai.com/v1"
    prm_api_key: str = ""

    # Majority-vote ensemble size for PRM (mirrors MetaClaw's prm_m)
    prm_m: int = 3

    # Weight for each reward type when combining scores
    reward_weights: dict[str, float] = field(default_factory=lambda: {"rule_based": 1.0, "prm": 1.0})

    # Clip rewards to [-clip, +clip] before advantage computation
    reward_clip: Optional[float] = None


@dataclass
class AdvantageConfig:
    """GRPO advantage computation settings."""

    # Epsilon for numerical stability in (r - mean) / (std + eps)
    eps: float = 1e-8

    # If True, also subtract a global running baseline across groups
    use_global_baseline: bool = False

    # EMA decay for global baseline
    baseline_ema: float = 0.99

    # KL penalty coefficient (mirrors MetaClaw's kl_penalty_coef)
    # Applied as: advantage -= kl_coef * (log p_student - log p_ref)
    kl_penalty_coef: float = 0.0


@dataclass
class AnalyzerConfig:
    """Controls trajectory filtering and quality analysis."""

    # Drop groups where all rewards are identical (no learning signal)
    drop_uniform_reward_groups: bool = True

    # Drop individual trajectories with reward below this threshold
    min_reward_threshold: Optional[float] = None

    # Drop trajectories whose token length is below / above these limits
    min_response_tokens: int = 1
    max_response_tokens: int = 4096

    # Diversity: flag groups whose response Jaccard similarity exceeds this
    diversity_warn_threshold: float = 0.95


@dataclass
class FormatterConfig:
    """Controls training data serialisation."""

    # Output format
    output_format: Literal["jsonl", "parquet", "hf_dataset"] = "jsonl"

    # Output path (file for jsonl/parquet, directory for hf_dataset)
    output_path: str = "grpo_train_data.jsonl"

    # Whether to include per-token logprobs in output (needed for PPO-clip / KL)
    include_logprobs: bool = True

    # Whether to include raw text alongside token ids
    include_text: bool = True

    # Shard size (number of groups per output shard); 0 = single file
    shard_size: int = 0


@dataclass
class GRPOFactoryConfig:
    """Top-level config combining all sub-configs."""

    collector: CollectorConfig = field(default_factory=CollectorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    formatter: FormatterConfig = field(default_factory=FormatterConfig)

    # LLM endpoint for rollout sampling
    model_name: str = "gpt-4o-mini"
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""

    # Seed for reproducibility
    seed: int = 42
