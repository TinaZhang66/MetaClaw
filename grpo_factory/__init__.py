"""
grpo_factory — Lightweight GRPO trajectory analysis & training data producer.

Built on top of MetaClaw-bench's architecture:
- Trajectory collection (AsyncRolloutWorker pattern)
- Reward scoring (PRM majority-vote + rule-based)
- GRPO advantage computation: (r - mean) / (std + eps)
- Quality filtering & statistics (AnalysisReport)
- Training data serialisation (JSONL / Parquet / HF Dataset)

Quick start::

    import asyncio
    from grpo_factory import GRPOFactory, GRPOFactoryConfig
    from grpo_factory.rewards import ExactMatchReward

    config = GRPOFactoryConfig(
        model_name="gpt-4o-mini",
        llm_api_key="sk-...",
    )
    factory = GRPOFactory(config)
    factory.add_reward(ExactMatchReward())

    prompts = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}],
         "metadata": {"answer": "4"}},
    ]
    report = asyncio.run(factory.run(prompts, output_path="train.jsonl"))
    print(report.summary())
"""

from .config import (
    AdvantageConfig,
    AnalyzerConfig,
    CollectorConfig,
    FormatterConfig,
    GRPOFactoryConfig,
    RewardConfig,
)
from .trajectory import (
    AnalysisReport,
    Trajectory,
    TrajectoryGroup,
    TrajectoryStep,
)
from .collector import FunctionCollector, RolloutCollector
from .advantage import GRPOAdvantageComputer, broadcast_advantage_to_tokens
from .analyzer import TrajectoryAnalyzer, trajectory_quality_score
from .formatter import TrainingDataFormatter, TrainingRecord, load_jsonl
from .pipeline import GRPOFactory, produce_from_dataset, diversity_report
from .rewards import (
    BaseReward,
    LengthReward,
    FormatReward,
    ExactMatchReward,
    MultiChoiceReward,
    CompositeRuleReward,
    FunctionReward,
    PRMScorer,
    StepwisePRMScorer,
    RewardPipeline,
)

__all__ = [
    # Config
    "GRPOFactoryConfig",
    "CollectorConfig",
    "RewardConfig",
    "AdvantageConfig",
    "AnalyzerConfig",
    "FormatterConfig",
    # Core data structures
    "Trajectory",
    "TrajectoryGroup",
    "TrajectoryStep",
    "AnalysisReport",
    # Components
    "RolloutCollector",
    "FunctionCollector",
    "GRPOAdvantageComputer",
    "broadcast_advantage_to_tokens",
    "TrajectoryAnalyzer",
    "trajectory_quality_score",
    "TrainingDataFormatter",
    "TrainingRecord",
    "load_jsonl",
    # Pipeline
    "GRPOFactory",
    "produce_from_dataset",
    "diversity_report",
    # Rewards
    "BaseReward",
    "LengthReward",
    "FormatReward",
    "ExactMatchReward",
    "MultiChoiceReward",
    "CompositeRuleReward",
    "FunctionReward",
    "PRMScorer",
    "StepwisePRMScorer",
    "RewardPipeline",
]
