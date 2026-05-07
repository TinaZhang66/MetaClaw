"""
Example: Produce GRPO training data for a math reasoning dataset.

Demonstrates:
1. Using ExactMatchReward (rule-based, fast)
2. Using PRMScorer (LLM judge, async)
3. Full pipeline with analysis report
4. Custom FunctionCollector for in-process models

Run:
    python -m grpo_factory.examples.math_grpo_example
"""

import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Sample math problems with ground-truth answers
MATH_PROBLEMS = [
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a math tutor. Solve step by step and put your final answer in \\boxed{}.",
            },
            {"role": "user", "content": "Solve: 3x + 7 = 22. What is x?"},
        ],
        "metadata": {"answer": "5", "topic": "algebra"},
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a math tutor. Solve step by step and put your final answer in \\boxed{}.",
            },
            {"role": "user", "content": "What is the area of a circle with radius 4?"},
        ],
        "metadata": {"answer": "16π", "topic": "geometry"},
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a math tutor. Solve step by step and put your final answer in \\boxed{}.",
            },
            {"role": "user", "content": "If f(x) = 2x² - 3x + 1, what is f(3)?"},
        ],
        "metadata": {"answer": "10", "topic": "functions"},
    },
]


# ---------------------------------------------------------------------------
# Option A: Use RolloutCollector (real LLM API)
# ---------------------------------------------------------------------------

async def run_with_api(api_key: str, output_path: str = "math_grpo_data.jsonl"):
    from grpo_factory import (
        GRPOFactory,
        GRPOFactoryConfig,
        CollectorConfig,
        RewardConfig,
        AdvantageConfig,
        AnalyzerConfig,
        FormatterConfig,
        ExactMatchReward,
        LengthReward,
        PRMScorer,
    )

    config = GRPOFactoryConfig(
        model_name="gpt-4o-mini",
        llm_api_base="https://api.openai.com/v1",
        llm_api_key=api_key,
        collector=CollectorConfig(
            group_size=8,          # G=8 completions per prompt
            temperature=0.9,
            max_new_tokens=512,
        ),
        reward=RewardConfig(
            prm_model="gpt-4o-mini",
            prm_api_key=api_key,
            prm_m=3,               # majority vote over 3 judge calls
        ),
        advantage=AdvantageConfig(
            eps=1e-8,
            kl_penalty_coef=0.0,
        ),
        analyzer=AnalyzerConfig(
            drop_uniform_reward_groups=True,
            min_response_tokens=10,
            max_response_tokens=1024,
        ),
        formatter=FormatterConfig(
            output_format="jsonl",
            output_path=output_path,
            include_logprobs=True,
            include_text=True,
        ),
    )

    factory = GRPOFactory(config)

    # Combine rule-based (fast) + PRM (accurate)
    factory.add_reward(ExactMatchReward(
        extract_pattern=r"\\boxed\{(?P<answer>[^}]+)\}"
    ), weight=2.0)
    factory.add_reward(LengthReward(min_tokens=30, max_tokens=500), weight=0.5)
    factory.add_reward(PRMScorer(config.reward), weight=1.0)

    report = await factory.run(MATH_PROBLEMS, output_path=output_path)
    print(report.summary())
    return report


# ---------------------------------------------------------------------------
# Option B: FunctionCollector — mock/in-process model (no API key needed)
# ---------------------------------------------------------------------------

async def run_with_mock_model(output_path: str = "math_grpo_mock.jsonl"):
    """
    Demonstrates using a custom sampling function (e.g. local vLLM model).
    Uses mock responses for demonstration purposes.
    """
    import random
    from grpo_factory import (
        GRPOFactory,
        GRPOFactoryConfig,
        CollectorConfig,
        AdvantageConfig,
        AnalyzerConfig,
        FormatterConfig,
        FunctionCollector,
        ExactMatchReward,
        LengthReward,
    )

    MOCK_ANSWERS = {
        "Solve: 3x + 7 = 22. What is x?": [
            "Step 1: 3x = 22-7 = 15. Step 2: x = 15/3 = \\boxed{5}",
            "3x + 7 = 22 → 3x = 15 → \\boxed{5}",
            "Let me solve: x = (22-7)/3 = \\boxed{5}",
            "I think x = \\boxed{3}",           # wrong
            "\\boxed{5}",
            "Subtracting 7: 3x=15, so x=\\boxed{5}",
            "x = \\boxed{4}",                   # wrong
            "Step 1: subtract 7 from both sides → 3x=15 → x=\\boxed{5}",
        ],
    }

    async def mock_sample_fn(messages, temperature=1.0, max_tokens=512):
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        options = MOCK_ANSWERS.get(user_msg, [
            f"The answer is \\boxed{{{random.randint(1, 20)}}}",
        ])
        content = random.choice(options)
        # Simulate token ids and logprobs
        token_ids = list(range(len(content.split())))
        logprobs = [-random.uniform(0.1, 2.0) for _ in token_ids]
        return {"content": content, "token_ids": token_ids, "logprobs": logprobs}

    config = GRPOFactoryConfig(
        collector=CollectorConfig(group_size=8, temperature=1.0),
        advantage=AdvantageConfig(eps=1e-8),
        analyzer=AnalyzerConfig(drop_uniform_reward_groups=True, min_response_tokens=1),
        formatter=FormatterConfig(output_format="jsonl", output_path=output_path),
    )

    factory = GRPOFactory(config)
    factory.set_collector(FunctionCollector(config.collector, mock_sample_fn))
    factory.add_reward(ExactMatchReward(), weight=1.0)
    factory.add_reward(LengthReward(min_tokens=5, max_tokens=200), weight=0.3)

    report = await factory.run(MATH_PROBLEMS[:1], output_path=output_path)
    print(report.summary())

    # Show a few sample records
    from grpo_factory import load_jsonl
    records = load_jsonl(output_path)
    print(f"\nSample training records ({len(records)} total):")
    for r in records[:3]:
        print(f"  reward={r['reward']:.2f} adv={r['advantage']:.3f} | {r['response_text'][:80]!r}")

    return report


# ---------------------------------------------------------------------------
# Option C: Offline analysis — load existing rollouts and re-score
# ---------------------------------------------------------------------------

async def reanalyze_existing(rollouts_path: str, output_path: str = "reanalyzed.jsonl"):
    """
    Load pre-collected trajectory groups from a checkpoint and re-score them.
    Useful when you want to experiment with different reward functions without
    re-running the expensive LLM sampling step.
    """
    import pickle
    from grpo_factory import GRPOFactory, GRPOFactoryConfig, ExactMatchReward

    with open(rollouts_path, "rb") as f:
        groups = pickle.load(f)  # list[TrajectoryGroup]

    config = GRPOFactoryConfig(
        formatter=FormatterConfig(output_path=output_path),
    )
    factory = GRPOFactory(config)
    factory.add_reward(ExactMatchReward(), weight=1.0)

    scored_groups = await factory.score_groups(groups)
    factory._compute_advantages(scored_groups)
    kept_groups, report = factory.analyze_groups(scored_groups)
    factory.write_groups(kept_groups, output_path)

    print(report.summary())
    return report


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"

    if mode == "mock":
        asyncio.run(run_with_mock_model())
    elif mode == "api":
        api_key = sys.argv[2] if len(sys.argv) > 2 else ""
        asyncio.run(run_with_api(api_key))
    else:
        print("Usage: python math_grpo_example.py [mock|api <api_key>]")
