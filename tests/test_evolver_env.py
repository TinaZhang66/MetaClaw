import os

from metaclaw.config import MetaClawConfig
from metaclaw.config_store import ConfigStore
from metaclaw.launcher import MetaClawLauncher


def test_setup_evolver_env_falls_back_to_llm_settings_in_skills_only(monkeypatch):
    launcher = MetaClawLauncher(ConfigStore())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SKILL_EVOLVER_MODEL", raising=False)

    cfg = MetaClawConfig(
        mode="skills_only",
        llm_provider="tinker",
        llm_api_key="tml-test-123",
        llm_api_base="https://mint.macaron.xin/",
        llm_model_id="Qwen/Qwen3-8B",
    )

    launcher._setup_evolver_env(cfg)

    assert os.environ["OPENAI_API_KEY"] == "tml-test-123"
    assert os.environ["OPENAI_BASE_URL"] == "https://mint.macaron.xin/"
    assert os.environ["SKILL_EVOLVER_MODEL"] == "Qwen/Qwen3-8B"


def test_setup_evolver_env_uses_tinker_env_when_config_base_missing(monkeypatch):
    launcher = MetaClawLauncher(ConfigStore())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SKILL_EVOLVER_MODEL", raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "tml-env-456")
    monkeypatch.setenv("TINKER_BASE_URL", "https://tinker.example/v1")

    cfg = MetaClawConfig(
        mode="skills_only",
        llm_provider="tinker",
        llm_model_id="Qwen/Qwen3-8B",
    )

    launcher._setup_evolver_env(cfg)

    assert os.environ["OPENAI_API_KEY"] == "tml-env-456"
    assert os.environ["OPENAI_BASE_URL"] == "https://tinker.example/v1"
    assert os.environ["SKILL_EVOLVER_MODEL"] == "Qwen/Qwen3-8B"
