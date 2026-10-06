import pytest

from support_agent.config import DEFAULT_MAX_ITERATIONS, ConfigError, Settings


def test_reads_the_model_from_the_environment():
    settings = Settings.from_env({"ANTHROPIC_MODEL": "  some-model  "})

    assert settings.model == "some-model"
    assert settings.effort is None
    assert settings.max_iterations == DEFAULT_MAX_ITERATIONS


@pytest.mark.parametrize("env", [{}, {"ANTHROPIC_MODEL": "   "}])
def test_model_is_required(env):
    with pytest.raises(ConfigError, match="ANTHROPIC_MODEL"):
        Settings.from_env(env)


def test_optional_settings():
    settings = Settings.from_env(
        {"ANTHROPIC_MODEL": "m", "ANTHROPIC_EFFORT": "Medium", "AGENT_MAX_ITERATIONS": "3"}
    )

    assert settings.effort == "medium"
    assert settings.max_iterations == 3


@pytest.mark.parametrize(
    "env",
    [
        {"ANTHROPIC_EFFORT": "extreme"},
        {"AGENT_MAX_ITERATIONS": "many"},
        {"AGENT_MAX_ITERATIONS": "0"},
    ],
)
def test_invalid_optional_settings_are_rejected(env):
    with pytest.raises(ConfigError):
        Settings.from_env({"ANTHROPIC_MODEL": "m", **env})
