"""Tests for installed-only desktop model policy."""

from finance_agent.llm.model_policy import ModelTier, select_installed_model


def test_model_policy_never_selects_or_invents_uninstalled_model() -> None:
    """Tier routing returns only exact locally installed names."""

    configured = {ModelTier.QUALITY: "large", ModelTier.BALANCED: "balanced", ModelTier.FAST: "fast"}
    assert select_installed_model(tier=ModelTier.BALANCED, installed_models=["balanced"], configured_models=configured) == "balanced"
    assert select_installed_model(tier=ModelTier.FAST, installed_models=["balanced"], configured_models=configured) is None
