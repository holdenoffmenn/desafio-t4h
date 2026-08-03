"""Testes da factory provider-agnostic de LLM (sem chamadas reais)."""

from __future__ import annotations

from typing import Any

import pytest

from banco_agil.config import Settings
from banco_agil.llm.factory import _resolve, build_chat_model


def _settings(**overrides: Any) -> Settings:
    """Cria Settings de teste sem ler o ``.env`` do projeto."""
    base: dict[str, Any] = {"llm_provider": "none"}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def test_disabled_provider_returns_none() -> None:
    assert build_chat_model(_settings(llm_provider="none")) is None
    assert build_chat_model(_settings(llm_provider="fake")) is None
    assert build_chat_model(_settings(llm_provider="")) is None


def test_missing_api_key_degrades_to_none() -> None:
    assert build_chat_model(_settings(llm_provider="gemini", gemini_api_key="")) is None
    assert build_chat_model(_settings(llm_provider="groq", groq_api_key="")) is None


def test_unknown_provider_degrades_to_none() -> None:
    assert build_chat_model(_settings(llm_provider="banana")) is None


def test_resolve_maps_aliases() -> None:
    cfg = _resolve(_settings(llm_provider="gemini", gemini_api_key="k"))
    assert cfg is not None
    assert cfg.model_provider == "google_genai"
    assert cfg.kwargs["google_api_key"] == "k"

    cfg_or = _resolve(_settings(llm_provider="openrouter", openrouter_api_key="k"))
    assert cfg_or is not None
    assert cfg_or.model_provider == "openai"
    assert cfg_or.kwargs["base_url"].startswith("https://openrouter.ai")


def test_build_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com chave e init_chat_model mockado, retorna o modelo instanciado."""
    import langchain.chat_models as lcm

    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> object:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(lcm, "init_chat_model", fake_init)
    result = build_chat_model(
        _settings(llm_provider="groq", groq_api_key="secret", llm_model="llama-3.1-8b-instant")
    )
    assert result is sentinel
    assert captured["model"] == "llama-3.1-8b-instant"
    assert captured["kwargs"]["model_provider"] == "groq"
    assert captured["kwargs"]["api_key"] == "secret"


def test_temperature_omitted_for_fixed_sampling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gemini-3.6-flash ignora sampling: temperature não deve ir no request."""
    import langchain.chat_models as lcm

    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(lcm, "init_chat_model", fake_init)
    build_chat_model(
        _settings(
            llm_provider="gemini",
            gemini_api_key="k",
            llm_model="gemini-3.6-flash",
        )
    )
    assert "temperature" not in captured


def test_temperature_passed_for_regular_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modelo que respeita sampling recebe temperature (determinismo bancário)."""
    import langchain.chat_models as lcm

    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(lcm, "init_chat_model", fake_init)
    build_chat_model(
        _settings(
            llm_provider="gemini",
            gemini_api_key="k",
            llm_model="gemini-flash-latest",
            llm_temperature=0.0,
        )
    )
    assert captured["temperature"] == 0.0


def test_build_init_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qualquer erro no init deve degradar para None (não levantar)."""
    import langchain.chat_models as lcm

    def boom(model: str, **kwargs: Any) -> object:
        raise RuntimeError("provider package missing")

    monkeypatch.setattr(lcm, "init_chat_model", boom)
    assert build_chat_model(_settings(llm_provider="openai", openai_api_key="k")) is None
