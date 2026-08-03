"""Tracer Langfuse degrada sem chaves e registra turnos localmente."""

from __future__ import annotations

from typing import Any

from banco_agil.config import Settings
from banco_agil.infrastructure.langfuse_tracer import SessionTracer


def _settings(**overrides: Any) -> Settings:
    """Settings de teste sem ler o ``.env`` do desenvolvedor (determinismo)."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_tracer_disabled_without_keys() -> None:
    tracer = SessionTracer(settings=_settings(langfuse_public_key="", langfuse_secret_key=""))
    assert tracer.enabled is False
    result = tracer.record_turn(
        session_id="s1",
        state={"active_agent": "triage", "authenticated": False, "cpf": "52998224725"},
    )
    assert result.enabled is False
    assert result.trace_url is None
    assert result.trace_id


def test_tracer_records_business_events_locally() -> None:
    tracer = SessionTracer(settings=_settings())
    result = tracer.record_turn(
        session_id="s2",
        state={
            "active_agent": "credit",
            "authenticated": True,
            "last_request_status": "rejeitado",
            "route_source": "context",
        },
    )
    assert result.enabled is False
