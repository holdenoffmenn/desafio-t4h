"""Smoke tests da redaction de PII no structlog."""

from __future__ import annotations

from banco_agil.observability.logging import redact_pii


def test_redacts_cpf_key_and_value() -> None:
    event = redact_pii(
        None,  # type: ignore[arg-type]
        "info",
        {
            "event": "auth",
            "cpf": "52998224725",
            "message": "cliente 529.982.247-25 autenticado",
        },
    )
    assert event["cpf"] == "***"
    assert "529" not in event["message"]
    assert "***" in event["message"]


def test_redacts_birth_date_key() -> None:
    event = redact_pii(
        None,  # type: ignore[arg-type]
        "info",
        {
            "event": "auth",
            "data_nascimento": "1990-05-15",
            "birth_date": "15/05/1990",
        },
    )
    assert event["data_nascimento"] == "[REDACTED]"
    assert event["birth_date"] == "[REDACTED]"
