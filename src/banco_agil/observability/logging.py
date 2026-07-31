"""Logging estruturado com redaction de PII (CPF / data de nascimento)."""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

_CPF_KEY_RE = re.compile(r"cpf", re.IGNORECASE)
_BIRTH_KEY_RE = re.compile(r"(data_nascimento|birth_date|nascimento)", re.IGNORECASE)
_CPF_VALUE_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")

_logging_configured = False


def _redact_value(key: str, value: Any) -> Any:
    """Aplica regras de redaction a um par chave/valor."""
    if _BIRTH_KEY_RE.search(key):
        return "[REDACTED]"
    if _CPF_KEY_RE.search(key):
        return "***"
    if isinstance(value, str):
        return _CPF_VALUE_RE.sub("***", value)
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    return value


def redact_pii(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Processador structlog que mascara CPF e remove data de nascimento.

    Args:
        _logger: Logger (não usado).
        _method_name: Nome do método de log.
        event_dict: Evento estruturado.

    Returns:
        Evento com PII redigida.
    """
    return {str(key): _redact_value(str(key), value) for key, value in event_dict.items()}


def configure_logging(*, json_logs: bool = False, force: bool = False) -> None:
    """Configura structlog no startup.

    Args:
        json_logs: Se True, emite JSON (melhor para containers).
        force: Se True, reconfigura mesmo se já inicializado.
    """
    global _logging_configured
    if _logging_configured and not force:
        return

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_pii,
        structlog.processors.StackInfoRenderer(),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _logging_configured = True


def get_logger(name: str = "banco_agil") -> Any:
    """Retorna um logger estruturado com redaction ativa.

    Args:
        name: Nome lógico do logger.

    Returns:
        BoundLogger do structlog.
    """
    if not _logging_configured:
        configure_logging()
    return structlog.get_logger(name)
