"""Helpers determinísticos para extrair dados de mensagens do usuário."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Any

from banco_agil.utils.currency import normalize_brazilian_currency
from banco_agil.utils.dates import parse_flexible_date

_CPF_RE = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b")
_NUMBER_RE = re.compile(r"(?:R\$\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+(?:[.,]\d+)?)")


def last_user_text(messages: Sequence[Any]) -> str:
    """Retorna o texto da última mensagem humana.

    Args:
        messages: Lista de mensagens LangChain / objetos com ``type``/``content``.

    Returns:
        Conteúdo textual (string vazia se não houver).
    """
    for message in reversed(messages):
        msg_type = getattr(message, "type", None)
        if msg_type == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def extract_cpf(text: str) -> str | None:
    """Extrai o primeiro CPF candidato do texto.

    Args:
        text: Mensagem do usuário.

    Returns:
        CPF (com máscara original) ou None.
    """
    match = _CPF_RE.search(text)
    return match.group(1) if match else None


def extract_date(text: str) -> date | None:
    """Extrai a primeira data reconhecível do texto.

    Args:
        text: Mensagem do usuário.

    Returns:
        ``date`` ou None.
    """
    match = _DATE_RE.search(text)
    if not match:
        return None
    try:
        return parse_flexible_date(match.group(1))
    except ValueError:
        return None


def extract_money(text: str) -> float | None:
    """Extrai o primeiro valor monetário do texto.

    Args:
        text: Mensagem do usuário.

    Returns:
        Float em reais ou None.
    """
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return normalize_brazilian_currency(match.group(1))
    except (ValueError, TypeError):
        return None


def extract_currency_code(text: str, default: str = "USD") -> str:
    """Detecta código de moeda na mensagem.

    Args:
        text: Mensagem do usuário.
        default: Fallback (USD).

    Returns:
        Código ISO em maiúsculas.
    """
    upper = text.upper()
    for code in ("USD", "EUR", "GBP", "JPY", "ARS", "CAD", "AUD", "CHF", "CNY"):
        if code in upper:
            return code
    if "DÓLAR" in upper or "DOLAR" in upper:
        return "USD"
    if "EURO" in upper:
        return "EUR"
    if "LIBRA" in upper:
        return "GBP"
    return default


def looks_like_affirmative(text: str) -> bool:
    """Heurística simples de aceite (sim/quero/aceito)."""
    lowered = text.lower()
    tokens = (
        "sim",
        "quero",
        "aceito",
        "pode ser",
        "vamos",
        "ok",
        "claro",
        "topa",
        "topo",
    )
    return any(token in lowered for token in tokens)


def looks_like_negative(text: str) -> bool:
    """Heurística simples de recusa."""
    lowered = text.lower()
    tokens = ("não", "nao", "agora não", "dispenso", "obrigado não")
    return any(token in lowered for token in tokens)


def looks_like_end(text: str) -> bool:
    """Detecta pedido de encerramento."""
    lowered = text.lower()
    tokens = (
        "encerrar",
        "tchau",
        "adeus",
        "finalizar",
        "sair",
        "até logo",
        "ate logo",
        "obrigado, pode terminar",
    )
    return any(token in lowered for token in tokens)


def heuristic_intent(text: str) -> str | None:
    """Classifica intenção por palavras-chave (fallback do roteador).

    Args:
        text: Mensagem do usuário.

    Returns:
        Intent ou None se não houver sinal claro.
    """
    lowered = text.lower()
    if looks_like_end(text):
        return "end"
    if any(k in lowered for k in ("limite", "crédito", "credito", "cartão", "cartao", "aument")):
        return "credit"
    if any(
        k in lowered for k in ("dólar", "dolar", "euro", "câmbio", "cambio", "cotação", "cotacao")
    ):
        return "exchange"
    if any(k in lowered for k in ("entrevista", "score", "renda", "financeiro")):
        return "interview"
    return None
