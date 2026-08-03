"""Helpers determinísticos para extrair dados de mensagens do usuário."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Any

from banco_agil.utils.currency import normalize_brazilian_currency
from banco_agil.utils.dates import parse_flexible_date

_CPF_RE = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|\d{8})\b")
_NUMBER_RE = re.compile(r"(?:R\$\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+(?:[.,]\d+)?)")
# "25 mil", "2,5 mil", "3 milhões": multiplicador por extenso parcial. A
# alternativa de "milhão" vem antes de "mil" para não casar só o prefixo.
_SCALE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(milh[ãa]o|milh[õo]es|mil)\b",
    re.IGNORECASE,
)


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

    Reconhece multiplicadores parciais por extenso (``"25 mil"`` → ``25000``,
    ``"3 milhões"`` → ``3000000``). Valores totalmente por extenso (``"sete
    mil"``) ficam a cargo do interpretador LLM; aqui é apenas a rede de
    segurança determinística.

    Args:
        text: Mensagem do usuário.

    Returns:
        Float em reais ou None.
    """
    scaled = _SCALE_RE.search(text)
    if scaled is not None:
        try:
            base = normalize_brazilian_currency(scaled.group(1))
        except (ValueError, TypeError):
            base = None
        if base is not None:
            multiplier = 1_000_000 if scaled.group(2).lower().startswith("milh") else 1_000
            return base * multiplier

    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return normalize_brazilian_currency(match.group(1))
    except (ValueError, TypeError):
        return None


# Códigos ISO frequentes (rede de segurança sem LLM). Moedas fora desta lista
# são resolvidas pelo extrator LLM — a disponibilidade real fica com a API.
_CURRENCY_CODE_RE = re.compile(
    r"\b(USD|EUR|GBP|JPY|ARS|CAD|AUD|CHF|CNY|BTC|RUB|MXN|ZAR|INR|NOK|SEK|DKK"
    r"|NZD|SGD|HKD|TRY|CLP|COP|PYG|UYU|BOB|PEN|ILS|AED|KRW|THB|PLN|CZK|HUF"
    r"|RON|BGN|HRK|ISK|PHP|IDR|MYR|TWD|VND|EGP|SAR|QAR|KWD|MAD|NGN)\b",
    re.IGNORECASE,
)

# Nomes comuns → código ISO. As variantes mais específicas ("dólar canadense",
# "peso mexicano") precisam vir antes das genéricas ("dólar", "peso") para não
# serem ofuscadas.
_CURRENCY_NAME_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dólar canadense", "dolar canadense", "canadense"), "CAD"),
    (("dólar australiano", "dolar australiano", "australiano"), "AUD"),
    (("dólar neozelandês", "neozeland", "nova zelândia", "nova zelandia"), "NZD"),
    (("dólar de singapura", "singapura"), "SGD"),
    (("dólar de hong kong", "hong kong"), "HKD"),
    (("dólar americano", "dolar americano", "dólar", "dolar"), "USD"),
    (("euro",), "EUR"),
    (("libra esterlina", "libra"), "GBP"),
    (("iene", "yen", "japon"), "JPY"),
    (("franco suíço", "franco suico", "franco", "suíço", "suico", "suiça", "suíça"), "CHF"),
    (("iuan", "yuan", "renminbi", "chinês", "chines", "china"), "CNY"),
    (("rublo", "rússia", "russia", "russo"), "RUB"),
    (("peso mexicano", "méxico", "mexico", "mexican"), "MXN"),
    (("peso chileno", "chile"), "CLP"),
    (("peso colombiano", "colômbia", "colombia"), "COP"),
    (("peso uruguaio", "uruguai"), "UYU"),
    (("guarani", "paraguai"), "PYG"),
    (("boliviano", "bolívia", "bolivia"), "BOB"),
    (("sol peruano", "peru"), "PEN"),
    (("rand", "áfrica do sul", "africa do sul", "sul-africano"), "ZAR"),
    (("rúpia", "rupia", "índia", "india"), "INR"),
    (("coroa norueguesa", "noruega"), "NOK"),
    (("coroa sueca", "suécia", "suecia"), "SEK"),
    (("coroa dinamarquesa", "dinamarca"), "DKK"),
    (("lira turca", "turquia"), "TRY"),
    (("shekel", "israel"), "ILS"),
    (("dirham", "emirados"), "AED"),
    (("won", "coreia", "coreano"), "KRW"),
    (("baht", "tailândia", "tailandia", "tailandês", "tailandes"), "THB"),
    (("zloty", "polônia", "polonia", "polonês", "polones"), "PLN"),
    (("peso argentino", "argentin", "peso"), "ARS"),
    (("bitcoin",), "BTC"),
)


def extract_currency_code(text: str, default: str | None = None) -> str | None:
    """Detecta o código de moeda na mensagem de forma determinística.

    Reconhece códigos ISO frequentes e nomes usuais em português
    (``"peso argentino"`` → ``ARS``, ``"iene"`` → ``JPY``, ...). É a rede de
    segurança determinística; quando nada casa, devolve ``default`` (``None``)
    para que o chamador acione o fallback via LLM ou peça clarificação — em vez
    de assumir dólar silenciosamente.

    Args:
        text: Mensagem do usuário.
        default: Valor retornado quando nenhuma moeda é reconhecida.

    Returns:
        Código ISO em maiúsculas ou ``default`` se não houver correspondência.
    """
    code_match = _CURRENCY_CODE_RE.search(text)
    if code_match is not None:
        return code_match.group(1).upper()

    lowered = text.lower()
    for keywords, code in _CURRENCY_NAME_RULES:
        if any(keyword in lowered for keyword in keywords):
            return code
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


def wants_credit_increase(text: str) -> bool:
    """Detecta pedido explícito de aumento de limite.

    Args:
        text: Mensagem do usuário.

    Returns:
        True se houver sinal claro de solicitação de aumento.
    """
    lowered = text.lower()
    return any(
        k in lowered
        for k in (
            "aument",
            "elevar",
            "subir o limite",
            "novo limite",
            "solicitar aumento",
            "pedir aumento",
            "quero aumentar",
            "quero um aumento",
        )
    )
