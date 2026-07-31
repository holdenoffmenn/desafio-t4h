"""Parse e normalização de valores monetários brasileiros."""

from __future__ import annotations

from typing import Any


def format_brl(value: float) -> str:
    """Formata valor monetário no padrão brasileiro.

    Args:
        value: Valor em reais.

    Returns:
        String como ``23.000,00`` (sem prefixo R$).
    """
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_brazilian_currency(value: Any) -> float:
    """Converte string monetária BR ou número para ``float``.

    Aceita formatos como ``"5.000,00"``, ``"5000,00"``, ``"5000"`` e números.

    Args:
        value: Valor em string brasileira ou numérico.

    Returns:
        Valor float em reais.

    Raises:
        ValueError: Se a conversão falhar.
        TypeError: Se o tipo não for string nem numérico.
    """
    if isinstance(value, bool):
        raise TypeError("Boolean não é um valor monetário válido.")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Tipo inválido para moeda: {type(value)!r}")

    cleaned = value.strip().replace("R$", "").replace(" ", "")
    if not cleaned:
        raise ValueError("Valor monetário vazio.")

    if "," in cleaned:
        # Formato BR: milhar com ponto, decimal com vírgula
        cleaned = cleaned.replace(".", "").replace(",", ".")
    # else: já está em formato com ponto decimal ou inteiro

    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Não foi possível converter valor monetário: {value!r}") from exc
