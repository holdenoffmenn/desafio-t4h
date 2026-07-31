"""Normalização e validação de CPF brasileiro."""

from __future__ import annotations


def normalize_cpf(cpf: str) -> str:
    """Remove pontuação e retorna apenas os 11 dígitos do CPF.

    Args:
        cpf: CPF com ou sem máscara (ex.: ``123.456.789-09``).

    Returns:
        String com exatamente os dígitos restantes (pode ter tamanho != 11
        se a entrada for inválida; use :func:`is_valid_cpf` para validar).

    Raises:
        ValueError: Se ``cpf`` estiver vazio após remover não-dígitos.
    """
    digits = "".join(ch for ch in cpf if ch.isdigit())
    if not digits:
        raise ValueError("CPF vazio ou sem dígitos.")
    return digits


def is_valid_cpf(cpf: str) -> bool:
    """Valida CPF pelos dígitos verificadores (módulo 11).

    Args:
        cpf: CPF com ou sem máscara.

    Returns:
        True se o CPF tiver 11 dígitos e dígitos verificadores corretos.
    """
    try:
        digits = normalize_cpf(cpf)
    except ValueError:
        return False

    if len(digits) != 11:
        return False
    if digits == digits[0] * 11:
        return False

    def _check_digit(base: str, weights: list[int]) -> str:
        total = sum(int(d) * w for d, w in zip(base, weights, strict=True))
        remainder = total % 11
        return "0" if remainder < 2 else str(11 - remainder)

    d1 = _check_digit(digits[:9], list(range(10, 1, -1)))
    d2 = _check_digit(digits[:9] + d1, list(range(11, 1, -1)))
    return digits[-2:] == d1 + d2
