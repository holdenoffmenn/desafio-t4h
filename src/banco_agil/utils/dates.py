"""Parse flexível de datas de nascimento."""

from __future__ import annotations

from datetime import date, datetime


def parse_flexible_date(value: str | date) -> date:
    """Converte string de data em ``date``.

    Formatos aceitos: ``YYYY-MM-DD``, ``DD/MM/YYYY``, ``DD-MM-YYYY``, ``DDMMAAAA``.

    Args:
        value: Data como string ou já como ``date``.

    Returns:
        Objeto ``date``.

    Raises:
        ValueError: Se nenhum formato conhecido bater.
        TypeError: Se o tipo for inválido.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Tipo inválido para data: {type(value)!r}")

    text = value.strip()
    formats = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d%m%Y")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Data inválida: {value!r}. Use YYYY-MM-DD, DD/MM/YYYY ou DDMMAAAA.")
