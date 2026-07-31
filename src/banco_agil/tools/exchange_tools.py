"""Tools de câmbio."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from banco_agil.infrastructure.fx_client import FxClient


def get_exchange_rate_update(fx: FxClient, currency: str) -> dict[str, Any]:
    """Consulta cotação e monta update de observabilidade.

    Args:
        fx: Cliente de câmbio.
        currency: Código ISO da moeda.

    Returns:
        Update com last_tool_calls contendo a cotação.
    """
    rate = fx.get_rate(currency)
    return {
        "last_tool_calls": [
            {
                "name": "get_exchange_rate",
                "args": {"currency": currency},
                "result": rate.model_dump(mode="json"),
            }
        ],
    }


@tool
def get_exchange_rate(currency: str = "USD") -> str:
    """Busca a cotação atual de uma moeda.

    Args:
        currency: Código ISO (default USD).
    """
    return f"get_exchange_rate:{currency}"
