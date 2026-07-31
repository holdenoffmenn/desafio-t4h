"""Tools de controle de sessão."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


def end_conversation_update() -> dict[str, Any]:
    """Marca a sessão para encerramento.

    Returns:
        Update parcial de SessionState com ``should_end=True``.
    """
    return {"should_end": True}


@tool
def end_conversation() -> str:
    """Encerra o atendimento a pedido do usuário.

    Returns:
        Confirmação textual para o LLM.
    """
    return "conversation_ended"
