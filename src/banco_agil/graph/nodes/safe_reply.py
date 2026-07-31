"""Nó de resposta segura quando o input é bloqueado."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from banco_agil.graph.state import SessionState

_SAFE_MESSAGE = (
    "Não consigo ajudar com esse tipo de solicitação. "
    "Posso auxiliar com crédito, câmbio ou sua conta."
)


def safe_reply_node(state: SessionState) -> dict[str, Any]:
    """Responde de forma segura sem invocar LLM.

    Args:
        state: Estado da sessão.

    Returns:
        Update com mensagem segura e flags resetados para o próximo turno.
    """
    _ = state
    return {
        "active_agent": "safe_reply",
        "messages": [AIMessage(content=_SAFE_MESSAGE)],
        "input_blocked": False,
    }
