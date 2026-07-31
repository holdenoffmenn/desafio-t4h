"""Nó de encerramento do atendimento."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from banco_agil.graph.state import SessionState

_END_AUTH_FAIL = (
    "Infelizmente não foi possível autenticar seus dados após algumas tentativas. "
    "Por segurança, encerramos este atendimento. "
    "Quando quiser, inicie uma nova conversa. Estamos à disposição!"
)

_END_DEFAULT = "Foi um prazer atender você. Se precisar de algo mais, é só chamar. Até logo!"


def end_node(state: SessionState) -> dict[str, Any]:
    """Emite mensagem de despedida e marca o fim da sessão.

    Args:
        state: Estado da sessão.

    Returns:
        Update com mensagem final e ``should_end=True``.
    """
    if not state.get("authenticated") and state.get("auth_attempts", 0) >= 3:
        content = _END_AUTH_FAIL
    else:
        content = _END_DEFAULT
    return {
        "active_agent": "end",
        "should_end": True,
        "messages": [AIMessage(content=content)],
    }
