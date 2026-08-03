"""Nó de encerramento do atendimento."""

from __future__ import annotations

from typing import Any

from banco_agil.deps import AppDeps
from banco_agil.graph.nodes._compose import speak
from banco_agil.graph.state import SessionState
from banco_agil.llm.composer import MessageSpec

_END_AUTH_FAIL = (
    "Infelizmente não foi possível autenticar seus dados após algumas tentativas. "
    "Por segurança, encerramos este atendimento. "
    "Quando quiser, inicie uma nova conversa. Estamos à disposição!"
)

_END_DEFAULT = "Foi um prazer atender você. Se precisar de algo mais, é só chamar. Até logo!"


def make_end_node(deps: AppDeps):
    """Factory do nó de encerramento.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def end_node(state: SessionState) -> dict[str, Any]:
        """Emite mensagem de despedida e marca o fim da sessão.

        Args:
            state: Estado da sessão.

        Returns:
            Update com mensagem final e ``should_end=True``.
        """
        if not state.get("authenticated") and state.get("auth_attempts", 0) >= 3:
            spec = MessageSpec(
                goal="encerrar por segurança após falhas de autenticação, com cordialidade "
                "e convite para tentar novamente depois",
                fallback=_END_AUTH_FAIL,
            )
        else:
            spec = MessageSpec(
                goal="despedir-se cordialmente ao encerrar o atendimento",
                fallback=_END_DEFAULT,
            )
        return {
            "active_agent": "end",
            "should_end": True,
            "messages": speak(deps.composer, spec),
        }

    return end_node
