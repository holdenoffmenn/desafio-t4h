"""Nó guard: filtro de intenções maliciosas antes de qualquer skill."""

from __future__ import annotations

from typing import Any

from banco_agil.deps import AppDeps
from banco_agil.graph.state import SessionState
from banco_agil.utils.conversation import last_user_text


def make_guard_node(deps: AppDeps):
    """Factory do nó de segurança.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def guard_node(state: SessionState) -> dict[str, Any]:
        """Avalia a última mensagem do usuário contra o SafetyClassifier.

        Args:
            state: Estado da sessão.

        Returns:
            Update com flags de segurança e ``active_agent=guard``.
        """
        text = last_user_text(state.get("messages", []))
        if not deps.settings.safety_enabled:
            return {
                "active_agent": "guard",
                "input_blocked": False,
                "safety_label": "ok",
                "safety_score": 0.0,
            }

        result = deps.safety.check(text)
        return {
            "active_agent": "guard",
            "input_blocked": result.blocked,
            "safety_label": result.label,
            "safety_score": result.score,
        }

    return guard_node
