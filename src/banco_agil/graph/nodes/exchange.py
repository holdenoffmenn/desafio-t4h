"""Nó de câmbio: consulta cotação via FxClient."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from banco_agil.deps import AppDeps
from banco_agil.domain.errors import FxUnavailableError
from banco_agil.graph.state import SessionState
from banco_agil.tools.exchange_tools import get_exchange_rate_update
from banco_agil.tools.session_tools import end_conversation_update
from banco_agil.utils.conversation import (
    extract_currency_code,
    last_user_text,
    looks_like_end,
)


def make_exchange_node(deps: AppDeps):
    """Factory do nó de câmbio.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def exchange_node(state: SessionState) -> dict[str, Any]:
        """Consulta e apresenta a cotação solicitada.

        Args:
            state: Estado da sessão.

        Returns:
            Update com cotação ou mensagem de erro controlada.
        """
        text = last_user_text(state.get("messages", []))
        if looks_like_end(text):
            update = end_conversation_update()
            update["active_agent"] = "exchange"
            update["messages"] = [AIMessage(content="Atendimento encerrado. Até logo!")]
            return update

        currency = extract_currency_code(text, default="USD")
        try:
            obs = get_exchange_rate_update(deps.fx, currency)
            rate = obs["last_tool_calls"][0]["result"]
            assert isinstance(rate, dict)
            bid = float(rate["bid"])
            ts = rate.get("timestamp", "")
            msg = (
                f"Cotação atual {currency}/BRL: compra R$ {bid:.4f}".replace(".", ",")
                + (f" (atualizado em {ts})." if ts else ".")
                + " Posso ajudar com mais alguma coisa?"
            )
            return {
                "active_agent": "exchange",
                **obs,
                "messages": [AIMessage(content=msg)],
            }
        except (FxUnavailableError, ValueError) as exc:
            return {
                "active_agent": "exchange",
                "error": str(exc),
                "messages": [
                    AIMessage(
                        content=(
                            "Não consegui obter a cotação agora. "
                            "Tente novamente em instantes ou peça outro assunto."
                        )
                    )
                ],
            }

    return exchange_node
