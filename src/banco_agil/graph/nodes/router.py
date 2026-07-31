"""Nó hub: roteamento semântico híbrido com fallback heurístico/LLM."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

from banco_agil.deps import AppDeps
from banco_agil.graph.state import Intent, SessionState
from banco_agil.tools.session_tools import end_conversation_update
from banco_agil.utils.conversation import (
    extract_money,
    heuristic_intent,
    last_user_text,
    looks_like_affirmative,
    looks_like_end,
)

IntentFallback = Callable[[str], str | None]


def make_router_node(
    deps: AppDeps,
    *,
    llm_fallback: IntentFallback | None = None,
):
    """Factory do nó router (hub).

    Args:
        deps: Dependências da aplicação.
        llm_fallback: Função opcional de classificação (FakeLLM / provider real).

    Returns:
        Função de nó LangGraph.
    """

    def router_node(state: SessionState) -> dict[str, Any]:
        """Decide a intenção do turno e encaminha para o skill adequado.

        Ordem: encerramento → aceite de entrevista → semântico → heurística → LLM.

        Args:
            state: Estado da sessão.

        Returns:
            Update com intent, confiança e fonte do roteamento.
        """
        text = last_user_text(state.get("messages", []))

        if looks_like_end(text):
            update = end_conversation_update()
            update.update(
                {
                    "active_agent": "router",
                    "intent": "end",
                    "route_source": "heuristic",
                    "route_confidence": 1.0,
                }
            )
            return update

        # Continuação do fluxo de aumento de limite (contexto entre turnos)
        if state.get("awaiting_limit_value") or (
            state.get("awaiting_increase_confirm")
            and (looks_like_affirmative(text) or extract_money(text) is not None)
        ):
            return {
                "active_agent": "router",
                "intent": "credit",
                "route_source": "heuristic",
                "route_confidence": 1.0,
            }

        # Após oferta de entrevista, "sim" / "quero" → interview
        if (
            state.get("offered_interview")
            and not state.get("interview_complete")
            and looks_like_affirmative(text)
        ):
            return {
                "active_agent": "router",
                "intent": "interview",
                "interview_accepted": True,
                "route_source": "heuristic",
                "route_confidence": 1.0,
            }

        route = deps.intent_router.predict(text)
        if route is not None and route.intent in {
            "credit",
            "exchange",
            "interview",
            "end",
            "unknown",
        }:
            intent: Intent = route.intent  # type: ignore[assignment]
            if intent == "unknown":
                return _clarify(intent="unknown", source="semantic", confidence=route.confidence)
            if intent == "end":
                update = end_conversation_update()
                update.update(
                    {
                        "active_agent": "router",
                        "intent": "end",
                        "route_source": "semantic",
                        "route_confidence": route.confidence,
                    }
                )
                return update
            return {
                "active_agent": "router",
                "intent": intent,
                "route_source": "semantic",
                "route_confidence": route.confidence,
            }

        heur = heuristic_intent(text)
        if heur is not None:
            if heur == "end":
                update = end_conversation_update()
                update.update(
                    {
                        "active_agent": "router",
                        "intent": "end",
                        "route_source": "heuristic",
                        "route_confidence": 0.9,
                    }
                )
                return update
            return {
                "active_agent": "router",
                "intent": heur,  # type: ignore[typeddict-item]
                "route_source": "heuristic",
                "route_confidence": 0.85,
            }

        if llm_fallback is not None:
            llm_intent = llm_fallback(text)
            if llm_intent in {"credit", "exchange", "interview", "end"}:
                if llm_intent == "end":
                    update = end_conversation_update()
                    update.update(
                        {
                            "active_agent": "router",
                            "intent": "end",
                            "route_source": "llm_fallback",
                            "route_confidence": None,
                        }
                    )
                    return update
                return {
                    "active_agent": "router",
                    "intent": llm_intent,  # type: ignore[typeddict-item]
                    "route_source": "llm_fallback",
                    "route_confidence": None,
                }

        return _clarify(intent="unknown", source="heuristic", confidence=None)

    return router_node


def _clarify(
    *,
    intent: Intent,
    source: str,
    confidence: float | None,
) -> dict[str, Any]:
    """Pede clarificação quando a intenção é desconhecida."""
    return {
        "active_agent": "router",
        "intent": intent,
        "route_source": source,  # type: ignore[typeddict-item]
        "route_confidence": confidence,
        "messages": [
            AIMessage(
                content=(
                    "Posso ajudar com consulta ou aumento de limite de crédito, "
                    "cotação de câmbio, ou encerrar o atendimento. "
                    "O que você prefere?"
                )
            )
        ],
    }
