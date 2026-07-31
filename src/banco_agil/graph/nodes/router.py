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

        Guardas determinísticas de continuidade (encerramento, entrevista/valor
        em andamento, aceite de entrevista) têm prioridade. Em seguida, quando há
        LLM configurado, ela é o **intérprete primário** da intenção; o roteador
        semântico e a heurística de palavras-chave são a rede de segurança
        (usados quando não há LLM ou quando a LLM não tem confiança).

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

        # Entrevista em andamento: mantém o cliente no fluxo até completar.
        # Sem esta guarda, respostas como "8000" ou "formal" cairiam em intent
        # desconhecida e a entrevista nunca seria concluída (bug de continuidade).
        if state.get("awaiting_interview") and not state.get("interview_complete"):
            return {
                "active_agent": "router",
                "intent": "interview",
                "route_source": "heuristic",
                "route_confidence": 1.0,
            }

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

        # LLM como intérprete primário da intenção (quando configurado).
        if llm_fallback is not None:
            llm_intent = llm_fallback(text)
            if llm_intent in _ACTIONABLE:
                return _routed(llm_intent, source="llm_fallback", confidence=None)

        # Rede de segurança 1: roteador semântico (embeddings). Só aceitamos
        # intenções acionáveis; "unknown" cai para a heurística/clarificação.
        route = deps.intent_router.predict(text)
        if route is not None and route.intent in _ACTIONABLE:
            return _routed(route.intent, source="semantic", confidence=route.confidence)

        # Rede de segurança 2: heurística de palavras-chave.
        heur = heuristic_intent(text)
        if heur is not None:
            return _routed(heur, source="heuristic", confidence=0.85)

        return _clarify(intent="unknown", source="heuristic", confidence=None)

    return router_node


_ACTIONABLE: frozenset[str] = frozenset({"credit", "exchange", "interview", "end"})


def _routed(intent: str, *, source: str, confidence: float | None) -> dict[str, Any]:
    """Monta o update de roteamento, tratando ``end`` de forma especial.

    Args:
        intent: Intenção acionável (``credit`` | ``exchange`` | ``interview`` | ``end``).
        source: Fonte do roteamento para observabilidade.
        confidence: Confiança quando aplicável (``None`` para LLM/heurística fixa).

    Returns:
        Update de estado com a intenção resolvida.
    """
    if intent == "end":
        update = end_conversation_update()
        update.update(
            {
                "active_agent": "router",
                "intent": "end",
                "route_source": source,  # type: ignore[typeddict-item]
                "route_confidence": confidence,
            }
        )
        return update
    return {
        "active_agent": "router",
        "intent": intent,  # type: ignore[typeddict-item]
        "route_source": source,  # type: ignore[typeddict-item]
        "route_confidence": confidence,
    }


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
                    "Posso te ajudar com:\n"
                    "- **Limite de crédito** — consultar seu limite atual ou "
                    "solicitar um aumento\n"
                    "- **Câmbio** — cotação de moedas (dólar, euro, etc.)\n"
                    "- **Encerrar** o atendimento\n\n"
                    "Sobre qual desses assuntos você gostaria de falar?"
                )
            )
        ],
    }
