"""Montagem de ChatMetadata a partir do estado do grafo."""

from __future__ import annotations

from typing import Any

from banco_agil.api.schemas import ChatMetadata, RouteMeta, SafetyMeta
from banco_agil.graph.workflow import last_ai_text


def build_metadata(state: dict[str, Any]) -> ChatMetadata:
    """Converte SessionState/dict do grafo em metadata da API.

    Args:
        state: Estado retornado pelo ``graph.invoke``.

    Returns:
        ChatMetadata tipado para Backoffice / response.
    """
    route_source = state.get("route_source")
    return ChatMetadata(
        active_agent=state.get("active_agent"),
        authenticated=bool(state.get("authenticated")),
        intent=state.get("intent"),
        route=RouteMeta(
            source=route_source,  # type: ignore[arg-type]
            confidence=state.get("route_confidence"),
        ),
        safety=SafetyMeta(
            blocked=bool(state.get("input_blocked")),
            label=state.get("safety_label"),
            score=state.get("safety_score"),
        ),
        last_tool_calls=list(state.get("last_tool_calls") or []),
        last_score_calculation=state.get("last_score_calculation"),
        langfuse_trace_url=state.get("langfuse_trace_url"),
        auth_attempts=int(state.get("auth_attempts") or 0),
        last_request_status=state.get("last_request_status"),
        should_end=bool(state.get("should_end")),
    )


def build_reply(state: dict[str, Any], fallback: str = "") -> str:
    """Extrai a resposta ao cliente do estado.

    Args:
        state: Estado do grafo.
        fallback: Texto se não houver AIMessage.

    Returns:
        Conteúdo da última mensagem da IA.
    """
    text = last_ai_text(state)
    return text or fallback
