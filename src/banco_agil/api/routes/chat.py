"""Endpoint POST /chat — um turno de atendimento."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage

from banco_agil.api.metadata import build_metadata, build_reply
from banco_agil.api.schemas import ChatRequest, ChatResponse
from banco_agil.domain.errors import DomainError
from banco_agil.graph.state import initial_state
from banco_agil.observability.logging import get_logger

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Processa uma mensagem do usuário e retorna a resposta do assistente.

    O estado de negócio é reconstruído pelo checkpointer via ``thread_id``.
    Erros de domínio viram mensagem amigável (sem stack trace).
    Observabilidade (structlog + Langfuse) é best-effort e nunca derruba o chat.

    Args:
        payload: session_id + message.
        request: Request FastAPI (acesso ao grafo no ``app.state``).

    Returns:
        ChatResponse com reply e metadata de backoffice.

    Raises:
        HTTPException: 503 se o grafo não estiver inicializado; 500 genérico controlado.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Grafo não inicializado.")

    tracer = getattr(request.app.state, "tracer", None)
    config: dict[str, Any] = {"configurable": {"thread_id": payload.session_id}}
    try:
        snapshot = graph.get_state(config)
        if not snapshot.values:
            invoke_payload: dict[str, Any] = dict(initial_state(payload.session_id))
            invoke_payload["messages"] = [HumanMessage(content=payload.message)]
        else:
            invoke_payload = {"messages": [HumanMessage(content=payload.message)]}

        state = graph.invoke(invoke_payload, config=config)
    except DomainError as exc:
        logger.warning(
            "domain_error",
            session_id=payload.session_id,
            error=str(exc),
        )
        return ChatResponse(
            reply=(
                "Encontrei uma dificuldade ao processar sua solicitação. "
                "Pode tentar novamente em instantes?"
            ),
            session_id=payload.session_id,
            metadata=build_metadata(
                {
                    "error": str(exc),
                    "active_agent": "router",
                    "authenticated": False,
                }
            ),
        )
    except Exception as exc:  # noqa: BLE001 — fronteira HTTP: nunca vazar stack
        logger.error(
            "chat_unhandled_error",
            session_id=payload.session_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar o atendimento.",
        ) from exc

    trace_url: str | None = None
    if tracer is not None:
        try:
            result = tracer.record_turn(session_id=payload.session_id, state=state)
            trace_url = result.trace_url
        except Exception as exc:  # noqa: BLE001
            logger.warning("tracer_failed", error=str(exc), session_id=payload.session_id)

    if trace_url:
        state = {**state, "langfuse_trace_url": trace_url}

    return ChatResponse(
        reply=build_reply(
            state,
            fallback="Desculpe, não consegui formular uma resposta agora.",
        ),
        session_id=payload.session_id,
        metadata=build_metadata(state),
    )
