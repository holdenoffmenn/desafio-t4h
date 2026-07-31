"""Endpoint POST /chat — um turno de atendimento."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage

from banco_agil.api.metadata import build_metadata, build_reply
from banco_agil.api.schemas import ChatRequest, ChatResponse
from banco_agil.domain.errors import DomainError
from banco_agil.graph.state import initial_state

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Processa uma mensagem do usuário e retorna a resposta do assistente.

    O estado de negócio é reconstruído pelo checkpointer via ``thread_id``.
    Erros de domínio viram mensagem amigável (sem stack trace).

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
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar o atendimento.",
        ) from exc

    return ChatResponse(
        reply=build_reply(
            state,
            fallback="Desculpe, não consegui formular uma resposta agora.",
        ),
        session_id=payload.session_id,
        metadata=build_metadata(state),
    )
