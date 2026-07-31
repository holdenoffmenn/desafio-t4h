"""Endpoint GET /session/{id} — snapshot para Backoffice."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from banco_agil.api.metadata import build_metadata
from banco_agil.api.schemas import ChatMetadata, SessionSnapshot

router = APIRouter(tags=["session"])


@router.get("/session/{session_id}", response_model=SessionSnapshot)
def get_session(session_id: str, request: Request) -> SessionSnapshot:
    """Retorna snapshot do estado da sessão no checkpointer.

    Args:
        session_id: Identificador da sessão.
        request: Request FastAPI.

    Returns:
        SessionSnapshot com metadata e contagem de mensagens.

    Raises:
        HTTPException: 503 se o grafo não estiver inicializado.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Grafo não inicializado.")

    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    exists = bool(values)
    messages = values.get("messages") or []
    metadata = build_metadata(values) if exists else ChatMetadata()

    return SessionSnapshot(
        session_id=session_id,
        exists=exists,
        metadata=metadata,
        message_count=len(messages),
    )
