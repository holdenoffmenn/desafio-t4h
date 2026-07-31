"""Entrypoint FastAPI do Banco Ágil."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from banco_agil import __version__
from banco_agil.api.routes import chat, health, session
from banco_agil.deps import build_deps
from banco_agil.graph.workflow import build_graph
from banco_agil.utils.conversation import heuristic_intent


def _llm_fallback(text: str) -> str | None:
    """Fallback determinístico de intenção (sem LLM real)."""
    return heuristic_intent(text)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Inicializa o grafo (singleton) no startup da API.

    Args:
        app: Instância FastAPI.

    Yields:
        Controle ao servidor após o grafo pronto.
    """
    deps = build_deps()
    # FX mock por padrão em demo local se configurado
    app.state.deps = deps
    app.state.graph = build_graph(deps, llm_fallback=_llm_fallback)
    yield


def create_app() -> FastAPI:
    """Factory da aplicação FastAPI.

    Returns:
        App com rotas de health, chat e session.
    """
    app = FastAPI(
        title="Banco Ágil API",
        description="API de atendimento bancário com agentes LangGraph.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(session.router)
    return app


app: FastAPI = create_app()


def get_graph() -> Any:
    """Acesso ao grafo (útil em testes após override de lifespan)."""
    return app.state.graph
