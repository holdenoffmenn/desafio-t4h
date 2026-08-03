"""DTOs Pydantic da API de chat."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload de um turno de conversa.

    Attributes:
        session_id: Identificador da sessão (thread_id do checkpointer).
        message: Texto enviado pelo usuário.
    """

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RouteMeta(BaseModel):
    """Metadados do roteamento de intenção."""

    source: Literal["semantic", "llm_fallback", "context", "error"] | None = None
    confidence: float | None = None


class SafetyMeta(BaseModel):
    """Metadados da camada de segurança."""

    blocked: bool = False
    label: str | None = None
    score: float | None = None


class ChatMetadata(BaseModel):
    """Metadados técnicos para a aba Backoffice.

    Attributes:
        active_agent: Último nó ativo do grafo.
        authenticated: Se o cliente está autenticado.
        intent: Intenção classificada no turno.
        route: Fonte e confiança do roteamento.
        safety: Resultado do guard.
        last_tool_calls: Tools executadas (args mascarados).
        last_score_calculation: Breakdown do score, se houver.
        langfuse_trace_url: URL do trace (Parte 4).
        auth_attempts: Tentativas de autenticação consumidas.
        last_request_status: Status do último pedido de aumento.
        should_end: Se a sessão foi encerrada.
    """

    active_agent: str | None = None
    authenticated: bool = False
    intent: str | None = None
    route: RouteMeta = Field(default_factory=RouteMeta)
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    last_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    last_score_calculation: dict[str, Any] | None = None
    langfuse_trace_url: str | None = None
    auth_attempts: int = 0
    last_request_status: str | None = None
    should_end: bool = False


class ChatResponse(BaseModel):
    """Resposta de um turno de conversa."""

    reply: str
    session_id: str
    metadata: ChatMetadata


class SessionSnapshot(BaseModel):
    """Snapshot do estado para GET /session/{id}."""

    session_id: str
    exists: bool
    metadata: ChatMetadata
    message_count: int = 0


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: Literal["ok"] = "ok"
    service: str = "banco-agil"
