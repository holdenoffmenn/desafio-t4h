"""Estado tipado da sessão conversacional (LangGraph)."""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

ActiveAgent = Literal[
    "guard",
    "triage",
    "router",
    "credit",
    "interview",
    "exchange",
    "safe_reply",
    "end",
]
Intent = Literal["credit", "exchange", "interview", "end", "unknown"]
RequestStatus = Literal["pendente", "aprovado", "rejeitado"]
RouteSource = Literal["semantic", "llm_fallback", "context", "error"]


class SessionState(TypedDict):
    """Estado completo de uma sessão de atendimento.

    O reducer ``add_messages`` acumula o histórico. Campos NotRequired
    permitem invokes parciais (só a nova mensagem do usuário).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    session_id: NotRequired[str]

    # Autenticação
    cpf: str | None
    authenticated: bool
    auth_attempts: int
    customer: dict[str, object] | None

    # Roteamento
    active_agent: ActiveAgent
    intent: Intent | None
    route_confidence: float | None
    route_source: RouteSource | None

    # Segurança
    input_blocked: bool
    safety_label: str | None
    safety_score: float | None

    # Crédito
    pending_new_limit: float | None
    last_request_id: str | None
    last_request_status: RequestStatus | None
    offered_interview: bool
    interview_accepted: bool
    awaiting_increase_confirm: bool
    awaiting_limit_value: bool

    # Entrevista
    interview_data: dict[str, object] | None
    interview_complete: bool
    awaiting_interview: bool

    # Clarificação (re-pergunta 1x → erro) na coleta de valores/campos
    clarify_attempts: int

    # Controle
    should_end: bool
    error: str | None

    # Observabilidade
    last_tool_calls: list[dict[str, object]]
    last_score_calculation: dict[str, object] | None
    langfuse_trace_url: str | None


def initial_state(session_id: str = "") -> SessionState:
    """Cria o estado inicial de uma sessão.

    Args:
        session_id: Identificador da sessão / thread_id.

    Returns:
        SessionState zerado pronto para o primeiro turno.
    """
    return {
        "messages": [],
        "session_id": session_id,
        "cpf": None,
        "authenticated": False,
        "auth_attempts": 0,
        "customer": None,
        "active_agent": "guard",
        "intent": None,
        "route_confidence": None,
        "route_source": None,
        "input_blocked": False,
        "safety_label": None,
        "safety_score": None,
        "pending_new_limit": None,
        "last_request_id": None,
        "last_request_status": None,
        "offered_interview": False,
        "interview_accepted": False,
        "awaiting_increase_confirm": False,
        "awaiting_limit_value": False,
        "interview_data": None,
        "interview_complete": False,
        "awaiting_interview": False,
        "clarify_attempts": 0,
        "should_end": False,
        "error": None,
        "last_tool_calls": [],
        "last_score_calculation": None,
        "langfuse_trace_url": None,
    }
