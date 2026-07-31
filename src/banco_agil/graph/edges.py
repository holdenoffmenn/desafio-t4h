"""Funções puras de roteamento condicional (testáveis sem LLM)."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END

from banco_agil.graph.state import SessionState

AfterGuard = Literal["safe_reply", "triage", "router"]
AfterTriage = Literal["end", "router"] | object
AfterRouter = Literal["credit", "exchange", "interview", "end"] | object
AfterCredit = Literal["interview", "end"] | object
AfterInterview = Literal["credit", "end"] | object
AfterExchange = Literal["end"] | object


def route_after_guard(state: SessionState) -> AfterGuard:
    """Roteia após o nó de segurança.

    Args:
        state: Estado atual da sessão.

    Returns:
        ``safe_reply`` se bloqueado; ``triage`` se não autenticado; senão ``router``.
    """
    if state.get("input_blocked"):
        return "safe_reply"
    if not state.get("authenticated"):
        return "triage"
    return "router"


def route_after_triage(state: SessionState) -> AfterTriage:
    """Roteia após a triagem/autenticação.

    Encerra o turno com ``END`` quando ainda falta input do usuário.
    Após 3 falhas de autenticação, vai para ``end``.

    Args:
        state: Estado atual da sessão.

    Returns:
        Nome do próximo nó ou ``END``.
    """
    if state.get("should_end"):
        return "end"
    if not state.get("authenticated"):
        if state.get("auth_attempts", 0) >= 3:
            return "end"
        return END
    return "router"


def route_after_router(state: SessionState) -> AfterRouter:
    """Roteia a partir do hub de intenções.

    Args:
        state: Estado atual da sessão.

    Returns:
        Skill correspondente ou ``END`` se intenção desconhecida.
    """
    if state.get("should_end"):
        return "end"
    match state.get("intent"):
        case "credit":
            return "credit"
        case "exchange":
            return "exchange"
        case "interview":
            return "interview"
        case "end":
            return "end"
        case _:
            return END


def route_after_credit(state: SessionState) -> AfterCredit:
    """Roteia após o skill de crédito.

    Se o cliente aceitou a entrevista neste turno, segue para ``interview``.
    Caso contrário encerra o turno (``END``) — o hub é retomado no próximo
    ``invoke`` via ``guard → router``.

    Args:
        state: Estado atual da sessão.

    Returns:
        ``interview``, ``end`` ou ``END``.
    """
    if state.get("should_end"):
        return "end"
    if state.get("interview_accepted"):
        return "interview"
    return END


def route_after_interview(state: SessionState) -> AfterInterview:
    """Roteia após a entrevista financeira.

    Com entrevista completa, reanalisa no crédito no mesmo turno.
    Caso contrário aguarda mais dados (``END``).

    Args:
        state: Estado atual da sessão.

    Returns:
        ``credit``, ``end`` ou ``END``.
    """
    if state.get("should_end"):
        return "end"
    if state.get("interview_complete"):
        return "credit"
    return END


def route_after_exchange(state: SessionState) -> AfterExchange:
    """Roteia após o skill de câmbio.

    Args:
        state: Estado atual da sessão.

    Returns:
        ``end`` se solicitado; senão ``END`` (próximo turno volta ao hub).
    """
    if state.get("should_end"):
        return "end"
    return END
