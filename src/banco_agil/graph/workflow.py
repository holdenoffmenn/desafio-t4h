"""Montagem do grafo LangGraph (state machine + checkpointer)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from banco_agil.deps import AppDeps, build_deps
from banco_agil.graph.edges import (
    route_after_credit,
    route_after_exchange,
    route_after_guard,
    route_after_interview,
    route_after_router,
    route_after_triage,
)
from banco_agil.graph.nodes.credit import make_credit_node
from banco_agil.graph.nodes.end import end_node
from banco_agil.graph.nodes.exchange import make_exchange_node
from banco_agil.graph.nodes.guard import make_guard_node
from banco_agil.graph.nodes.interview import make_interview_node
from banco_agil.graph.nodes.router import IntentFallback, make_router_node
from banco_agil.graph.nodes.safe_reply import safe_reply_node
from banco_agil.graph.nodes.triage import make_triage_node
from banco_agil.graph.state import SessionState
from banco_agil.infrastructure.session_checkpointer import build_checkpointer


def build_graph(
    deps: AppDeps | None = None,
    *,
    checkpointer: Any | None = None,
    llm_fallback: IntentFallback | None = None,
    use_memory_checkpointer: bool = False,
) -> Any:
    """Compila o workflow conversacional do Banco Ágil.

    Modelo de turnos:
        Cada ``invoke`` processa uma mensagem do usuário e termina em ``END``
        (fim do turno) ou no nó ``end`` (fim da sessão). O checkpointer
        persiste o estado por ``thread_id``; o próximo ``invoke`` resume.

    Args:
        deps: Container de dependências (default: ``build_deps()``).
        checkpointer: Checkpointer explícito (MemorySaver/SqliteSaver).
        llm_fallback: Classificador opcional para o router.
        use_memory_checkpointer: Força MemorySaver se checkpointer for None.

    Returns:
        Grafo compilado pronto para ``invoke`` / ``ainvoke``.
    """
    app_deps = deps or build_deps()
    saver = checkpointer
    if saver is None:
        if use_memory_checkpointer:
            saver = build_checkpointer(memory=True)
        else:
            saver = build_checkpointer(app_deps.settings.checkpointer_db)

    graph: StateGraph[SessionState] = StateGraph(SessionState)

    graph.add_node("guard", make_guard_node(app_deps))
    graph.add_node("triage", make_triage_node(app_deps))
    graph.add_node("router", make_router_node(app_deps, llm_fallback=llm_fallback))
    graph.add_node("credit", make_credit_node(app_deps))
    graph.add_node("interview", make_interview_node(app_deps))
    graph.add_node("exchange", make_exchange_node(app_deps))
    graph.add_node("safe_reply", safe_reply_node)
    graph.add_node("end", end_node)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard)
    graph.add_conditional_edges("triage", route_after_triage)
    graph.add_conditional_edges("router", route_after_router)
    graph.add_conditional_edges("credit", route_after_credit)
    graph.add_conditional_edges("interview", route_after_interview)
    graph.add_conditional_edges("exchange", route_after_exchange)
    graph.add_edge("safe_reply", END)
    graph.add_edge("end", END)

    return graph.compile(checkpointer=saver)


def invoke_turn(
    graph: Any,
    *,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    """Executa um turno de conversa (cria estado inicial se necessário).

    Args:
        graph: Grafo compilado.
        session_id: Identificador da sessão (thread_id).
        message: Texto do usuário.

    Returns:
        Estado completo após o turno.
    """
    from langchain_core.messages import HumanMessage

    from banco_agil.graph.state import initial_state

    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        payload: dict[str, Any] = dict(initial_state(session_id))
        payload["messages"] = [HumanMessage(content=message)]
    else:
        payload = {"messages": [HumanMessage(content=message)]}

    result: dict[str, Any] = graph.invoke(payload, config=config)
    return result


def last_ai_text(state: dict[str, Any]) -> str:
    """Extrai o texto da última mensagem da IA.

    Args:
        state: Estado retornado pelo grafo.

    Returns:
        Conteúdo textual (ou string vazia).
    """
    messages = state.get("messages") or []
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""
