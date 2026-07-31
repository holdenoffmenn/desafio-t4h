"""CLI de validação manual do fluxo multi-turno (Parte 2)."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from banco_agil.deps import build_deps
from banco_agil.graph.state import initial_state
from banco_agil.graph.workflow import build_graph, last_ai_text


def _fake_llm_fallback(text: str) -> str | None:
    """Fallback determinístico usado quando o roteador semântico falha.

    Args:
        text: Mensagem do usuário.

    Returns:
        Intent ou None.
    """
    from banco_agil.utils.conversation import heuristic_intent

    return heuristic_intent(text)


def run_cli() -> None:
    """Loop interativo no terminal para validar o grafo.

    Controles:
        ``/sair`` — encerra o CLI
        ``/novo`` — reinicia a sessão
    """
    deps = build_deps()
    graph = build_graph(
        deps,
        use_memory_checkpointer=False,
        llm_fallback=_fake_llm_fallback,
    )
    session_id = str(uuid.uuid4())
    print("Banco Ágil — CLI (Parte 2). Digite /sair para encerrar, /novo para nova sessão.")
    print(f"session_id={session_id}\n")

    first = True
    while True:
        try:
            text = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrado.")
            break

        if not text:
            continue
        if text in {"/sair", "/exit", "/quit"}:
            print("Até logo!")
            break
        if text == "/novo":
            session_id = str(uuid.uuid4())
            first = True
            print(f"Nova sessão: {session_id}\n")
            continue

        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        if first:
            payload: dict[str, Any] = dict(initial_state(session_id))
            payload["messages"] = [HumanMessage(content=text)]
            first = False
        else:
            payload = {"messages": [HumanMessage(content=text)]}

        state = graph.invoke(payload, config=config)
        reply = last_ai_text(state)
        agent = state.get("active_agent")
        print(f"agente[{agent}]> {reply}\n")

        if state.get("should_end"):
            print("(sessão encerrada — use /novo para recomeçar)\n")
            first = True
            session_id = str(uuid.uuid4())


def main() -> None:
    """Entry point ``python -m banco_agil.cli``."""
    run_cli()


if __name__ == "__main__":
    main()
