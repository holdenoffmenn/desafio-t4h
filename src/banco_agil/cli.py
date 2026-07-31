"""CLI de validação manual do fluxo multi-turno (Parte 2)."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from banco_agil.deps import build_deps
from banco_agil.graph.state import initial_state
from banco_agil.graph.workflow import build_graph, last_ai_text
from banco_agil.llm import (
    build_chat_model,
    make_llm_extractor,
    make_llm_intent_fallback,
    make_llm_responder,
)


def run_cli() -> None:
    """Loop interativo no terminal para validar o grafo.

    Usa o LLM configurado no ``.env`` como fallback de intenção quando
    disponível; caso contrário, opera com roteamento semântico + heurística.

    Controles:
        ``/sair`` — encerra o CLI
        ``/novo`` — reinicia a sessão
    """
    deps = build_deps()
    chat_model = build_chat_model(deps.settings)
    llm_fallback = make_llm_intent_fallback(chat_model)
    deps.nlu = make_llm_extractor(chat_model)
    responder = make_llm_responder(chat_model)
    graph = build_graph(
        deps,
        use_memory_checkpointer=False,
        llm_fallback=llm_fallback,
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
        if responder is not None and agent != "safe_reply":
            reply = responder.humanize(reply)
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
