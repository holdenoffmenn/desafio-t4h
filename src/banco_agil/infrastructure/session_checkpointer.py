"""Fábrica do checkpointer LangGraph (SQLite ou memória)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver


def build_checkpointer(db_path: Path | None = None, *, memory: bool = False) -> Any:
    """Cria um checkpointer para persistência de sessão.

    Args:
        db_path: Caminho do SQLite (ignorado se ``memory=True``).
        memory: Se True, usa MemorySaver (testes / efêmero).

    Returns:
        Instância de checkpointer compatível com ``graph.compile``.
    """
    if memory or db_path is None:
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise ImportError(
            "langgraph-checkpoint-sqlite é necessário para SqliteSaver. "
            "pip install langgraph-checkpoint-sqlite"
        ) from exc

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
