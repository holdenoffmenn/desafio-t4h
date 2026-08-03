"""Ponte entre os nós do grafo e o compositor de mensagens (LLM opcional)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from banco_agil.llm.composer import MessageComposer, MessageSpec


def render(composer: MessageComposer | None, spec: MessageSpec) -> str:
    """Redige a mensagem via LLM, degradando para o texto canônico da spec.

    Args:
        composer: Compositor LLM ou ``None`` (modo determinístico).
        spec: Briefing estruturado da mensagem.

    Returns:
        Texto redigido pelo LLM ou o ``fallback`` canônico.
    """
    if composer is None:
        return spec.fallback
    return composer.compose(spec)


def speak(composer: MessageComposer | None, spec: MessageSpec) -> list[AIMessage]:
    """Atalho para produzir a lista ``messages`` de um update de nó.

    Args:
        composer: Compositor LLM ou ``None``.
        spec: Briefing estruturado da mensagem.

    Returns:
        Lista com uma ``AIMessage`` pronta para o estado do grafo.
    """
    return [AIMessage(content=render(composer, spec))]
