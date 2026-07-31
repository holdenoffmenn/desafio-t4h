"""Utilitário compartilhado para normalizar o ``content`` de respostas LLM."""

from __future__ import annotations


def content_to_text(content: object) -> str:
    """Normaliza o ``content`` de uma resposta do modelo para texto plano.

    Suporta string simples e a lista de blocos (``[{'type': 'text',
    'text': ...}]``) usada por provedores como Gemini/Anthropic.

    Args:
        content: Campo ``content`` da resposta do modelo.

    Returns:
        Texto concatenado dos blocos textuais.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return " ".join(parts)
    return str(content)
