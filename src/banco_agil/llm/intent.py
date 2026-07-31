"""Classificador de intenção via LLM (fallback do roteador semântico).

Segue o princípio "LLM conversa, código decide": o LLM apenas **entende** a
intenção do cliente em linguagem natural; toda decisão financeira permanece no
código determinístico. Só é acionado quando o roteador semântico e a heurística
não têm confiança suficiente.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

IntentFallback = Callable[[str], str | None]

_ACTIONABLE: frozenset[str] = frozenset({"credit", "exchange", "interview", "end"})

_SYSTEM_PROMPT = (
    "Você classifica a intenção de mensagens de clientes de um banco digital. "
    "Responda APENAS com uma destas palavras, sem pontuação nem explicações: "
    "credit, exchange, interview, end, unknown.\n"
    "- credit: consultar ou aumentar limite de crédito / cartão.\n"
    "- exchange: cotação de moedas / câmbio (dólar, euro, etc.).\n"
    "- interview: aceitar ou conduzir a entrevista financeira / informar renda.\n"
    "- end: encerrar ou finalizar o atendimento.\n"
    "- unknown: qualquer outro assunto, saudação ou mensagem ambígua."
)


class LlmIntentClassifier:
    """Classifica a intenção do cliente usando um ``BaseChatModel``.

    Colaboradores:
        model: Qualquer chat model LangChain (Gemini, Groq, OpenAI, ...).
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Inicializa o classificador.

        Args:
            model: Chat model já instanciado pela factory.
        """
        self._model = model

    def classify(self, text: str) -> str | None:
        """Classifica ``text`` em uma intenção acionável.

        Args:
            text: Mensagem do usuário.

        Returns:
            ``credit`` | ``exchange`` | ``interview`` | ``end`` quando
            confiante; ``None`` para ambíguo/erro (o router pede clarificação).
        """
        if not text.strip():
            return None

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self._model.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=text)]
            )
        # Falha de rede/quota/timeout não pode derrubar o turno.
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_intent_failed", error=str(exc))
            return None

        raw = _content_to_text(response.content).lower()
        # Aceita respostas ruidosas ("A intenção é credit.") pegando a
        # primeira palavra que casa com uma intenção acionável.
        for word in re.findall(r"[a-zà-ú]+", raw):
            if word in _ACTIONABLE:
                return word
        return None


def _content_to_text(content: Any) -> str:
    """Normaliza o ``content`` de uma AIMessage para texto plano.

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
        for block in content:  # type: ignore[union-attr]
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return str(content)


def make_llm_intent_fallback(model: BaseChatModel | None) -> IntentFallback | None:
    """Cria a função de fallback de intenção a partir de um chat model.

    Args:
        model: Chat model ou ``None`` (LLM desligado).

    Returns:
        Callable compatível com ``make_router_node`` ou ``None`` se não houver
        modelo (o router seguirá apenas com semântico + heurística).
    """
    if model is None:
        return None
    return LlmIntentClassifier(model).classify
