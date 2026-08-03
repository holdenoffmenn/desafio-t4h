"""Classificador de intenção via LLM (intérprete primário do roteamento).

Segue o princípio "LLM conversa, código decide": o LLM apenas **entende** a
intenção do cliente em linguagem natural (com o contexto recente da conversa);
toda decisão financeira permanece no código determinístico.

Diferente de um simples ``str | None``, o classificador devolve um
``IntentResult`` que distingue dois cenários que exigem tratamento diferente no
router:

- ``intent`` preenchido → intenção acionável reconhecida;
- ``intent=None`` sem ``failed`` → a LLM respondeu mas a mensagem é ambígua;
- ``failed=True`` → a LLM falhou após as tentativas (rede/quota) e o router
  deve devolver uma mensagem de erro amigável, não silenciar a falha.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from banco_agil.llm._content import content_to_text
from banco_agil.llm._retry import invoke_with_retry
from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)


@dataclass(frozen=True)
class IntentResult:
    """Resultado da classificação de intenção.

    Attributes:
        intent: Intenção acionável (``credit`` | ``exchange`` | ``interview`` |
            ``end``) ou ``None`` quando ambígua/desconhecida.
        failed: ``True`` se a LLM falhou após todas as tentativas (o router deve
            responder com erro amigável em vez de degradar silenciosamente).
    """

    intent: str | None = None
    failed: bool = False


# Assinatura do fallback do router: recebe a mensagem e o contexto recente.
IntentFallback = Callable[[str, str], IntentResult]

_ACTIONABLE: frozenset[str] = frozenset({"credit", "exchange", "interview", "end"})

_SYSTEM_PROMPT = (
    "Você classifica a intenção da ÚLTIMA mensagem do cliente de um banco "
    "digital, usando o CONTEXTO da conversa para desambiguar respostas curtas "
    "(ex.: 'sim', 'o primeiro', 'quero esse'). "
    "Responda APENAS com uma destas palavras, sem pontuação nem explicações: "
    "credit, exchange, interview, end, unknown.\n"
    "- credit: consultar limite, aumentar limite de crédito, cartão, ou "
    "confirmar/prosseguir com uma solicitação de crédito em andamento.\n"
    "- exchange: cotação de moedas / câmbio (dólar, euro, etc.).\n"
    "- interview: aceitar ou conduzir a entrevista financeira / informar renda.\n"
    "- end: encerrar ou finalizar todo o atendimento.\n"
    "- unknown: saudação, assunto fora de escopo, ou mensagem que não dá para "
    "classificar com segurança mesmo com o contexto."
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

    def classify(self, text: str, context: str = "") -> IntentResult:
        """Classifica ``text`` em uma intenção acionável.

        Args:
            text: Última mensagem do usuário.
            context: Transcrição recente da conversa (para desambiguar).

        Returns:
            ``IntentResult`` com a intenção acionável, ``intent=None`` quando
            ambíguo, ou ``failed=True`` quando a LLM falhou após as tentativas.
        """
        if not text.strip():
            return IntentResult(intent=None)

        from langchain_core.messages import HumanMessage, SystemMessage

        user_content = f"{context}\n\nÚLTIMA MENSAGEM: {text}" if context else text
        try:
            response = invoke_with_retry(
                self._model,
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)],
                event="llm_intent_failed",
            )
        # Esgotadas as tentativas: sinaliza falha para o router responder com erro.
        except Exception:  # noqa: BLE001
            return IntentResult(intent=None, failed=True)

        raw = content_to_text(response.content).lower()
        # Aceita respostas ruidosas ("A intenção é credit.") pegando a
        # primeira palavra que casa com uma intenção acionável.
        for word in re.findall(r"[a-zà-ú]+", raw):
            if word in _ACTIONABLE:
                return IntentResult(intent=word)
        return IntentResult(intent=None)


def make_llm_intent_fallback(model: BaseChatModel | None) -> IntentFallback | None:
    """Cria a função de fallback de intenção a partir de um chat model.

    Args:
        model: Chat model ou ``None`` (LLM desligado).

    Returns:
        Callable compatível com ``make_router_node`` ou ``None`` se não houver
        modelo (o router seguirá apenas com o roteador semântico).
    """
    if model is None:
        return None
    classifier = LlmIntentClassifier(model)
    return classifier.classify
