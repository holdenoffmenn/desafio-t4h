"""Testes do classificador de intenção via LLM (com fake chat model)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.llm.intent import (
    IntentResult,
    LlmIntentClassifier,
    make_llm_intent_fallback,
)


class _FakeChat:
    """Chat model mínimo: devolve conteúdo fixo, conta chamadas ou levanta erro."""

    def __init__(
        self,
        reply: str | list[Any] | None = None,
        *,
        raises: bool = False,
    ) -> None:
        self._reply = reply
        self._raises = raises
        self.calls = 0

    def invoke(self, _messages: Any, **_kwargs: Any) -> AIMessage:
        self.calls += 1
        if self._raises:
            raise RuntimeError("network down")
        return AIMessage(content=self._reply if self._reply is not None else "")


def _clf(
    reply: str | list[Any] | None = None,
    *,
    raises: bool = False,
) -> LlmIntentClassifier:
    return LlmIntentClassifier(cast(BaseChatModel, _FakeChat(reply, raises=raises)))


def test_classifies_actionable_intents() -> None:
    assert _clf("credit").classify("quero aumentar meu limite") == IntentResult("credit")
    assert _clf("exchange").classify("cotação do dólar") == IntentResult("exchange")
    assert _clf("interview").classify("aceito a entrevista") == IntentResult("interview")
    assert _clf("end").classify("encerrar") == IntentResult("end")


def test_normalizes_noisy_output() -> None:
    assert _clf(" Credit. ").classify("x").intent == "credit"
    assert _clf("EXCHANGE").classify("x").intent == "exchange"
    assert _clf("A intenção é credit.").classify("x").intent == "credit"


def test_uses_conversation_context() -> None:
    """O contexto é enviado ao modelo para desambiguar respostas curtas."""
    result = _clf("credit").classify("sim", context="Assistente: Deseja aumentar o limite?")
    assert result.intent == "credit"


def test_handles_content_block_list() -> None:
    """Formato de blocos usado por Gemini/Anthropic: [{'type':'text',...}]."""
    blocks = [{"type": "text", "text": "credit"}]
    assert _clf(blocks).classify("quero limite").intent == "credit"


def test_unknown_or_invalid_returns_none_without_failure() -> None:
    assert _clf("unknown").classify("bom dia") == IntentResult(None, failed=False)
    assert _clf("qualquer coisa").classify("x") == IntentResult(None, failed=False)
    assert _clf("").classify("x") == IntentResult(None, failed=False)


def test_empty_text_returns_none() -> None:
    assert _clf("credit").classify("   ") == IntentResult(None, failed=False)


def test_error_after_retries_flags_failed() -> None:
    fake = _FakeChat(raises=True)
    result = LlmIntentClassifier(cast(BaseChatModel, fake)).classify("quero crédito")
    assert result == IntentResult(None, failed=True)
    # Retry aconteceu: mais de uma tentativa antes de desistir.
    assert fake.calls > 1


def test_make_fallback_none_when_no_model() -> None:
    assert make_llm_intent_fallback(None) is None


def test_make_fallback_wraps_model() -> None:
    fallback = make_llm_intent_fallback(cast(BaseChatModel, _FakeChat("credit")))
    assert fallback is not None
    assert fallback("quero aumentar limite", "") == IntentResult("credit")
