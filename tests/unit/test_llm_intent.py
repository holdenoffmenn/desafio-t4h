"""Testes do classificador de intenção via LLM (com fake chat model)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.llm.intent import LlmIntentClassifier, make_llm_intent_fallback


class _FakeChat:
    """Chat model mínimo: devolve conteúdo fixo ou levanta exceção."""

    def __init__(
        self,
        reply: str | list[Any] | None = None,
        *,
        raises: bool = False,
    ) -> None:
        self._reply = reply
        self._raises = raises

    def invoke(self, _messages: Any, **_kwargs: Any) -> AIMessage:
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
    assert _clf("credit").classify("quero aumentar meu limite") == "credit"
    assert _clf("exchange").classify("cotação do dólar") == "exchange"
    assert _clf("interview").classify("aceito a entrevista") == "interview"
    assert _clf("end").classify("encerrar") == "end"


def test_normalizes_noisy_output() -> None:
    assert _clf(" Credit. ").classify("x") == "credit"
    assert _clf("EXCHANGE").classify("x") == "exchange"
    assert _clf("A intenção é credit.").classify("x") == "credit"


def test_handles_content_block_list() -> None:
    """Formato de blocos usado por Gemini/Anthropic: [{'type':'text',...}]."""
    blocks = [{"type": "text", "text": "credit"}]
    assert _clf(blocks).classify("quero limite") == "credit"


def test_unknown_or_invalid_returns_none() -> None:
    assert _clf("unknown").classify("bom dia") is None
    assert _clf("qualquer coisa").classify("x") is None
    assert _clf("").classify("x") is None


def test_empty_text_returns_none() -> None:
    assert _clf("credit").classify("   ") is None


def test_error_degrades_to_none() -> None:
    assert _clf(raises=True).classify("quero crédito") is None


def test_make_fallback_none_when_no_model() -> None:
    assert make_llm_intent_fallback(None) is None


def test_make_fallback_wraps_model() -> None:
    fallback = make_llm_intent_fallback(cast(BaseChatModel, _FakeChat("credit")))
    assert fallback is not None
    assert fallback("quero aumentar limite") == "credit"
