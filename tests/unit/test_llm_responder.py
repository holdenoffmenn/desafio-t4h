"""Testes da camada de voz (LlmResponder) com fake chat model."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.llm.responder import LlmResponder, make_llm_responder


class _FakeChat:
    """Chat model mínimo: devolve conteúdo fixo ou levanta exceção."""

    def __init__(self, reply: str | list[Any] | None = None, *, raises: bool = False) -> None:
        self._reply = reply
        self._raises = raises

    def invoke(self, _messages: Any, **_kwargs: Any) -> AIMessage:
        if self._raises:
            raise RuntimeError("network down")
        return AIMessage(content=self._reply if self._reply is not None else "")


def _resp(reply: str | list[Any] | None = None, *, raises: bool = False) -> LlmResponder:
    return LlmResponder(cast(BaseChatModel, _FakeChat(reply, raises=raises)))


def test_humanizes_conversational_message() -> None:
    original = "Informe seu CPF, por favor."
    rewritten = "Claro! Para começarmos, você poderia me informar o seu CPF?"
    assert _resp(rewritten).humanize(original) == rewritten


def test_preserves_facts_accepts_when_numbers_kept() -> None:
    original = "Seu limite atual é R$ 3.000,00 e seu score é 564."
    rewritten = "Consultei aqui: seu limite hoje é de R$ 3.000,00 e o score está em 564."
    assert _resp(rewritten).humanize(original) == rewritten


def test_falls_back_when_number_is_dropped() -> None:
    original = "Seu limite atual é R$ 3.000,00."
    # "três mil" não contém os dígitos 300000 -> guarda rejeita a reescrita.
    rewritten = "Seu limite atual é de três mil reais."
    assert _resp(rewritten).humanize(original) == original


def test_falls_back_when_number_is_altered() -> None:
    original = "Sua solicitação de R$ 7.000,00 foi rejeitada (score_insuficiente)."
    rewritten = "Sua solicitação de R$ 8.000,00 foi rejeitada."
    assert _resp(rewritten).humanize(original) == original


def test_handles_content_block_list() -> None:
    original = "Bom dia."
    blocks = [{"type": "text", "text": "Bom dia! Como posso ajudar?"}]
    assert _resp(blocks).humanize(original) == "Bom dia! Como posso ajudar?"


def test_empty_rewrite_falls_back() -> None:
    original = "Olá."
    assert _resp("   ").humanize(original) == original


def test_error_degrades_to_original() -> None:
    original = "Olá, tudo bem?"
    assert _resp(raises=True).humanize(original) == original


def test_empty_text_returns_as_is() -> None:
    assert _resp("qualquer coisa").humanize("   ") == "   "


def test_make_responder_none_when_no_model() -> None:
    assert make_llm_responder(None) is None


def test_make_responder_wraps_model() -> None:
    responder = make_llm_responder(cast(BaseChatModel, _FakeChat("Oi!")))
    assert responder is not None
    assert responder.humanize("Olá.") == "Oi!"
