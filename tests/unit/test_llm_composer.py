"""Testes do compositor de mensagens (MessageComposer) com fake chat model."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.llm.composer import MessageComposer, MessageSpec, make_message_composer


class _FakeChat:
    """Chat model mínimo: devolve conteúdo fixo, conta chamadas ou levanta erro."""

    def __init__(self, reply: str | list[Any] | None = None, *, raises: bool = False) -> None:
        self._reply = reply
        self._raises = raises
        self.calls = 0

    def invoke(self, _messages: Any, **_kwargs: Any) -> AIMessage:
        self.calls += 1
        if self._raises:
            raise RuntimeError("network down")
        return AIMessage(content=self._reply if self._reply is not None else "")


def _composer(reply: str | list[Any] | None = None, *, raises: bool = False) -> MessageComposer:
    return MessageComposer(cast(BaseChatModel, _FakeChat(reply, raises=raises)))


def _spec(**overrides: Any) -> MessageSpec:
    base: dict[str, Any] = {
        "goal": "informar o limite disponível",
        "fallback": "Seu limite é R$ 3.000,00.",
        "facts": {"limite": "R$ 3.000,00"},
    }
    base.update(overrides)
    return MessageSpec(**base)


def test_composes_when_facts_preserved() -> None:
    rewritten = "Consultei aqui: você tem R$ 3.000,00 de limite disponível hoje. Posso ajudar?"
    assert _composer(rewritten).compose(_spec()) == rewritten


def test_fallback_when_number_dropped() -> None:
    # "três mil" não contém os dígitos 300000 → guarda rejeita a redação.
    rewritten = "Seu limite é de três mil reais."
    spec = _spec()
    assert _composer(rewritten).compose(spec) == spec.fallback


def test_fallback_when_number_altered() -> None:
    rewritten = "Seu limite é R$ 5.000,00."
    spec = _spec()
    assert _composer(rewritten).compose(spec) == spec.fallback


def test_enforces_non_numeric_must_include() -> None:
    spec = _spec(
        fallback="Sua solicitação foi rejeitada.",
        facts={},
        must_include=("rejeitad",),
    )
    # Sem o termo obrigatório → cai no fallback.
    assert _composer("Sua solicitação não foi aceita.").compose(spec) == spec.fallback
    # Com o termo obrigatório → aceita a redação.
    ok = "Infelizmente sua solicitação foi rejeitada desta vez."
    assert _composer(ok).compose(spec) == ok


def test_empty_rewrite_falls_back() -> None:
    spec = _spec()
    assert _composer("   ").compose(spec) == spec.fallback


def test_error_after_retries_falls_back() -> None:
    fake = _FakeChat(raises=True)
    spec = _spec()
    assert MessageComposer(cast(BaseChatModel, fake)).compose(spec) == spec.fallback
    assert fake.calls > 1  # houve retry antes de desistir


def test_handles_content_block_list() -> None:
    blocks = [{"type": "text", "text": "Você tem R$ 3.000,00 de limite disponível."}]
    result = _composer(blocks).compose(_spec())
    assert result == "Você tem R$ 3.000,00 de limite disponível."


def test_make_composer_none_when_no_model() -> None:
    assert make_message_composer(None) is None


def test_make_composer_wraps_model() -> None:
    composer = make_message_composer(cast(BaseChatModel, _FakeChat("R$ 3.000,00 disponíveis.")))
    assert composer is not None
    assert composer.compose(_spec()) == "R$ 3.000,00 disponíveis."
