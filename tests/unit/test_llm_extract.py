"""Testes do extrator de linguagem natural via LLM (com fake chat model)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.llm.extract import LlmExtractor, make_llm_extractor


class _FakeChat:
    """Chat model mínimo: devolve conteúdo fixo ou levanta exceção."""

    def __init__(self, reply: str | list[Any] | None = None, *, raises: bool = False) -> None:
        self._reply = reply
        self._raises = raises

    def invoke(self, _messages: Any, **_kwargs: Any) -> AIMessage:
        if self._raises:
            raise RuntimeError("network down")
        return AIMessage(content=self._reply if self._reply is not None else "")


def _ext(reply: str | list[Any] | None = None, *, raises: bool = False) -> LlmExtractor:
    return LlmExtractor(cast(BaseChatModel, _FakeChat(reply, raises=raises)))


def test_money_parses_number() -> None:
    assert _ext("7000").money("sete mil") == 7000.0
    assert _ext("3500.50").money("R$ 3.500,50", field="despesas_fixas") == 3500.50
    assert _ext(" 15000 ").money("quinze mil reais", field="renda_mensal") == 15000.0


def test_money_null_and_negative_return_none() -> None:
    assert _ext("null").money("não sei dizer") is None
    assert _ext("-100").money("x") is None
    assert _ext("sem número aqui").money("x") is None


def test_money_handles_content_block_list() -> None:
    blocks = [{"type": "text", "text": "7000"}]
    assert _ext(blocks).money("sete mil") == 7000.0


def test_tipo_emprego() -> None:
    assert _ext("formal").tipo_emprego("carteira assinada") == "formal"
    assert _ext("autônomo").tipo_emprego("sou MEI") == "autônomo"
    assert _ext("desempregado").tipo_emprego("estou sem emprego") == "desempregado"
    assert _ext("null").tipo_emprego("sei lá") is None


def test_num_dependentes() -> None:
    assert _ext("0").num_dependentes("moro sozinho") == 0
    assert _ext("2").num_dependentes("tenho dois filhos") == 2
    assert _ext("null").num_dependentes("não entendi") is None


def test_tem_dividas() -> None:
    assert _ext("não").tem_dividas("já quitei tudo") == "não"
    assert _ext("sim").tem_dividas("estou negativado") == "sim"
    assert _ext("null").tem_dividas("prefiro não dizer") is None


# Casos amplos de currency via LLM: ``tests/unit/test_llm_currency.py``.


def test_currency_returns_iso_code() -> None:
    """Smoke do extrator LLM de moeda (suite completa em test_llm_currency)."""
    assert _ext("ARS").currency("peso argentino") == "ARS"
    assert _ext("usd").currency("dólar") == "USD"
    assert _ext("JPY.").currency("iene japonês") == "JPY"


def test_currency_null_and_garbage_return_none() -> None:
    assert _ext("null").currency("não sei") is None
    assert _ext("moeda estrangeira").currency("x") is None


def test_birth_date_normalizes_to_iso() -> None:
    assert _ext("1991-09-19").birth_date("19-091991") == "1991-09-19"
    assert _ext("a data é 1990-05-15.").birth_date("15 de maio de 1990") == "1990-05-15"


def test_birth_date_null_and_garbage_return_none() -> None:
    assert _ext("null").birth_date("não lembro") is None
    assert _ext("sem data").birth_date("x") is None


def test_error_degrades_to_none() -> None:
    ext = _ext(raises=True)
    assert ext.money("sete mil") is None
    assert ext.tipo_emprego("clt") is None
    assert ext.num_dependentes("dois") is None
    assert ext.tem_dividas("não devo") is None
    assert ext.currency("peso argentino") is None
    assert ext.birth_date("19/09/1991") is None


def test_empty_text_returns_none() -> None:
    assert _ext("7000").money("   ") is None


def test_make_extractor_none_when_no_model() -> None:
    assert make_llm_extractor(None) is None


def test_make_extractor_wraps_model() -> None:
    ext = make_llm_extractor(cast(BaseChatModel, _FakeChat("7000")))
    assert ext is not None
    assert ext.money("sete mil") == 7000.0
