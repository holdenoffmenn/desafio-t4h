"""Suite LLM-only dos extratores usados no fluxo de limite de crédito.

Cobre ``money`` (limite/renda/despesas), ``tipo_emprego``, ``num_dependentes``
e ``tem_dividas``. Não exercita a heurística de ``utils.conversation`` /
``interview._heuristic_field`` — o fake chat devolve a resposta da LLM.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.graph.nodes.interview import _heuristic_field, _parse_emprego
from banco_agil.llm.extract import LlmExtractor, MoneyField
from banco_agil.utils.conversation import extract_money

Emprego = Literal["formal", "autônomo", "desempregado"]


class _FakeChat:
    """Chat model mínimo: reply fixo, fila, ou raise."""

    def __init__(
        self,
        reply: str | list[Any] | None = None,
        *,
        raises: bool = False,
    ) -> None:
        self._reply = reply
        self._raises = raises
        self.calls = 0
        self.last_messages: Any = None

    def invoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
        self.calls += 1
        self.last_messages = messages
        if self._raises:
            raise RuntimeError("network down")
        return AIMessage(content=self._reply if self._reply is not None else "")


def _ext(reply: str | list[Any] | None = None, *, raises: bool = False) -> LlmExtractor:
    return LlmExtractor(cast(BaseChatModel, _FakeChat(reply, raises=raises)))


# ---------------------------------------------------------------------------
# money — parser + frases que a heurística não cobre
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_text", "llm_reply", "expected", "field"),
    [
        ("sete mil", "7000", 7000.0, "limite_credito"),
        ("quinze mil reais", "15000", 15000.0, "renda_mensal"),
        ("vinte e cinco mil", "25000", 25000.0, "limite_credito"),
        ("uns dois mil e quinhentos", "2500", 2500.0, "despesas_fixas"),
        ("cinco milzinho", "5000", 5000.0, "limite_credito"),
        ("dez mil", "10000", 10000.0, "renda_mensal"),
        ("oito mil e quinhentos", "8500", 8500.0, "despesas_fixas"),
        ("mil e quinhentos", "1500", 1500.0, "despesas_fixas"),
        ("quatro mil", "4000", 4000.0, "limite_credito"),
        ("seis mil", "6000", 6000.0, "limite_credito"),
        ("três mil e duzentos", "3200", 3200.0, "renda_mensal"),
        ("doze mil", "12000", 12000.0, "limite_credito"),
        ("trinta mil", "30000", 30000.0, "limite_credito"),
        ("cinquenta mil", "50000", 50000.0, "renda_mensal"),
        ("um milhão", "1000000", 1_000_000.0, "renda_mensal"),
        ("quero uns sete mil", "7000", 7000.0, "limite_credito"),
        ("mais ou menos quinze mil", "15000", 15000.0, "renda_mensal"),
        # Formatos de resposta do modelo.
        ("sete mil", " 7000 ", 7000.0, "limite_credito"),
        ("sete mil", "r$7000", 7000.0, "limite_credito"),
        ("sete mil", "o valor é 7000.", 7000.0, "limite_credito"),
        ("despesas", "3500.50", 3500.50, "despesas_fixas"),
        ("despesas", "3500.5", 3500.5, "despesas_fixas"),
        ("zero", "0", 0.0, "despesas_fixas"),
    ],
)
def test_llm_money_parses(
    user_text: str,
    llm_reply: str,
    expected: float,
    field: MoneyField,
) -> None:
    assert _ext(llm_reply).money(user_text, field=field) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "sete mil",
        "quinze mil reais",
        "vinte e cinco mil",
        "uns dois mil e quinhentos",
        "cinco milzinho",
        "dez mil",
        "oito mil e quinhentos",
        "mil e quinhentos",
        "quatro mil",
        "seis mil",
        "três mil e duzentos",
        "doze mil",
        "trinta mil",
        "um milhão",
        "quero uns sete mil",
    ],
)
def test_llm_money_phrases_bypass_heuristic(phrase: str) -> None:
    assert extract_money(phrase) is None, phrase


@pytest.mark.parametrize(
    ("llm_reply", "user_text"),
    [
        ("null", "não sei"),
        ("none", "sei lá"),
        ("", "x"),
        ("sem número aqui", "y"),
        ("-100", "negativo"),
        ("abc", "lixo"),
    ],
)
def test_llm_money_invalid_returns_none(llm_reply: str, user_text: str) -> None:
    assert _ext(llm_reply).money(user_text) is None


def test_llm_money_content_blocks() -> None:
    blocks = [{"type": "text", "text": "7000"}]
    assert _ext(blocks).money("sete mil") == 7000.0


def test_llm_money_network_error_returns_none() -> None:
    assert _ext(raises=True).money("sete mil") is None


def test_llm_money_empty_user_skips_model() -> None:
    fake = _FakeChat("7000")
    ext = LlmExtractor(cast(BaseChatModel, fake))
    assert ext.money("   ") is None
    assert fake.calls == 0


@pytest.mark.parametrize(
    ("field", "needle"),
    [
        ("limite_credito", "novo limite de crédito desejado"),
        ("renda_mensal", "renda mensal bruta"),
        ("despesas_fixas", "despesas fixas mensais"),
    ],
)
def test_llm_money_prompt_includes_field_label(field: MoneyField, needle: str) -> None:
    fake = _FakeChat("4000")
    ext = LlmExtractor(cast(BaseChatModel, fake))
    assert ext.money("quatro mil", field=field) == 4000.0
    system = fake.last_messages[0].content.lower()
    assert needle in system


# ---------------------------------------------------------------------------
# tipo_emprego
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_text", "llm_reply", "expected"),
    [
        ("sou MEI", "autônomo", "autônomo"),
        ("trabalho PJ", "autônomo", "autônomo"),
        ("empreendedor", "autônomo", "autônomo"),
        ("conta própria", "autônomo", "autônomo"),
        ("freelancer", "autônomo", "autônomo"),
        ("estou sem emprego", "desempregado", "desempregado"),
        ("perdi o emprego", "desempregado", "desempregado"),
        ("servidor público", "formal", "formal"),
        ("carteira assinada", "formal", "formal"),
        ("sou CLT", "formal", "formal"),
        ("trabalho registrado", "formal", "formal"),
        ("autonomo sem acento", "autonomo", "autônomo"),
        ("FORMAL", "formal", "formal"),
    ],
)
def test_llm_tipo_emprego(
    user_text: str,
    llm_reply: str,
    expected: Emprego,
) -> None:
    assert _ext(llm_reply).tipo_emprego(user_text) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "sou MEI",
        "trabalho PJ",
        "empreendedor",
        "estou sem emprego",
        "servidor público",
        "conta própria",
        "perdi o emprego",
        "trabalho registrado",
    ],
)
def test_llm_emprego_phrases_bypass_heuristic(phrase: str) -> None:
    assert _parse_emprego(phrase.lower()) is None, phrase


@pytest.mark.parametrize("llm_reply", ["null", "none", "", "estagiário", "aposentado"])
def test_llm_tipo_emprego_invalid_returns_none(llm_reply: str) -> None:
    assert _ext(llm_reply).tipo_emprego("x") is None


# ---------------------------------------------------------------------------
# num_dependentes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_text", "llm_reply", "expected"),
    [
        ("moro sozinho", "0", 0),
        ("dois filhos", "2", 2),
        ("três", "3", 3),
        ("tenho dois", "2", 2),
        ("quatro dependentes", "4", 4),
        ("só eu", "0", 0),
        ("nenhum filho", "0", 0),
        ("um filho", "1", 1),
        ("cinco", "5", 5),
        ("  2  ", "2", 2),
    ],
)
def test_llm_num_dependentes(user_text: str, llm_reply: str, expected: int) -> None:
    assert _ext(llm_reply).num_dependentes(user_text) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "moro sozinho",
        "dois filhos",
        "três",
        "tenho dois",
        "só eu",
        "um filho",
        "quatro dependentes",
    ],
)
def test_llm_dependentes_phrases_bypass_heuristic(phrase: str) -> None:
    assert _heuristic_field("num_dependentes", phrase) is None, phrase


@pytest.mark.parametrize("llm_reply", ["null", "none", "", "vários", "muitos"])
def test_llm_num_dependentes_invalid_returns_none(llm_reply: str) -> None:
    assert _ext(llm_reply).num_dependentes("x") is None


def test_llm_num_dependentes_extracts_digits_from_noisy_reply() -> None:
    """O parser pega o primeiro bloco de dígitos (ex.: '-1' → 1)."""
    assert _ext("-1").num_dependentes("x") == 1


# ---------------------------------------------------------------------------
# tem_dividas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_text", "llm_reply", "expected"),
    [
        ("estou negativado", "sim", "sim"),
        ("tenho nome sujo", "sim", "sim"),
        ("devendo no cartão", "sim", "sim"),
        ("ainda pago um empréstimo", "sim", "sim"),
        ("já quitei tudo", "não", "não"),
        ("sem dívidas ativas", "não", "não"),
        ("estou limpo no SPC", "não", "não"),
        ("nada em aberto", "não", "não"),
        ("SIM", "sim", "sim"),
        ("NÃO", "não", "não"),
    ],
)
def test_llm_tem_dividas(
    user_text: str,
    llm_reply: str,
    expected: Literal["sim", "não"],
) -> None:
    assert _ext(llm_reply).tem_dividas(user_text) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "estou negativado",
        "ainda pago um empréstimo",
        "estou limpo no SPC",
        "meu score no serasa está ok",
        "financiei um carro e ainda resta parcela",
    ],
)
def test_llm_dividas_phrases_bypass_heuristic(phrase: str) -> None:
    assert _heuristic_field("tem_dividas", phrase) is None, phrase


@pytest.mark.parametrize("llm_reply", ["null", "none", "", "talvez", "às vezes"])
def test_llm_tem_dividas_invalid_returns_none(llm_reply: str) -> None:
    assert _ext(llm_reply).tem_dividas("x") is None


def test_llm_credit_fields_network_error_degrades() -> None:
    ext = _ext(raises=True)
    assert ext.money("sete mil") is None
    assert ext.tipo_emprego("sou MEI") is None
    assert ext.num_dependentes("dois filhos") is None
    assert ext.tem_dividas("estou negativado") is None
