"""Suite LLM-only do extrator de moeda (``LlmExtractor.currency``).

Os casos abaixo simulam a resposta do chat model — não exercitam a heurística
``extract_currency_code``. Frases do usuário são ilustrativas; o que importa é
o parsing da resposta da LLM (código ISO, null, lixo, blocos, falha de rede).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.llm.extract import LlmExtractor
from banco_agil.utils.conversation import extract_currency_code


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
# Happy path: resposta da LLM → código ISO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_text", "llm_reply", "expected"),
    [
        # Moedas comuns (gírias / nomes que a heurística também pegaria — aqui
        # validamos só o parser da LLM).
        ("cotação do dólar", "USD", "USD"),
        ("quanto está o euro", "EUR", "EUR"),
        ("libra esterlina", "GBP", "GBP"),
        ("iene japonês", "JPY", "JPY"),
        ("franco suíço", "CHF", "CHF"),
        ("iuan chinês", "CNY", "CNY"),
        ("peso argentino", "ARS", "ARS"),
        ("dólar canadense", "CAD", "CAD"),
        ("dólar australiano", "AUD", "AUD"),
        ("bitcoin", "BTC", "BTC"),
        # Resposta em minúsculas / com pontuação / ruído ao redor.
        ("dólar", "usd", "USD"),
        ("euro", "eur.", "EUR"),
        ("iene", "JPY!", "JPY"),
        ("peso mexicano", " o código é mxn ", "MXN"),
        ("rublo", "RUB\n", "RUB"),
        # Gírias e circunlóquios (heurística costuma falhar — ver asserts abaixo).
        ("quero a cotação do greenback", "USD", "USD"),
        ("quanto tá o cable hoje", "GBP", "GBP"),
        ("me passa o loonie", "CAD", "CAD"),
        ("bucks americanos", "USD", "USD"),
        ("sterling", "GBP", "GBP"),
        ("moeda usada em Tóquio", "JPY", "JPY"),
        ("dinheiro de Bangkok", "THB", "THB"),
        ("moeda que circula em Seul", "KRW", "KRW"),
        ("a moeda da Islândia", "ISK", "ISK"),
        ("forint húngaro", "HUF", "HUF"),
        ("leu romeno", "RON", "RON"),
        ("coroa tcheca", "CZK", "CZK"),
        ("ringgit malaio", "MYR", "MYR"),
        ("dong vietnamita", "VND", "VND"),
        ("riyal saudita", "SAR", "SAR"),
        ("dinar do Kuwait", "KWD", "KWD"),
        ("naira da Nigéria", "NGN", "NGN"),
        ("a paridade do kiwi", "NZD", "NZD"),
        ("eurinho pra viagem", "EUR", "EUR"),
        ("câmbio de Praga", "CZK", "CZK"),
        ("moeda da Hungria", "HUF", "HUF"),
        ("viajar a Budapeste", "HUF", "HUF"),
        ("cotação pra Dubai", "AED", "AED"),
        ("moeda do Marrocos", "MAD", "MAD"),
        ("quetzal da Guatemala", "GTQ", "GTQ"),
        ("córdoba nicaraguense", "NIO", "NIO"),
        ("balboa do Panamá", "PAB", "PAB"),
        ("lempira", "HNL", "HNL"),
        ("gourde haitiano", "HTG", "HTG"),
        ("pula de Botswana", "BWP", "BWP"),
        ("tugrik mongol", "MNT", "MNT"),
        ("tenge cazaque", "KZT", "KZT"),
        ("dram armênio", "AMD", "AMD"),
        ("lari georgiano", "GEL", "GEL"),
        ("kuna croata", "HRK", "HRK"),
        ("pataca de Macau", "MOP", "MOP"),
        ("taka bengali", "BDT", "BDT"),
        ("kip laosiano", "LAK", "LAK"),
        ("riel cambojano", "KHR", "KHR"),
        ("kyat birmanês", "MMK", "MMK"),
        ("dinar sérvio", "RSD", "RSD"),
        ("moeda da Sérvia", "RSD", "RSD"),
        ("won sul-coreano", "KRW", "KRW"),
        ("zloty polonês", "PLN", "PLN"),
        ("shekel israelense", "ILS", "ILS"),
        ("rand sul-africano", "ZAR", "ZAR"),
        ("rúpia indiana", "INR", "INR"),
        ("peso chileno", "CLP", "CLP"),
        ("sol peruano", "PEN", "PEN"),
        ("guarani paraguaio", "PYG", "PYG"),
    ],
)
def test_llm_currency_parses_iso(
    user_text: str,
    llm_reply: str,
    expected: str,
) -> None:
    assert _ext(llm_reply).currency(user_text) == expected


def test_llm_only_phrases_bypass_heuristic() -> None:
    """Garante que o subconjunto 'gíria/circunlóquio' realmente exige LLM."""
    llm_only = (
        "quero a cotação do greenback",
        "quanto tá o cable hoje",
        "me passa o loonie",
        "moeda usada em Tóquio",
        "dinheiro de Bangkok",
        "a moeda da Islândia",
        "forint húngaro",
        "leu romeno",
        "coroa tcheca",
        "ringgit malaio",
        "dong vietnamita",
        "riyal saudita",
        "dinar do Kuwait",
        "naira da Nigéria",
        "a paridade do kiwi",
        "eurinho pra viagem",
        "câmbio de Praga",
        "quetzal da Guatemala",
        "pula de Botswana",
        "kuna croata",
    )
    for phrase in llm_only:
        assert extract_currency_code(phrase) is None, phrase


# ---------------------------------------------------------------------------
# Degradação / respostas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("llm_reply", "user_text"),
    [
        ("null", "não sei qual moeda"),
        ("none", "qualquer uma"),
        ("n/a", "sei lá"),
        ("", "câmbio"),
        ("moeda estrangeira", "quero cotação"),
        ("não identifiquei", "x"),
        ("código inválido", "y"),
        ("US", "dólar"),  # 2 letras — rejeitado
        ("USDT", "crypto"),  # token de 4 letras, sem ISO isolado de 3
        ("12", "número"),
        ("$$$", "símbolos"),
    ],
)
def test_llm_currency_invalid_reply_returns_none(llm_reply: str, user_text: str) -> None:
    assert _ext(llm_reply).currency(user_text) is None


def test_llm_currency_usdt_does_not_yield_usd() -> None:
    """'USDT' é um único token de 4 letras — não deve extrair USD."""
    assert _ext("USDT").currency("tether") is None


def test_llm_currency_picks_first_iso_token() -> None:
    assert _ext("USD ou EUR").currency("dúvida") == "USD"


def test_llm_currency_content_blocks() -> None:
    blocks = [{"type": "text", "text": "KRW"}]
    assert _ext(blocks).currency("moeda de Seul") == "KRW"


def test_llm_currency_network_error_returns_none() -> None:
    assert _ext(raises=True).currency("greenback") is None


def test_llm_currency_empty_user_text_skips_model() -> None:
    fake = _FakeChat("USD")
    ext = LlmExtractor(cast(BaseChatModel, fake))
    assert ext.currency("   ") is None
    assert fake.calls == 0


def test_llm_currency_sends_user_text_to_model() -> None:
    fake = _FakeChat("THB")
    ext = LlmExtractor(cast(BaseChatModel, fake))
    assert ext.currency("dinheiro de Bangkok") == "THB"
    assert fake.calls == 1
    human = fake.last_messages[1]
    assert human.content == "dinheiro de Bangkok"
