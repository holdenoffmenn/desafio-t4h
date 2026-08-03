"""Testes dos utilitários de CPF, moeda e datas."""

from datetime import date

import pytest

from banco_agil.utils.conversation import extract_currency_code, extract_money
from banco_agil.utils.cpf import is_valid_cpf, normalize_cpf
from banco_agil.utils.currency import normalize_brazilian_currency
from banco_agil.utils.dates import parse_flexible_date


def test_normalize_cpf_removes_mask() -> None:
    assert normalize_cpf("529.982.247-25") == "52998224725"


def test_normalize_cpf_empty_raises() -> None:
    with pytest.raises(ValueError):
        normalize_cpf("abc")


def test_is_valid_cpf_known_valid() -> None:
    assert is_valid_cpf("529.982.247-25") is True


def test_is_valid_cpf_rejects_repeated() -> None:
    assert is_valid_cpf("111.111.111-11") is False


def test_normalize_brazilian_currency_formats() -> None:
    assert normalize_brazilian_currency("5.000,00") == 5000.0
    assert normalize_brazilian_currency("5000,50") == 5000.5
    assert normalize_brazilian_currency(1000) == 1000.0
    assert normalize_brazilian_currency("R$ 1.200,00") == 1200.0


def test_normalize_brazilian_currency_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_brazilian_currency("abc")


def test_extract_money_plain_and_formatted() -> None:
    assert extract_money("quero 30000") == 30000.0
    assert extract_money("R$ 3.000,00") == 3000.0
    assert extract_money("sem valor aqui") is None


def test_extract_money_partial_scale_words() -> None:
    # Regressão: "25 mil" era lido como 25 (bug do R$ 25,00).
    assert extract_money("queria passar para 25 mil reais") == 25000.0
    assert extract_money("2,5 mil") == 2500.0
    assert extract_money("aumentar para 3 milhões") == 3_000_000.0
    # Por extenso completo continua a cargo do interpretador LLM.
    assert extract_money("sete mil") is None


# ---------------------------------------------------------------------------
# Heurística determinística (sem LLM). Quando a LLM está off, só este caminho
# existe. Cobertura do extrator LLM: ``tests/unit/test_llm_currency.py``.
# ---------------------------------------------------------------------------


def test_extract_currency_code_by_iso() -> None:
    """Heurística: códigos ISO frequentes (sem LLM)."""
    assert extract_currency_code("quero USD") == "USD"
    assert extract_currency_code("quero usd") == "USD"
    assert extract_currency_code("cotação EUR/BRL") == "EUR"
    assert extract_currency_code("quanto está o ARS hoje") == "ARS"
    assert extract_currency_code("cotação de KRW") == "KRW"
    assert extract_currency_code("quero PLN") == "PLN"


def test_extract_currency_code_by_name() -> None:
    """Heurística: nomes em português (sem LLM)."""
    # Regressão: "peso argentino" era silenciosamente lido como USD.
    assert extract_currency_code("preciso saber o valor do peso argentino") == "ARS"
    assert extract_currency_code("cotação do dólar") == "USD"
    assert extract_currency_code("quanto está o euro") == "EUR"
    assert extract_currency_code("libra esterlina") == "GBP"
    assert extract_currency_code("iene japonês") == "JPY"
    assert extract_currency_code("franco suíço") == "CHF"
    assert extract_currency_code("iuan chinês") == "CNY"
    # Variante específica não pode ser ofuscada pelo genérico "dólar".
    assert extract_currency_code("dólar canadense") == "CAD"
    assert extract_currency_code("dólar australiano") == "AUD"
    assert extract_currency_code("won sul-coreano") == "KRW"
    assert extract_currency_code("baht tailandês") == "THB"


def test_extract_currency_code_by_country_or_named_currency() -> None:
    """Heurística: país / nome → ISO (sem LLM)."""
    # Regressão: "moeda da Rússia" precisa resolver para RUB.
    assert extract_currency_code("tenho uma viagem para a Rússia") == "RUB"
    assert extract_currency_code("quero rublo") == "RUB"
    assert extract_currency_code("peso mexicano") == "MXN"
    assert extract_currency_code("lira turca") == "TRY"
    assert extract_currency_code("coroa norueguesa") == "NOK"
    # "peso" genérico continua caindo em ARS (mais comum para o público BR).
    assert extract_currency_code("quanto está o peso") == "ARS"


def test_extract_currency_code_unknown_returns_default() -> None:
    """Heurística: sem moeda clara devolve default (LLM cobriria em produção)."""
    assert extract_currency_code("quero saber o câmbio") is None
    assert extract_currency_code("bom dia") is None
    assert extract_currency_code("qual a taxa?", default="USD") == "USD"


def test_extract_currency_code_avoids_substring_false_positive() -> None:
    """Heurística: 'mercado' não deve casar CAD por substring."""
    assert extract_currency_code("como está o mercado?") is None


def test_parse_flexible_date_formats() -> None:
    assert parse_flexible_date("1990-05-15") == date(1990, 5, 15)
    assert parse_flexible_date("15/05/1990") == date(1990, 5, 15)
    assert parse_flexible_date("15-05-1990") == date(1990, 5, 15)
    assert parse_flexible_date("15051990") == date(1990, 5, 15)
    assert parse_flexible_date(date(1990, 5, 15)) == date(1990, 5, 15)


def test_parse_flexible_date_invalid() -> None:
    with pytest.raises(ValueError):
        parse_flexible_date("15.05.1990")
