"""Testes dos utilitários de CPF, moeda e datas."""

from datetime import date

import pytest

from banco_agil.utils.conversation import extract_money
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


def test_parse_flexible_date_formats() -> None:
    assert parse_flexible_date("1990-05-15") == date(1990, 5, 15)
    assert parse_flexible_date("15/05/1990") == date(1990, 5, 15)
    assert parse_flexible_date("15-05-1990") == date(1990, 5, 15)
    assert parse_flexible_date("15051990") == date(1990, 5, 15)
    assert parse_flexible_date(date(1990, 5, 15)) == date(1990, 5, 15)


def test_parse_flexible_date_invalid() -> None:
    with pytest.raises(ValueError):
        parse_flexible_date("15.05.1990")
