"""Testes de validação Pydantic do InterviewInput."""

import pytest
from pydantic import ValidationError

from banco_agil.domain.models import InterviewInput


def test_parse_brazilian_currency_fields() -> None:
    data = InterviewInput(
        renda_mensal="5.000,00",  # type: ignore[arg-type]
        tipo_emprego="formal",
        despesas_fixas="1.200,50",  # type: ignore[arg-type]
        num_dependentes=2,
        tem_dividas="não",
    )
    assert data.renda_mensal == 5000.0
    assert data.despesas_fixas == 1200.5


def test_rejects_zero_renda() -> None:
    with pytest.raises(ValidationError):
        InterviewInput(
            renda_mensal=0,
            tipo_emprego="formal",
            despesas_fixas=100,
            num_dependentes=0,
            tem_dividas="sim",
        )


def test_rejects_invalid_emprego() -> None:
    with pytest.raises(ValidationError):
        InterviewInput(
            renda_mensal=1000,
            tipo_emprego="freelancer",  # type: ignore[arg-type]
            despesas_fixas=100,
            num_dependentes=0,
            tem_dividas="não",
        )


def test_rejects_too_many_dependents() -> None:
    with pytest.raises(ValidationError):
        InterviewInput(
            renda_mensal=1000,
            tipo_emprego="formal",
            despesas_fixas=100,
            num_dependentes=21,
            tem_dividas="não",
        )
