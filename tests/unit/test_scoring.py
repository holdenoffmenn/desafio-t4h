"""Testes do ScoringService."""

from banco_agil.domain.models import InterviewInput
from banco_agil.domain.scoring import ScoringService


def test_calculate_score_formal_no_debts() -> None:
    data = InterviewInput(
        renda_mensal=5000,
        tipo_emprego="formal",
        despesas_fixas=1000,
        num_dependentes=0,
        tem_dividas="não",
    )
    score = ScoringService().calculate_score(data)
    # (5000/1001)*30 + 300 + 100 + 100 ≈ 649
    assert 0 <= score <= 1000
    assert score == 649


def test_calculate_score_dependentes_3_mais() -> None:
    data = InterviewInput(
        renda_mensal=1000,
        tipo_emprego="desempregado",
        despesas_fixas=0,
        num_dependentes=5,
        tem_dividas="sim",
    )
    score = ScoringService().calculate_score(data)
    # (1000/1)*30 + 0 + 30 + (-100) = 29930 → clamp 1000
    assert score == 1000


def test_calculate_score_clamps_to_zero() -> None:
    data = InterviewInput(
        renda_mensal=0.01,
        tipo_emprego="desempregado",
        despesas_fixas=1_000_000,
        num_dependentes=3,
        tem_dividas="sim",
    )
    # quase zero + 0 + 30 - 100 = negativo → 0
    assert ScoringService().calculate_score(data) == 0


def test_all_employment_types() -> None:
    for tipo in ("formal", "autônomo", "desempregado"):
        data = InterviewInput(
            renda_mensal=2000,
            tipo_emprego=tipo,  # type: ignore[arg-type]
            despesas_fixas=500,
            num_dependentes=1,
            tem_dividas="não",
        )
        score = ScoringService().calculate_score(data)
        assert 0 <= score <= 1000
