"""Testes do CreditLimitService."""

from datetime import date

from banco_agil.domain.credit_limit import CreditLimitService
from banco_agil.domain.models import Customer


class FakeScoreRepo:
    """Tabela de score em memória."""

    def __init__(self, bands: dict[tuple[int, int], float]) -> None:
        self._bands = bands

    def get_max_limit_for_score(self, score: int) -> float:
        for (lo, hi), limit in self._bands.items():
            if lo <= score <= hi:
                return limit
        raise LookupError(score)


def _customer(score: int = 450, limite: float = 3000.0) -> Customer:
    return Customer(
        cpf="52998224725",
        data_nascimento=date(1990, 5, 15),
        nome="Ana",
        limite_atual=limite,
        score=score,
    )


def test_approve_within_band() -> None:
    repo = FakeScoreRepo({(300, 599): 5000.0})
    decision = CreditLimitService().evaluate_request(_customer(), 4000.0, repo)
    assert decision.status == "aprovado"
    assert decision.approved_limit == 4000.0


def test_reject_score_insufficient() -> None:
    repo = FakeScoreRepo({(300, 599): 5000.0})
    decision = CreditLimitService().evaluate_request(_customer(), 6000.0, repo)
    assert decision.status == "rejeitado"
    assert decision.reason == "score_insuficiente"


def test_reject_limite_menor_que_atual() -> None:
    repo = FakeScoreRepo({(300, 599): 5000.0})
    decision = CreditLimitService().evaluate_request(_customer(), 2000.0, repo)
    assert decision.status == "rejeitado"
    assert decision.reason == "limite_menor_que_atual"


def test_reject_equal_to_current() -> None:
    repo = FakeScoreRepo({(300, 599): 5000.0})
    decision = CreditLimitService().evaluate_request(_customer(), 3000.0, repo)
    assert decision.status == "rejeitado"
    assert decision.reason == "limite_menor_que_atual"
