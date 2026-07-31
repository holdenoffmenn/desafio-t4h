"""Testes do AuthService."""

from datetime import date

from banco_agil.domain.auth import AuthService
from banco_agil.domain.models import Customer


class FakeCustomerRepo:
    """Repositório em memória para testes de autenticação."""

    def __init__(self, customers: list[Customer]) -> None:
        self._by_cpf = {c.cpf: c for c in customers}

    def find_by_cpf(self, cpf: str) -> Customer | None:
        return self._by_cpf.get(cpf)

    def authenticate(self, cpf: str, birth_date: date) -> Customer | None:
        customer = self.find_by_cpf(cpf)
        if customer is None or customer.data_nascimento != birth_date:
            return None
        return customer

    def update_score(self, cpf: str, new_score: int) -> None:
        raise NotImplementedError

    def update_limit(self, cpf: str, new_limit: float) -> None:
        raise NotImplementedError


def _sample(
    *,
    cpf: str = "52998224725",
    data_nascimento: date = date(1990, 5, 15),
    nome: str = "Ana Souza",
    limite_atual: float = 3000.0,
    score: int = 450,
) -> Customer:
    return Customer(
        cpf=cpf,
        data_nascimento=data_nascimento,
        nome=nome,
        limite_atual=limite_atual,
        score=score,
    )


def test_authenticate_success() -> None:
    repo = FakeCustomerRepo([_sample()])
    result = AuthService().authenticate("529.982.247-25", date(1990, 5, 15), repo)
    assert result is not None
    assert result.nome == "Ana Souza"


def test_authenticate_wrong_birth_date() -> None:
    repo = FakeCustomerRepo([_sample()])
    result = AuthService().authenticate("52998224725", date(1991, 1, 1), repo)
    assert result is None


def test_authenticate_unknown_cpf() -> None:
    repo = FakeCustomerRepo([_sample()])
    result = AuthService().authenticate("00000000000", date(1990, 5, 15), repo)
    assert result is None
