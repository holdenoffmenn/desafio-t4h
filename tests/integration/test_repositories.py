"""Testes de integração dos repositórios CSV (atomicidade e ciclo de status)."""

from datetime import date
from pathlib import Path

import httpx
import pytest

from banco_agil.domain.errors import (
    CustomerNotFoundError,
    FxPairNotFoundError,
    ScoreTableError,
)
from banco_agil.infrastructure.credit_request_repository import CsvCreditRequestRepository
from banco_agil.infrastructure.customer_repository import CsvCustomerRepository
from banco_agil.infrastructure.fx_client import FxClient
from banco_agil.infrastructure.score_limit_repository import CsvScoreLimitRepository


def _seed_clientes(path: Path) -> None:
    path.write_text(
        "cpf,data_nascimento,nome,limite_atual,score\n"
        "52998224725,1990-05-15,Ana Souza,3000.00,450\n"
        "39053344705,1985-11-20,Bruno Lima,8000.00,720\n",
        encoding="utf-8",
    )


def _seed_score(path: Path) -> None:
    path.write_text(
        "score_min,score_max,limite_max_permitido\n"
        "0,299,1000.00\n"
        "300,599,5000.00\n"
        "600,799,15000.00\n"
        "800,1000,50000.00\n",
        encoding="utf-8",
    )


def test_customer_find_and_authenticate(tmp_path: Path) -> None:
    path = tmp_path / "clientes.csv"
    _seed_clientes(path)
    repo = CsvCustomerRepository(path)

    customer = repo.find_by_cpf("529.982.247-25")
    assert customer is not None
    assert customer.nome == "Ana Souza"

    ok = repo.authenticate("52998224725", date(1990, 5, 15))
    assert ok is not None

    fail = repo.authenticate("52998224725", date(2000, 1, 1))
    assert fail is None


def test_update_score_atomic(tmp_path: Path) -> None:
    path = tmp_path / "clientes.csv"
    _seed_clientes(path)
    repo = CsvCustomerRepository(path)

    repo.update_score("52998224725", 777)
    updated = repo.find_by_cpf("52998224725")
    assert updated is not None
    assert updated.score == 777
    # Arquivo ainda legível (não corrompido)
    assert "777" in path.read_text(encoding="utf-8")


def test_update_score_unknown_customer(tmp_path: Path) -> None:
    path = tmp_path / "clientes.csv"
    _seed_clientes(path)
    repo = CsvCustomerRepository(path)
    with pytest.raises(CustomerNotFoundError):
        repo.update_score("00000000000", 100)


def test_credit_request_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "solicitacoes.csv"
    repo = CsvCreditRequestRepository(path)

    request_id = repo.create(
        cpf_cliente="52998224725",
        limite_atual=3000.0,
        novo_limite_solicitado=6000.0,
    )
    pending = repo.get_by_id(request_id)
    assert pending is not None
    assert pending.status_pedido == "pendente"

    repo.update_status(request_id, "rejeitado")
    updated = repo.get_by_id(request_id)
    assert updated is not None
    assert updated.status_pedido == "rejeitado"


def test_score_limit_lookup(tmp_path: Path) -> None:
    path = tmp_path / "score_limite.csv"
    _seed_score(path)
    repo = CsvScoreLimitRepository(path)

    assert repo.get_max_limit_for_score(450) == 5000.0
    assert repo.get_max_limit_for_score(850) == 50000.0
    with pytest.raises(ScoreTableError):
        # score fora de qualquer faixa (se tabela incompleta)
        # nossa tabela cobre 0-1000; forçamos erro com score negativo via cast
        CsvScoreLimitRepository(path).get_max_limit_for_score(-1)


def test_fx_client_mock() -> None:
    client = FxClient(
        api_url_template="https://example.com/{pair}",
        mock=True,
    )
    rate = client.get_rate("usd")
    assert rate.currency == "USD"
    assert rate.bid > 0
    assert rate.timestamp == "mock"


def test_fx_client_pair_not_found() -> None:
    """404 / CoinNotExists da AwesomeAPI vira FxPairNotFoundError (sem retry)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "status": 404,
                "code": "CoinNotExists",
                "message": "moeda nao encontrada XYZ-BRL",
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = FxClient(
            api_url_template="https://example.com/json/last/{pair}",
            client=http_client,
        )
        with pytest.raises(FxPairNotFoundError):
            client.get_rate("XYZ")


def test_fx_client_rejects_invalid_currency_code() -> None:
    client = FxClient(api_url_template="https://example.com/{pair}", mock=True)
    with pytest.raises(ValueError):
        client.get_rate("XY")
