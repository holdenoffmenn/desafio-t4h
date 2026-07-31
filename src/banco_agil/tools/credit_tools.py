"""Tools de crédito: consulta e aumento de limite."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from banco_agil.domain.credit_limit import CreditLimitService
from banco_agil.domain.models import Customer
from banco_agil.domain.protocols import CustomerRepository, ScoreLimitRepository
from banco_agil.infrastructure.credit_request_repository import CsvCreditRequestRepository


def get_credit_limit_update(customer: Customer) -> dict[str, Any]:
    """Monta update/observabilidade para consulta de limite.

    Args:
        customer: Cliente autenticado.

    Returns:
        Update com last_tool_calls.
    """
    return {
        "last_tool_calls": [
            {
                "name": "get_credit_limit",
                "args": {"cpf": "***"},
                "result": customer.limite_atual,
            }
        ],
    }


def request_limit_increase_update(
    *,
    customer: Customer,
    new_limit: float,
    credit_service: CreditLimitService,
    score_repo: ScoreLimitRepository,
    request_repo: CsvCreditRequestRepository,
    customer_repo: CustomerRepository,
) -> dict[str, Any]:
    """Cria pedido pendente, avalia e atualiza status (ciclo do PDF).

    Args:
        customer: Cliente autenticado.
        new_limit: Novo limite solicitado.
        credit_service: Serviço de decisão.
        score_repo: Tabela score → limite.
        request_repo: Persistência de solicitações.
        customer_repo: Para atualizar limite se aprovado.

    Returns:
        Update de SessionState com status e ids.
    """
    request_id = request_repo.create(
        cpf_cliente=customer.cpf,
        limite_atual=customer.limite_atual,
        novo_limite_solicitado=new_limit,
    )
    decision = credit_service.evaluate_request(customer, new_limit, score_repo)
    request_repo.update_status(request_id, decision.status)

    customer_snapshot = customer.model_dump(mode="json")
    if decision.status == "aprovado" and decision.approved_limit is not None:
        customer_repo.update_limit(customer.cpf, decision.approved_limit)
        customer_snapshot["limite_atual"] = decision.approved_limit

    return {
        "pending_new_limit": new_limit,
        "last_request_id": request_id,
        "last_request_status": decision.status,
        "customer": customer_snapshot,
        "offered_interview": False,
        "interview_accepted": False,
        "last_tool_calls": [
            {
                "name": "request_limit_increase",
                "args": {"cpf": "***", "new_limit": new_limit},
                "result": {
                    "request_id": request_id,
                    "status": decision.status,
                    "reason": decision.reason,
                },
            }
        ],
    }


def offer_credit_interview_update() -> dict[str, Any]:
    """Marca que a entrevista foi oferecida.

    Returns:
        Update com ``offered_interview=True``.
    """
    return {"offered_interview": True}


@tool
def get_credit_limit() -> str:
    """Consulta o limite de crédito do cliente autenticado."""
    return "get_credit_limit_requested"


@tool
def request_limit_increase(new_limit: float) -> str:
    """Solicita aumento de limite e persiste o pedido formal.

    Args:
        new_limit: Novo limite desejado em reais.
    """
    return f"request_limit_increase:{new_limit}"


@tool
def offer_credit_interview() -> str:
    """Oferece entrevista de crédito após rejeição."""
    return "interview_offered"
