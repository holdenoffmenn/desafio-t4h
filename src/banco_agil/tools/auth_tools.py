"""Tools de autenticação."""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.tools import tool

from banco_agil.domain.auth import AuthService
from banco_agil.domain.protocols import CustomerRepository
from banco_agil.utils.cpf import normalize_cpf


def authenticate_customer_update(
    *,
    cpf: str,
    birth_date: date,
    auth_service: AuthService,
    repo: CustomerRepository,
    auth_attempts: int,
) -> dict[str, Any]:
    """Autentica e devolve update de estado.

    Args:
        cpf: CPF informado.
        birth_date: Data de nascimento.
        auth_service: Serviço de autenticação.
        repo: Repositório de clientes.
        auth_attempts: Tentativas já consumidas.

    Returns:
        Update parcial de SessionState.
    """
    customer = auth_service.authenticate(cpf, birth_date, repo)
    normalized = normalize_cpf(cpf)
    if customer is None:
        return {
            "cpf": normalized,
            "authenticated": False,
            "customer": None,
            "auth_attempts": auth_attempts + 1,
            "last_tool_calls": [
                {
                    "name": "authenticate_customer",
                    "args": {"cpf": "***", "birth_date": "***"},
                    "result": "auth_failed",
                }
            ],
        }
    return {
        "cpf": normalized,
        "authenticated": True,
        "customer": customer.model_dump(mode="json"),
        "auth_attempts": auth_attempts,
        "last_tool_calls": [
            {
                "name": "authenticate_customer",
                "args": {"cpf": "***", "birth_date": "***"},
                "result": "auth_ok",
            }
        ],
    }


@tool
def authenticate_customer(cpf: str, birth_date: str) -> str:
    """Autentica o cliente por CPF e data de nascimento (YYYY-MM-DD).

    Args:
        cpf: CPF do cliente.
        birth_date: Data no formato ISO ou brasileiro.

    Returns:
        ``auth_ok`` ou ``auth_failed`` (a mutação de estado ocorre no nó).
    """
    return f"authenticate_requested:{cpf}:{birth_date}"
