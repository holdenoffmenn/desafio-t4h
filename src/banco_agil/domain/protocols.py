"""Protocols (interfaces) da camada de domínio para Injeção de Dependência."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from banco_agil.domain.models import Customer


class CustomerRepository(Protocol):
    """Contrato de persistência de clientes."""

    def find_by_cpf(self, cpf: str) -> Customer | None:
        """Busca cliente pelo CPF normalizado."""
        ...

    def authenticate(self, cpf: str, birth_date: date) -> Customer | None:
        """Autentica por CPF + data de nascimento."""
        ...

    def update_score(self, cpf: str, new_score: int) -> None:
        """Atualiza o score do cliente."""
        ...

    def update_limit(self, cpf: str, new_limit: float) -> None:
        """Atualiza o limite de crédito do cliente."""
        ...


class ScoreLimitRepository(Protocol):
    """Contrato da tabela score → limite máximo."""

    def get_max_limit_for_score(self, score: int) -> float:
        """Retorna o limite máximo permitido para o score informado."""
        ...
