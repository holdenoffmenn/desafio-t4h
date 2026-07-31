"""Serviço de autenticação de clientes."""

from __future__ import annotations

from datetime import date

from banco_agil.domain.models import Customer
from banco_agil.domain.protocols import CustomerRepository
from banco_agil.utils.cpf import normalize_cpf


class AuthService:
    """Autentica clientes contra a base via CPF + data de nascimento.

    Colaboradores:
        CustomerRepository: fonte de dados de clientes (injetado por chamada).
    """

    def authenticate(
        self,
        cpf: str,
        birth_date: date,
        repo: CustomerRepository,
    ) -> Customer | None:
        """Valida credenciais e retorna o cliente autenticado.

        Fluxo: normaliza CPF → busca no repositório → compara data de nascimento.
        Sem efeito colateral.

        Args:
            cpf: CPF com ou sem máscara.
            birth_date: Data de nascimento informada.
            repo: Repositório de clientes.

        Returns:
            ``Customer`` se autenticado; ``None`` se CPF inexistente ou data errada.

        Raises:
            ValueError: Se o CPF não contiver dígitos.
        """
        normalized = normalize_cpf(cpf)
        customer = repo.find_by_cpf(normalized)
        if customer is None:
            return None
        if customer.data_nascimento != birth_date:
            return None
        return customer
