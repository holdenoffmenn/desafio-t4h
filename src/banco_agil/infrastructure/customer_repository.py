"""Repositório CSV de clientes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from banco_agil.domain.errors import CustomerNotFoundError, PersistenceError
from banco_agil.domain.models import Customer
from banco_agil.infrastructure.csv_repository import AtomicCsvStore
from banco_agil.utils.cpf import normalize_cpf
from banco_agil.utils.dates import parse_flexible_date

CUSTOMER_FIELDS = ["cpf", "data_nascimento", "nome", "limite_atual", "score"]


class CsvCustomerRepository:
    """Implementação CSV do contrato CustomerRepository.

    Colaboradores:
        AtomicCsvStore: leitura/escrita atômica do arquivo ``clientes.csv``.
    """

    def __init__(self, path: Path) -> None:
        """Inicializa o repositório.

        Args:
            path: Caminho para ``clientes.csv``.
        """
        self._store = AtomicCsvStore(path)

    def find_by_cpf(self, cpf: str) -> Customer | None:
        """Busca cliente pelo CPF normalizado.

        Args:
            cpf: CPF (será normalizado).

        Returns:
            Customer ou None se não encontrado.
        """
        normalized = normalize_cpf(cpf)
        for row in self._store.read_all():
            if normalize_cpf(row["cpf"]) == normalized:
                return self._to_customer(row)
        return None

    def authenticate(self, cpf: str, birth_date: date) -> Customer | None:
        """Autentica por CPF + data de nascimento.

        Args:
            cpf: CPF informado.
            birth_date: Data de nascimento informada.

        Returns:
            Customer autenticado ou None.
        """
        customer = self.find_by_cpf(cpf)
        if customer is None:
            return None
        if customer.data_nascimento != birth_date:
            return None
        return customer

    def update_score(self, cpf: str, new_score: int) -> None:
        """Atualiza o score do cliente (read-modify-write atômico).

        Args:
            cpf: CPF do cliente.
            new_score: Novo score (0–1000).

        Raises:
            CustomerNotFoundError: Se o CPF não existir.
            PersistenceError: Em falha de I/O.
        """
        self._update_field(cpf, "score", str(new_score))

    def update_limit(self, cpf: str, new_limit: float) -> None:
        """Atualiza o limite de crédito do cliente.

        Args:
            cpf: CPF do cliente.
            new_limit: Novo limite em reais.

        Raises:
            CustomerNotFoundError: Se o CPF não existir.
            PersistenceError: Em falha de I/O.
        """
        self._update_field(cpf, "limite_atual", f"{new_limit:.2f}")

    def _update_field(self, cpf: str, field: str, value: str) -> None:
        """Atualiza um campo do cliente e persiste atomicamente."""
        normalized = normalize_cpf(cpf)
        rows = self._store.read_all()
        found = False
        for row in rows:
            if normalize_cpf(row["cpf"]) == normalized:
                row[field] = value
                found = True
                break
        if not found:
            raise CustomerNotFoundError(f"Cliente não encontrado: {normalized}")
        self._store.write_all(CUSTOMER_FIELDS, rows)

    @staticmethod
    def _to_customer(row: dict[str, str]) -> Customer:
        """Converte linha CSV em Customer tipado."""
        try:
            return Customer(
                cpf=normalize_cpf(row["cpf"]),
                data_nascimento=parse_flexible_date(row["data_nascimento"]),
                nome=row["nome"],
                limite_atual=float(row["limite_atual"]),
                score=int(row["score"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise PersistenceError(f"Linha de cliente inválida: {row}") from exc
