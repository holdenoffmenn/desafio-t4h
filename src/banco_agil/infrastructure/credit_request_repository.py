"""Repositório CSV de solicitações de aumento de limite."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from banco_agil.domain.errors import PersistenceError
from banco_agil.domain.models import CreditRequest
from banco_agil.infrastructure.csv_repository import AtomicCsvStore

REQUEST_FIELDS = [
    "request_id",
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]

StatusPedido = Literal["pendente", "aprovado", "rejeitado"]


class CsvCreditRequestRepository:
    """Persiste pedidos de aumento com ciclo ``pendente → aprovado|rejeitado``.

    Colaboradores:
        AtomicCsvStore: escrita atômica de ``solicitacoes_aumento_limite.csv``.
    """

    def __init__(self, path: Path) -> None:
        """Inicializa o repositório.

        Args:
            path: Caminho para ``solicitacoes_aumento_limite.csv``.
        """
        self._store = AtomicCsvStore(path)

    def create(
        self,
        *,
        cpf_cliente: str,
        limite_atual: float,
        novo_limite_solicitado: float,
        data_hora: datetime | None = None,
    ) -> str:
        """Registra um pedido como ``pendente`` e retorna o ``request_id``.

        Args:
            cpf_cliente: CPF autenticado.
            limite_atual: Limite no momento da solicitação.
            novo_limite_solicitado: Valor pedido.
            data_hora: Timestamp opcional (default: agora UTC).

        Returns:
            UUID do pedido criado.
        """
        request_id = str(uuid4())
        ts = data_hora or datetime.now(UTC)
        row = {
            "request_id": request_id,
            "cpf_cliente": cpf_cliente,
            "data_hora_solicitacao": ts.isoformat(),
            "limite_atual": f"{limite_atual:.2f}",
            "novo_limite_solicitado": f"{novo_limite_solicitado:.2f}",
            "status_pedido": "pendente",
        }
        self._store.append_row(REQUEST_FIELDS, row)
        return request_id

    def update_status(self, request_id: str, status: StatusPedido) -> None:
        """Atualiza o status da **mesma** linha do pedido.

        Args:
            request_id: ID retornado por :meth:`create`.
            status: Novo status (``aprovado`` ou ``rejeitado`` tipicamente).

        Raises:
            PersistenceError: Se o ``request_id`` não existir.
        """
        if not self._store.path.exists():
            raise PersistenceError("Arquivo de solicitações inexistente.")

        rows = self._store.read_all()
        found = False
        for row in rows:
            if row.get("request_id") == request_id:
                row["status_pedido"] = status
                found = True
                break
        if not found:
            raise PersistenceError(f"Solicitação não encontrada: {request_id}")
        self._store.write_all(REQUEST_FIELDS, rows)

    def get_by_id(self, request_id: str) -> CreditRequest | None:
        """Recupera um pedido pelo ID.

        Args:
            request_id: Identificador do pedido.

        Returns:
            CreditRequest ou None.
        """
        if not self._store.path.exists():
            return None
        for row in self._store.read_all():
            if row.get("request_id") == request_id:
                return CreditRequest(
                    request_id=row["request_id"],
                    cpf_cliente=row["cpf_cliente"],
                    data_hora_solicitacao=datetime.fromisoformat(row["data_hora_solicitacao"]),
                    limite_atual=float(row["limite_atual"]),
                    novo_limite_solicitado=float(row["novo_limite_solicitado"]),
                    status_pedido=row["status_pedido"],  # type: ignore[arg-type]
                )
        return None
