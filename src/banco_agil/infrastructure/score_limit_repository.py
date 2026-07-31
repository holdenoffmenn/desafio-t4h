"""Repositório CSV da tabela score → limite máximo."""

from __future__ import annotations

from pathlib import Path

from banco_agil.domain.errors import ScoreTableError
from banco_agil.domain.models import ScoreBand
from banco_agil.infrastructure.csv_repository import AtomicCsvStore


class CsvScoreLimitRepository:
    """Lê faixas de score e resolve o limite máximo permitido.

    Colaboradores:
        AtomicCsvStore: leitura de ``score_limite.csv``.
    """

    def __init__(self, path: Path) -> None:
        """Inicializa o repositório.

        Args:
            path: Caminho para ``score_limite.csv``.
        """
        self._store = AtomicCsvStore(path)

    def list_bands(self) -> list[ScoreBand]:
        """Lista todas as faixas de score.

        Returns:
            Lista de ScoreBand ordenada por score_min.
        """
        bands = [
            ScoreBand(
                score_min=int(row["score_min"]),
                score_max=int(row["score_max"]),
                limite_max_permitido=float(row["limite_max_permitido"]),
            )
            for row in self._store.read_all()
        ]
        return sorted(bands, key=lambda b: b.score_min)

    def get_max_limit_for_score(self, score: int) -> float:
        """Retorna o limite máximo permitido para o score informado.

        Args:
            score: Score do cliente (0–1000).

        Returns:
            Limite máximo aprovável na faixa correspondente.

        Raises:
            ScoreTableError: Se nenhuma faixa cobrir o score.
        """
        for band in self.list_bands():
            if band.score_min <= score <= band.score_max:
                return band.limite_max_permitido
        raise ScoreTableError(f"Nenhuma faixa de score para valor {score}.")
