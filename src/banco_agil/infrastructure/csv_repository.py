"""Leitura e escrita atômica de arquivos CSV com file lock."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from banco_agil.domain.errors import PersistenceError


class AtomicCsvStore:
    """Store CSV com concorrência (filelock) e escrita atômica (temp + replace).

    Responsabilidade:
        Garantir que rewrites de CSV nunca deixem o arquivo destino truncado
        (``os.replace`` atômico no mesmo filesystem) e que processos concorrentes
        não se sobreponham (``FileLock``).
    """

    def __init__(self, path: Path, lock_timeout: float = 10.0) -> None:
        """Inicializa o store apontando para um arquivo CSV.

        Args:
            path: Caminho do arquivo CSV.
            lock_timeout: Timeout em segundos para adquirir o lock.
        """
        self._path = path
        self._lock_timeout = lock_timeout
        self._lock_path = path.with_suffix(path.suffix + ".lock")

    @property
    def path(self) -> Path:
        """Caminho do CSV gerenciado."""
        return self._path

    def read_all(self) -> list[dict[str, str]]:
        """Lê todas as linhas do CSV como dicionários.

        Returns:
            Lista de linhas (chaves = cabeçalho).

        Raises:
            PersistenceError: Se o arquivo não existir ou estiver corrompido.
        """
        if not self._path.exists():
            raise PersistenceError(f"CSV não encontrado: {self._path}")
        try:
            with self._path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except (OSError, csv.Error) as exc:
            raise PersistenceError(f"Falha ao ler CSV {self._path}: {exc}") from exc

    def write_all(self, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        """Reescreve o CSV de forma atômica sob lock.

        Fluxo: adquirir lock → escrever ``.tmp`` no mesmo diretório →
        ``os.replace`` → liberar lock. Em falha, o arquivo destino permanece intacto.

        Args:
            fieldnames: Ordem das colunas do cabeçalho.
            rows: Linhas a persistir.

        Raises:
            PersistenceError: Em timeout de lock ou falha de I/O.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._lock_path), timeout=self._lock_timeout)
        try:
            with lock:
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{self._path.name}.",
                    suffix=".tmp",
                    dir=str(self._path.parent),
                )
                tmp_path = Path(tmp_name)
                try:
                    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=fieldnames,
                            extrasaction="ignore",
                        )
                        writer.writeheader()
                        for row in rows:
                            writer.writerow({k: row.get(k, "") for k in fieldnames})
                    os.replace(tmp_path, self._path)
                except Exception:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    raise
        except Timeout as exc:
            raise PersistenceError(f"Timeout ao adquirir lock de {self._path}") from exc
        except OSError as exc:
            raise PersistenceError(f"Falha ao escrever CSV {self._path}: {exc}") from exc

    def append_row(self, fieldnames: list[str], row: dict[str, Any]) -> None:
        """Anexa uma linha ao CSV (read-modify-write atômico).

        Se o arquivo não existir, cria com cabeçalho.

        Args:
            fieldnames: Colunas do CSV.
            row: Linha a anexar.
        """
        rows: list[dict[str, Any]] = list(self.read_all()) if self._path.exists() else []
        rows.append(row)
        self.write_all(fieldnames, rows)
