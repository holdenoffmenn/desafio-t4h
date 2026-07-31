#!/usr/bin/env python3
"""Gera CSVs seed: clientes.csv e score_limite.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CLIENTES = [
    {
        "cpf": "52998224725",
        "data_nascimento": "1990-05-15",
        "nome": "Ana Souza",
        "limite_atual": "3000.00",
        "score": "450",
    },
    {
        "cpf": "39053344705",
        "data_nascimento": "1985-11-20",
        "nome": "Bruno Lima",
        "limite_atual": "8000.00",
        "score": "720",
    },
    {
        "cpf": "11144477735",
        "data_nascimento": "1978-03-08",
        "nome": "Carla Mendes",
        "limite_atual": "1500.00",
        "score": "280",
    },
    {
        "cpf": "12345678909",
        "data_nascimento": "1995-07-01",
        "nome": "Diego Alves",
        "limite_atual": "12000.00",
        "score": "850",
    },
    {
        "cpf": "98765432100",
        "data_nascimento": "1992-12-25",
        "nome": "Elena Costa",
        "limite_atual": "5000.00",
        "score": "610",
    },
]

SCORE_LIMITE = [
    {"score_min": "0", "score_max": "299", "limite_max_permitido": "1000.00"},
    {"score_min": "300", "score_max": "599", "limite_max_permitido": "5000.00"},
    {"score_min": "600", "score_max": "799", "limite_max_permitido": "15000.00"},
    {"score_min": "800", "score_max": "1000", "limite_max_permitido": "50000.00"},
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Escreve um CSV com cabeçalho."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> int:
    """Gera os arquivos seed em data/."""
    _write_csv(
        DATA_DIR / "clientes.csv",
        ["cpf", "data_nascimento", "nome", "limite_atual", "score"],
        CLIENTES,
    )
    _write_csv(
        DATA_DIR / "score_limite.csv",
        ["score_min", "score_max", "limite_max_permitido"],
        SCORE_LIMITE,
    )
    solicitacoes = DATA_DIR / "solicitacoes_aumento_limite.csv"
    if not solicitacoes.exists():
        _write_csv(
            solicitacoes,
            [
                "request_id",
                "cpf_cliente",
                "data_hora_solicitacao",
                "limite_atual",
                "novo_limite_solicitado",
                "status_pedido",
            ],
            [],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
