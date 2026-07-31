"""Cálculo determinístico do score de crédito (fórmula do desafio)."""

from __future__ import annotations

from banco_agil.domain.models import InterviewInput

PESO_RENDA: float = 30.0
PESO_EMPREGO: dict[str, float] = {
    "formal": 300.0,
    "autônomo": 200.0,
    "desempregado": 0.0,
}
PESO_DEPENDENTES: dict[int, float] = {0: 100.0, 1: 80.0, 2: 60.0}
PESO_DEPENDENTES_3_MAIS: float = 30.0
PESO_DIVIDAS: dict[str, float] = {"sim": -100.0, "não": 100.0}


class ScoringService:
    """Calcula score de crédito 0–1000 a partir dos dados da entrevista.

    A fórmula segue o PDF do desafio. Dependentes ``>= 3`` usam peso 30
    (chave ``"3+"`` do enunciado).
    """

    def calculate_score(self, data: InterviewInput) -> int:
        """Calcula e faz clamp do score no intervalo [0, 1000].

        Args:
            data: Dados financeiros validados da entrevista.

        Returns:
            Score inteiro entre 0 e 1000.
        """
        # Regra do PDF: chave "3+" cobre num_dependentes >= 3.
        peso_dep = PESO_DEPENDENTES.get(data.num_dependentes, PESO_DEPENDENTES_3_MAIS)
        raw = (
            (data.renda_mensal / (data.despesas_fixas + 1.0)) * PESO_RENDA
            + PESO_EMPREGO[data.tipo_emprego]
            + peso_dep
            + PESO_DIVIDAS[data.tem_dividas]
        )
        return max(0, min(1000, int(raw)))
