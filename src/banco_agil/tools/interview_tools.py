"""Tools da entrevista de crédito."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from banco_agil.domain.models import InterviewInput
from banco_agil.domain.protocols import CustomerRepository
from banco_agil.domain.scoring import ScoringService


def submit_interview_data_update(
    *,
    cpf: str,
    data: InterviewInput,
    scoring: ScoringService,
    customer_repo: CustomerRepository,
    score_before: int,
) -> dict[str, Any]:
    """Calcula score, atualiza CSV e devolve update de estado.

    Args:
        cpf: CPF autenticado.
        data: Dados da entrevista validados.
        scoring: Serviço de cálculo.
        customer_repo: Persistência do score.
        score_before: Score anterior (observabilidade).

    Returns:
        Update com interview_complete e last_score_calculation.
    """
    new_score = scoring.calculate_score(data)
    customer_repo.update_score(cpf, new_score)
    customer = customer_repo.find_by_cpf(cpf)
    return {
        "interview_data": data.model_dump(mode="json"),
        "interview_complete": True,
        "interview_accepted": False,
        "customer": customer.model_dump(mode="json") if customer else None,
        "last_score_calculation": {
            "score_before": score_before,
            "score_after": new_score,
            "inputs": data.model_dump(mode="json"),
        },
        "last_tool_calls": [
            {
                "name": "submit_interview_data",
                "args": {"cpf": "***"},
                "result": {"score": new_score},
            }
        ],
    }


@tool
def submit_interview_data(
    renda_mensal: float,
    tipo_emprego: str,
    despesas_fixas: float,
    num_dependentes: int,
    tem_dividas: str,
) -> str:
    """Submete dados da entrevista para recálculo de score.

    Args:
        renda_mensal: Renda em reais.
        tipo_emprego: formal | autônomo | desempregado.
        despesas_fixas: Despesas fixas.
        num_dependentes: Quantidade de dependentes.
        tem_dividas: sim | não.
    """
    return "submit_interview_data_requested"
