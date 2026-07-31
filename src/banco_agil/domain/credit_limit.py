"""Avaliação de solicitações de aumento de limite de crédito."""

from __future__ import annotations

from banco_agil.domain.models import CreditDecision, Customer
from banco_agil.domain.protocols import ScoreLimitRepository


class CreditLimitService:
    """Decide aprovação/rejeição de aumento de limite com base no score.

    Colaboradores:
        ScoreLimitRepository: tabela score → limite máximo permitido.

    Nota de negócio:
        A regra ``new_limit <= limite_atual → rejeitado`` **não** consta no PDF;
        é decisão intencional para pedidos que não representam aumento real
        (``reason=limite_menor_que_atual``).
    """

    def evaluate_request(
        self,
        customer: Customer,
        new_limit: float,
        score_repo: ScoreLimitRepository,
    ) -> CreditDecision:
        """Decide aprovação de aumento de limite com base no score do cliente.

        Regra: aprova se ``new_limit`` > limite atual E <= limite máximo
        permitido para a faixa de score do cliente (tabela score_limite.csv).

        Args:
            customer: Cliente autenticado (contém score e limite atual).
            new_limit: Novo limite solicitado, em reais.
            score_repo: Fonte da tabela score → limite máximo.

        Returns:
            CreditDecision com status (``aprovado``|``rejeitado``) e motivo/valor.

        Raises:
            ScoreTableError: Se a faixa de score do cliente não existir na tabela.
        """
        if new_limit <= customer.limite_atual:
            return CreditDecision(
                status="rejeitado",
                reason="limite_menor_que_atual",
            )

        max_allowed = score_repo.get_max_limit_for_score(customer.score)
        if new_limit > max_allowed:
            return CreditDecision(
                status="rejeitado",
                reason="score_insuficiente",
            )

        return CreditDecision(
            status="aprovado",
            approved_limit=new_limit,
        )
