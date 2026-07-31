"""Nó de entrevista financeira: coleta estruturada e recálculo de score."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from banco_agil.deps import AppDeps
from banco_agil.domain.models import InterviewInput
from banco_agil.graph.state import SessionState
from banco_agil.tools.interview_tools import submit_interview_data_update
from banco_agil.utils.conversation import extract_money, last_user_text

Emprego = Literal["formal", "autônomo", "desempregado"]
_FIELDS = (
    "renda_mensal",
    "tipo_emprego",
    "despesas_fixas",
    "num_dependentes",
    "tem_dividas",
)


def make_interview_node(deps: AppDeps):
    """Factory do nó de entrevista.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def interview_node(state: SessionState) -> dict[str, Any]:
        """Coleta dados financeiros campo a campo e recalcula o score.

        Args:
            state: Estado da sessão.

        Returns:
            Update parcial; com dados completos chama ScoringService.
        """
        text = last_user_text(state.get("messages", []))
        # Primeira entrada: a mensagem é o aceite ("sim") — não deve ser
        # interpretada como resposta de campo (senão "sim" viraria tem_dividas).
        first_entry = state.get("interview_data") is None
        data: dict[str, object] = dict(state.get("interview_data") or {})

        if not first_entry:
            _merge_from_text(data, text)

        missing = [field for field in _FIELDS if field not in data]
        if missing:
            question = _question_for(missing[0])
            return {
                "active_agent": "interview",
                "interview_data": data,
                "interview_complete": False,
                "awaiting_interview": True,
                "messages": [AIMessage(content=question)],
            }

        try:
            interview = InterviewInput.model_validate(data)
        except ValidationError:
            return {
                "active_agent": "interview",
                "interview_data": data,
                "interview_complete": False,
                "awaiting_interview": True,
                "messages": [
                    AIMessage(
                        content=(
                            "Alguns dados parecem inválidos. "
                            "Informe a renda mensal em reais (ex.: 5000)."
                        )
                    )
                ],
            }

        customer = state.get("customer") or {}
        cpf = str(state.get("cpf") or customer.get("cpf") or "")
        score_before = int(customer.get("score", 0))  # type: ignore[arg-type]

        update = submit_interview_data_update(
            cpf=cpf,
            data=interview,
            scoring=deps.scoring,
            customer_repo=deps.customers,
            score_before=score_before,
        )
        new_score = update["last_score_calculation"]["score_after"]  # type: ignore[index]
        update["active_agent"] = "interview"
        update["awaiting_interview"] = False
        update["messages"] = [
            AIMessage(
                content=(
                    f"Entrevista concluída. Seu novo score é {new_score}. "
                    "Vou reanalisar sua solicitação de crédito."
                )
            )
        ]
        return update

    return interview_node


def _merge_from_text(data: dict[str, object], text: str) -> None:
    """Preenche o próximo campo pendente a partir do texto."""
    lowered = text.lower().strip()

    if "renda_mensal" not in data:
        money = extract_money(text)
        if money is not None:
            data["renda_mensal"] = money
            return

    if "tipo_emprego" not in data:
        emprego = _parse_emprego(lowered)
        if emprego is not None:
            data["tipo_emprego"] = emprego
            return

    if "despesas_fixas" not in data:
        money = extract_money(text)
        if money is not None:
            data["despesas_fixas"] = money
            return

    if "num_dependentes" not in data:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            data["num_dependentes"] = int(digits)
            return
        if any(w in lowered for w in ("nenhum", "zero", "não tenho", "nao tenho")):
            data["num_dependentes"] = 0
            return

    if "tem_dividas" not in data:
        # Negação é checada antes de "sim" para evitar falsos positivos
        # em frases como "não, não tenho dívidas".
        if any(w in lowered for w in ("não", "nao", "nenhuma", "sem", "zero", "quito")):
            data["tem_dividas"] = "não"
        elif any(w in lowered for w in ("sim", "tenho", "possuo", "devo", "com dívida")):
            data["tem_dividas"] = "sim"


def _parse_emprego(text: str) -> Emprego | None:
    """Mapeia texto livre para tipo de emprego."""
    if "desempreg" in text:
        return "desempregado"
    if "autônomo" in text or "autonomo" in text or "freelancer" in text:
        return "autônomo"
    if "formal" in text or "clt" in text or "carteira" in text:
        return "formal"
    return None


def _question_for(field: str) -> str:
    """Pergunta natural correspondente ao campo faltante."""
    questions = {
        "renda_mensal": "Qual é a sua renda mensal em reais?",
        "tipo_emprego": ("Qual o seu tipo de emprego? (formal, autônomo ou desempregado)"),
        "despesas_fixas": "Quais são suas despesas fixas mensais em reais?",
        "num_dependentes": "Quantos dependentes você possui?",
        "tem_dividas": "Você possui dívidas ativas? (sim ou não)",
    }
    return questions[field]
