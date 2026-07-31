"""Nó de entrevista financeira: coleta estruturada e recálculo de score."""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from banco_agil.deps import AppDeps
from banco_agil.domain.models import InterviewInput
from banco_agil.graph.state import SessionState
from banco_agil.llm.extract import LlmExtractor
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

        failed_field: str | None = None
        if not first_entry:
            # O campo perguntado no turno anterior é o primeiro pendente.
            target = next((field for field in _FIELDS if field not in data), None)
            if target is not None:
                value = _interpret_field(target, text, deps.nlu)
                if value is not None:
                    data[target] = value
                else:
                    failed_field = target

        missing = [field for field in _FIELDS if field not in data]
        if missing:
            # Se não interpretamos a resposta do campo perguntado, re-pergunta
            # 1x de forma amigável e, se persistir, mostra um erro curto com o
            # formato esperado. Do contrário, seguimos para o próximo campo.
            if failed_field is not None and failed_field == missing[0]:
                attempts = int(state.get("clarify_attempts", 0)) + 1
                question = _reask_field(missing[0], attempts)
            else:
                attempts = 0
                question = _question_for(missing[0])
            return {
                "active_agent": "interview",
                "interview_data": data,
                "interview_complete": False,
                "awaiting_interview": True,
                "clarify_attempts": attempts,
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
                "clarify_attempts": 0,
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
        update["clarify_attempts"] = 0
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


def _interpret_field(field: str, text: str, nlu: LlmExtractor | None) -> object | None:
    """Interpreta a resposta do campo perguntado (LLM primeiro, heurística depois).

    Quando há LLM configurado, ela **lê e interpreta** a resposta em linguagem
    natural; a heurística determinística só é usada como fallback (LLM ausente
    ou incapaz de interpretar), garantindo degradação graciosa.

    Args:
        field: Campo pendente sendo coletado.
        text: Resposta do usuário.
        nlu: Extrator LLM opcional.

    Returns:
        Valor normalizado do campo ou ``None`` se não interpretável.
    """
    if nlu is not None:
        value = _extract_with_llm(nlu, field, text)
        if value is not None:
            return value
    return _heuristic_field(field, text)


def _extract_with_llm(nlu: LlmExtractor, field: str, text: str) -> object | None:
    """Interpreta o campo perguntado via LLM.

    Args:
        nlu: Extrator de linguagem natural.
        field: Campo pendente sendo coletado.
        text: Resposta do usuário.

    Returns:
        Valor normalizado do campo ou ``None`` se a LLM não interpretar.
    """
    match field:
        case "renda_mensal":
            return nlu.money(text, field="renda_mensal")
        case "despesas_fixas":
            return nlu.money(text, field="despesas_fixas")
        case "tipo_emprego":
            return nlu.tipo_emprego(text)
        case "num_dependentes":
            return nlu.num_dependentes(text)
        case "tem_dividas":
            return nlu.tem_dividas(text)
        case _:  # pragma: no cover - campos são fixos em _FIELDS
            return None


def _heuristic_field(field: str, text: str) -> object | None:
    """Interpretação determinística de um campo (fallback sem LLM).

    Usa correspondência por tokens (não substring) em campos categóricos para
    evitar falsos positivos como ``"nada devo"`` → ``"sim"`` (o token ``devo``
    não deve sobrepor a negação ``nada``).

    Args:
        field: Campo pendente.
        text: Resposta do usuário.

    Returns:
        Valor normalizado ou ``None``.
    """
    lowered = text.lower().strip()
    tokens = set(re.findall(r"[a-zà-ú]+", lowered))

    match field:
        case "renda_mensal" | "despesas_fixas":
            return extract_money(text)
        case "tipo_emprego":
            return _parse_emprego(lowered)
        case "num_dependentes":
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits)
            if tokens & {"nenhum", "nenhuma", "zero"} or "não tenho" in lowered:
                return 0
            return None
        case "tem_dividas":
            negativos = {
                "não",
                "nao",
                "nenhuma",
                "nenhum",
                "sem",
                "zero",
                "quito",
                "quitei",
                "nada",
            }
            positivos = {"sim", "tenho", "possuo", "devo", "devendo", "endividado"}
            # Negação tem prioridade (ex.: "não, nada devo").
            if tokens & negativos:
                return "não"
            if tokens & positivos:
                return "sim"
            return None
        case _:  # pragma: no cover - campos são fixos em _FIELDS
            return None


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


_FORMAT_HINT: dict[str, str] = {
    "renda_mensal": "Informe apenas o valor em reais, por exemplo: 5000.",
    "tipo_emprego": "Responda formal, autônomo ou desempregado.",
    "despesas_fixas": "Informe apenas o valor em reais, por exemplo: 1500.",
    "num_dependentes": "Informe um número inteiro, por exemplo: 0, 1, 2.",
    "tem_dividas": "Responda apenas sim ou não.",
}


def _reask_field(field: str, attempts: int) -> str:
    """Re-pergunta o campo (1x amigável) e, se persistir, reforça o formato.

    Args:
        field: Campo pendente não interpretado.
        attempts: Nº de tentativas malsucedidas para este campo.

    Returns:
        Texto da pergunta ajustado ao nº de tentativas.
    """
    question = _question_for(field)
    if attempts <= 1:
        return f"Desculpe, não entendi. {question}"
    return f"Ainda não consegui entender. {question} {_FORMAT_HINT[field]}"
