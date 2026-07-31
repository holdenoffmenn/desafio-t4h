"""Nó de crédito: consulta de limite e solicitação de aumento."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from banco_agil.deps import AppDeps
from banco_agil.domain.models import Customer
from banco_agil.graph.state import SessionState
from banco_agil.tools.credit_tools import (
    get_credit_limit_update,
    offer_credit_interview_update,
    request_limit_increase_update,
)
from banco_agil.tools.session_tools import end_conversation_update
from banco_agil.utils.conversation import (
    extract_money,
    last_user_text,
    looks_like_end,
    looks_like_negative,
)


def make_credit_node(deps: AppDeps):
    """Factory do nó de crédito.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def credit_node(state: SessionState) -> dict[str, Any]:
        """Processa consulta/aumento de limite e oferta de entrevista.

        Args:
            state: Estado da sessão.

        Returns:
            Update de estado + resposta natural.
        """
        text = last_user_text(state.get("messages", []))
        if looks_like_end(text):
            update = end_conversation_update()
            update["active_agent"] = "credit"
            update["messages"] = [AIMessage(content="Atendimento encerrado. Até logo!")]
            return update

        customer_data = state.get("customer")
        if not customer_data:
            return {
                "active_agent": "credit",
                "messages": [
                    AIMessage(content="Preciso autenticar você antes de falar sobre crédito.")
                ],
            }

        customer = Customer.model_validate(customer_data)

        # Reanálise pós-entrevista: se há pending_new_limit e interview_complete
        if state.get("interview_complete") and state.get("pending_new_limit") is not None:
            new_limit = float(state["pending_new_limit"])  # type: ignore[arg-type]
            # Recarrega customer atualizado (score novo)
            refreshed = deps.customers.find_by_cpf(customer.cpf)
            if refreshed is not None:
                customer = refreshed
            return _process_increase(deps, state, customer, new_limit, reanalysis=True)

        # Recusa de entrevista após oferta
        if state.get("offered_interview") and looks_like_negative(text):
            return {
                "active_agent": "credit",
                "offered_interview": False,
                "interview_accepted": False,
                "messages": [
                    AIMessage(
                        content=(
                            "Tudo bem. Seu pedido permanece rejeitado por enquanto. "
                            "Posso ajudar com mais alguma coisa — crédito ou câmbio?"
                        )
                    )
                ],
            }

        new_limit = extract_money(text)
        wants_increase = any(
            k in text.lower()
            for k in ("aument", "elevar", "quero", "solicitar", "pedir", "novo limite")
        )

        if new_limit is not None and (wants_increase or state.get("pending_new_limit") is not None):
            return _process_increase(deps, state, customer, new_limit, reanalysis=False)

        if wants_increase and new_limit is None:
            obs = get_credit_limit_update(customer)
            return {
                "active_agent": "credit",
                **obs,
                "messages": [
                    AIMessage(
                        content=(
                            f"Seu limite atual é R$ {customer.limite_atual:,.2f}. "
                            "Qual o novo limite que você deseja solicitar?"
                        )
                        .replace(",", "X")
                        .replace(".", ",")
                        .replace("X", ".")
                    )
                ],
            }

        # Default: consulta de limite
        obs = get_credit_limit_update(customer)
        return {
            "active_agent": "credit",
            **obs,
            "messages": [
                AIMessage(
                    content=(
                        f"Seu limite de crédito disponível é R$ {customer.limite_atual:,.2f}. "
                        "Se quiser, posso registrar uma solicitação de aumento."
                    )
                    .replace(",", "X")
                    .replace(".", ",")
                    .replace("X", ".")
                )
            ],
        }

    return credit_node


def _process_increase(
    deps: AppDeps,
    state: SessionState,
    customer: Customer,
    new_limit: float,
    *,
    reanalysis: bool,
) -> dict[str, Any]:
    """Executa o ciclo pendente → avaliação → status final."""
    update = request_limit_increase_update(
        customer=customer,
        new_limit=new_limit,
        credit_service=deps.credit_limit,
        score_repo=deps.score_limits,
        request_repo=deps.credit_requests,
        customer_repo=deps.customers,
    )
    update["active_agent"] = "credit"
    update["pending_new_limit"] = new_limit

    status = update["last_request_status"]
    if status == "aprovado":
        update["interview_complete"] = False
        prefix = "Após a atualização do seu score, " if reanalysis else ""
        update["messages"] = [
            AIMessage(
                content=(
                    f"{prefix}sua solicitação de aumento para R$ {new_limit:,.2f} "
                    f"foi aprovada. Seu novo limite é R$ {new_limit:,.2f}."
                )
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
        ]
        return update

    # rejeitado
    reason = ""
    tool_result = update["last_tool_calls"][0]["result"]
    if isinstance(tool_result, dict) and tool_result.get("reason"):
        reason = str(tool_result["reason"])

    already_offered = bool(state.get("offered_interview"))
    if not already_offered:
        offer = offer_credit_interview_update()
        update.update(offer)
        update["messages"] = [
            AIMessage(
                content=(
                    f"Sua solicitação de R$ {new_limit:,.2f} foi rejeitada"
                    f"{f' ({reason})' if reason else ''}. "
                    "Posso conduzir uma entrevista financeira para atualizar seu score "
                    "e tentar novamente. Deseja seguir com a entrevista?"
                )
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
        ]
    else:
        update["messages"] = [
            AIMessage(
                content=(
                    f"A solicitação de R$ {new_limit:,.2f} continua rejeitada "
                    "para o score atual. Posso ajudar com mais alguma coisa?"
                )
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
        ]
    return update
