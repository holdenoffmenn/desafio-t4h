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
    looks_like_affirmative,
    looks_like_end,
    looks_like_negative,
    wants_credit_increase,
)
from banco_agil.utils.currency import format_brl


def make_credit_node(deps: AppDeps):
    """Factory do nó de crédito.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def credit_node(state: SessionState) -> dict[str, Any]:
        """Processa consulta/aumento de limite e oferta de entrevista.

        Mantém contexto entre turnos via ``awaiting_increase_confirm`` e
        ``awaiting_limit_value`` para que "sim" / "quero" / "30000" completem
        o fluxo de aumento.

        Args:
            state: Estado da sessão.

        Returns:
            Update de estado + resposta natural.
        """
        text = last_user_text(state.get("messages", []))
        if looks_like_end(text):
            update = end_conversation_update()
            update["active_agent"] = "credit"
            update["awaiting_increase_confirm"] = False
            update["awaiting_limit_value"] = False
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

        # Reanálise pós-entrevista
        if state.get("interview_complete") and state.get("pending_new_limit") is not None:
            new_limit = float(state["pending_new_limit"])  # type: ignore[arg-type]
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
                "awaiting_increase_confirm": False,
                "awaiting_limit_value": False,
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
        awaiting_value = bool(state.get("awaiting_limit_value"))
        awaiting_confirm = bool(state.get("awaiting_increase_confirm"))

        # Usuário recusou iniciar o aumento
        if awaiting_confirm and looks_like_negative(text):
            return {
                "active_agent": "credit",
                "awaiting_increase_confirm": False,
                "awaiting_limit_value": False,
                "messages": [
                    AIMessage(
                        content=(
                            "Sem problemas. Seu limite permanece em "
                            f"R$ {format_brl(customer.limite_atual)}. "
                            "Posso ajudar com mais alguma coisa?"
                        )
                    )
                ],
            }

        # "sim" / "quero" após oferta → pedir o valor
        if awaiting_confirm and looks_like_affirmative(text) and new_limit is None:
            return _ask_for_limit_value(customer)

        # Valor informado enquanto aguardamos (ou junto com pedido de aumento)
        if new_limit is not None and (
            awaiting_value
            or awaiting_confirm
            or wants_credit_increase(text)
            or state.get("pending_new_limit") is not None
        ):
            return _process_increase(deps, state, customer, new_limit, reanalysis=False)

        # Pedido de aumento sem valor
        if wants_credit_increase(text) or (awaiting_value and new_limit is None):
            return _ask_for_limit_value(customer)

        # Default: consulta + oferece aumento (abre contexto para "sim")
        obs = get_credit_limit_update(customer)
        return {
            "active_agent": "credit",
            **obs,
            "awaiting_increase_confirm": True,
            "awaiting_limit_value": False,
            "messages": [
                AIMessage(
                    content=(
                        "Seu limite de crédito disponível é "
                        f"R$ {format_brl(customer.limite_atual)}. "
                        "Se quiser, posso registrar uma solicitação de aumento. "
                        "Deseja solicitar?"
                    )
                )
            ],
        }

    return credit_node


def _ask_for_limit_value(customer: Customer) -> dict[str, Any]:
    """Pede o novo limite e marca o estado de espera do valor."""
    obs = get_credit_limit_update(customer)
    return {
        "active_agent": "credit",
        **obs,
        "awaiting_increase_confirm": False,
        "awaiting_limit_value": True,
        "messages": [
            AIMessage(
                content=(
                    f"Seu limite atual é R$ {format_brl(customer.limite_atual)}. "
                    "Qual o novo limite que você deseja solicitar? "
                    "(informe apenas o valor, por exemplo: 30000)"
                )
            )
        ],
    }


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
    update["awaiting_increase_confirm"] = False
    update["awaiting_limit_value"] = False

    status = update["last_request_status"]
    if status == "aprovado":
        update["interview_complete"] = False
        prefix = "Após a atualização do seu score, " if reanalysis else ""
        update["messages"] = [
            AIMessage(
                content=(
                    f"{prefix}sua solicitação de aumento para R$ {format_brl(new_limit)} "
                    f"foi aprovada. Seu novo limite é R$ {format_brl(new_limit)}."
                )
            )
        ]
        return update

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
                    f"Sua solicitação de R$ {format_brl(new_limit)} foi rejeitada"
                    f"{f' ({reason})' if reason else ''}. "
                    "Posso conduzir uma entrevista financeira para atualizar seu score "
                    "e tentar novamente. Deseja seguir com a entrevista?"
                )
            )
        ]
    else:
        update["messages"] = [
            AIMessage(
                content=(
                    f"A solicitação de R$ {format_brl(new_limit)} continua rejeitada "
                    "para o score atual. Posso ajudar com mais alguma coisa?"
                )
            )
        ]
    return update
