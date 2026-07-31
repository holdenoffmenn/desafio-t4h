"""Nó de crédito: consulta de limite e solicitação de aumento."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage

from banco_agil.deps import AppDeps
from banco_agil.domain.models import Customer
from banco_agil.graph.state import SessionState

if TYPE_CHECKING:
    from banco_agil.llm.extract import LlmExtractor
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
                "clarify_attempts": 0,
                "messages": [
                    AIMessage(
                        content=(
                            "Tudo bem. Seu pedido permanece rejeitado por enquanto. "
                            "Posso ajudar com mais alguma coisa — crédito ou câmbio?"
                        )
                    )
                ],
            }

        awaiting_value = bool(state.get("awaiting_limit_value"))
        awaiting_confirm = bool(state.get("awaiting_increase_confirm"))
        wants_increase = wants_credit_increase(text)
        value_context = (
            awaiting_value
            or awaiting_confirm
            or wants_increase
            or state.get("pending_new_limit") is not None
        )
        new_limit = _interpret_limit(text, deps.nlu, value_context=value_context)

        # Usuário recusou iniciar o aumento
        if awaiting_confirm and looks_like_negative(text):
            return {
                "active_agent": "credit",
                "awaiting_increase_confirm": False,
                "awaiting_limit_value": False,
                "clarify_attempts": 0,
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
            or wants_increase
            or state.get("pending_new_limit") is not None
        ):
            return _process_increase(deps, state, customer, new_limit, reanalysis=False)

        # Aguardávamos um valor e não conseguimos interpretá-lo: re-pergunta 1x
        # de forma amigável e, se persistir, mostra um erro curto (sem inventar
        # nem assumir valores).
        if awaiting_value and new_limit is None:
            attempts = int(state.get("clarify_attempts", 0)) + 1
            return _reask_limit_value(customer, attempts)

        # Pedido de aumento sem valor → primeira pergunta do valor
        if wants_increase:
            return _ask_for_limit_value(customer)

        # Default: consulta + oferece aumento (abre contexto para "sim")
        obs = get_credit_limit_update(customer)
        return {
            "active_agent": "credit",
            **obs,
            "awaiting_increase_confirm": True,
            "awaiting_limit_value": False,
            "clarify_attempts": 0,
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


def _interpret_limit(
    text: str,
    nlu: LlmExtractor | None,
    *,
    value_context: bool,
) -> float | None:
    """Interpreta o novo limite (LLM primeiro, heurística como rede de segurança).

    Quando o contexto espera um valor e há LLM configurado, ela interpreta a
    resposta em linguagem natural (ex.: "vinte e cinco mil" → ``25000``). A
    heurística determinística só é usada como fallback (LLM ausente ou incapaz
    de interpretar), preservando a degradação graciosa.

    Args:
        text: Mensagem do usuário.
        nlu: Extrator LLM opcional.
        value_context: ``True`` quando o turno espera um valor de limite.

    Returns:
        Valor em reais ou ``None`` se não interpretável.
    """
    if value_context and nlu is not None:
        value = nlu.money(text, field="limite_credito")
        if value is not None:
            return value
    return extract_money(text)


def _ask_for_limit_value(customer: Customer) -> dict[str, Any]:
    """Pede o novo limite e marca o estado de espera do valor."""
    obs = get_credit_limit_update(customer)
    return {
        "active_agent": "credit",
        **obs,
        "awaiting_increase_confirm": False,
        "awaiting_limit_value": True,
        "clarify_attempts": 0,
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


def _reask_limit_value(customer: Customer, attempts: int) -> dict[str, Any]:
    """Re-pergunta o valor (1x amigável) e, se persistir, mostra erro curto.

    Args:
        customer: Cliente autenticado (para manter o contexto do limite atual).
        attempts: Nº de tentativas malsucedidas de interpretar o valor.

    Returns:
        Update mantendo ``awaiting_limit_value`` e o contador de tentativas.
    """
    obs = get_credit_limit_update(customer)
    if attempts <= 1:
        message = (
            "Desculpe, não consegui identificar o valor. Qual novo limite você "
            "gostaria de solicitar, em reais? (por exemplo: 30000)"
        )
    else:
        message = (
            "Ainda não entendi o valor. Envie apenas o número, sem outras "
            "palavras — por exemplo: 30000."
        )
    return {
        "active_agent": "credit",
        **obs,
        "awaiting_increase_confirm": False,
        "awaiting_limit_value": True,
        "clarify_attempts": attempts,
        "messages": [AIMessage(content=message)],
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
    update["clarify_attempts"] = 0

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

    # Reanálise pós-entrevista: a entrevista já ocorreu, então não a
    # re-oferecemos (evita loop). Limpamos o estado pendente e informamos o
    # teto que o score atualizado permite.
    if reanalysis:
        max_allowed = deps.score_limits.get_max_limit_for_score(customer.score)
        update["interview_complete"] = False
        update["pending_new_limit"] = None
        update["offered_interview"] = False
        update["interview_accepted"] = False
        if reason == "limite_menor_que_atual":
            body = (
                f"O valor solicitado (R$ {format_brl(new_limit)}) não é maior que seu "
                f"limite atual de R$ {format_brl(customer.limite_atual)}."
            )
        else:
            body = (
                f"Mesmo após a atualização, seu score ({customer.score}) permite um "
                f"limite de até R$ {format_brl(max_allowed)}, abaixo dos "
                f"R$ {format_brl(new_limit)} solicitados. Posso registrar um aumento "
                f"de até R$ {format_brl(max_allowed)}, se desejar."
            )
        update["messages"] = [AIMessage(content=f"{body} Posso ajudar com mais alguma coisa?")]
        return update

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
