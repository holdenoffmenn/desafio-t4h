"""Nó de triagem: saudação, coleta CPF/data e autenticação multi-turno."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from banco_agil.deps import AppDeps
from banco_agil.graph.state import SessionState
from banco_agil.tools.auth_tools import authenticate_customer_update
from banco_agil.tools.session_tools import end_conversation_update
from banco_agil.utils.conversation import (
    extract_cpf,
    extract_date,
    last_user_text,
    looks_like_end,
)


def make_triage_node(deps: AppDeps):
    """Factory do nó de triagem.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def triage_node(state: SessionState) -> dict[str, Any]:
        """Coleta credenciais e autentica o cliente.

        Uma tentativa só é contada quando CPF + data completos falham.

        Args:
            state: Estado da sessão.

        Returns:
            Update de estado + mensagem ao cliente.
        """
        text = last_user_text(state.get("messages", []))
        if looks_like_end(text):
            update = end_conversation_update()
            update["active_agent"] = "triage"
            update["messages"] = [AIMessage(content="Sem problemas. Encerrando o atendimento.")]
            return update

        cpf = state.get("cpf")
        extracted_cpf = extract_cpf(text)
        if extracted_cpf:
            cpf = extracted_cpf

        birth = extract_date(text)
        human_messages = [
            m for m in state.get("messages", []) if getattr(m, "type", None) == "human"
        ]

        # Primeira interação: saudação
        if len(human_messages) <= 1 and not cpf and not birth:
            return {
                "active_agent": "triage",
                "messages": [
                    AIMessage(
                        content=(
                            "Olá! Sou o assistente virtual do Banco Ágil. "
                            "Para começar, informe seu CPF, por favor."
                        )
                    )
                ],
            }

        if not cpf:
            return {
                "active_agent": "triage",
                "messages": [
                    AIMessage(content="Por favor, informe seu CPF (apenas números ou com máscara).")
                ],
            }

        # Guarda CPF parcial entre turnos
        if birth is None:
            return {
                "active_agent": "triage",
                "cpf": cpf,
                "messages": [
                    AIMessage(
                        content=(
                            "Obrigado. Agora informe sua data de nascimento "
                            "(formato DD/MM/AAAA ou AAAA-MM-DD)."
                        )
                    )
                ],
            }

        auth_update = authenticate_customer_update(
            cpf=cpf,
            birth_date=birth,
            auth_service=deps.auth,
            repo=deps.customers,
            auth_attempts=state.get("auth_attempts", 0),
        )
        auth_update["active_agent"] = "triage"

        if auth_update["authenticated"]:
            nome = (auth_update.get("customer") or {}).get("nome", "cliente")
            auth_update["messages"] = [
                AIMessage(
                    content=(
                        f"Olá, {nome}! Autenticação concluída. "
                        "Posso ajudar com consulta/aumento de limite de crédito "
                        "ou cotação de câmbio. Como posso ajudar?"
                    )
                )
            ]
        else:
            attempts = auth_update["auth_attempts"]
            remaining = max(0, 3 - attempts)
            if remaining == 0:
                auth_update["messages"] = [
                    AIMessage(
                        content=(
                            "Não consegui validar seus dados. "
                            "Por segurança, este atendimento será encerrado."
                        )
                    )
                ]
            else:
                auth_update["messages"] = [
                    AIMessage(
                        content=(
                            "Não foi possível autenticar com esses dados. "
                            f"Você ainda tem {remaining} tentativa(s). "
                            "Informe novamente CPF e data de nascimento."
                        )
                    )
                ]
                # limpa CPF para nova tentativa completa
                auth_update["cpf"] = None

        return auth_update

    return triage_node
