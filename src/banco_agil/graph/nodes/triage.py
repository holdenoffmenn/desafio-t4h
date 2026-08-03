"""Nó de triagem: saudação, coleta CPF/data e autenticação multi-turno."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from banco_agil.deps import AppDeps
from banco_agil.graph.nodes._compose import speak
from banco_agil.graph.state import SessionState
from banco_agil.llm.composer import MessageSpec
from banco_agil.tools.auth_tools import authenticate_customer_update
from banco_agil.tools.session_tools import end_conversation_update
from banco_agil.utils.conversation import (
    extract_cpf,
    extract_date,
    last_user_text,
    looks_like_end,
)
from banco_agil.utils.dates import parse_flexible_date

if TYPE_CHECKING:
    from banco_agil.llm.extract import LlmExtractor


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
            update["messages"] = speak(
                deps.composer,
                MessageSpec(
                    goal="encerrar o atendimento a pedido do cliente, de forma cordial",
                    fallback="Sem problemas. Encerrando o atendimento.",
                ),
            )
            return update

        cpf = state.get("cpf")
        extracted_cpf = extract_cpf(text)
        if extracted_cpf:
            cpf = extracted_cpf

        birth = extract_date(text)
        # "Sinal de data": há dígitos além do próprio CPF na mensagem — indica
        # que o cliente tentou informar uma data (inclusive junto do CPF, como
        # em "meu cpf é X, nasci em Y").
        attempted_date = _has_date_signal(text, extracted_cpf)
        # Fallback de interpretação: formatos fora do padrão (ex.: "19-091991"),
        # cobrindo também CPF + data no mesmo turno. Só aciona a LLM quando já há
        # um CPF e um sinal de data, evitando chamadas em turnos só de CPF.
        if birth is None and cpf and attempted_date and deps.nlu is not None:
            birth = _nlu_birth_date(deps.nlu, text)

        human_messages = [
            m for m in state.get("messages", []) if getattr(m, "type", None) == "human"
        ]

        # Primeira interação: saudação
        if len(human_messages) <= 1 and not cpf and not birth:
            return {
                "active_agent": "triage",
                "messages": speak(
                    deps.composer,
                    MessageSpec(
                        goal="saudar o cliente como assistente do Banco Ágil e iniciar "
                        "a autenticação",
                        fallback=(
                            "Olá! Sou o assistente virtual do Banco Ágil. "
                            "Para começar, informe seu CPF, por favor."
                        ),
                        ask="informe o CPF para começarmos",
                    ),
                ),
            }

        if not cpf:
            return {
                "active_agent": "triage",
                "messages": speak(
                    deps.composer,
                    MessageSpec(
                        goal="pedir o CPF do cliente para autenticação",
                        fallback="Por favor, informe seu CPF (apenas números ou com máscara).",
                        ask="informe o CPF (apenas números ou com máscara)",
                    ),
                ),
            }

        # Data ausente: distinguir "acabou de informar apenas o CPF" (primeira
        # solicitação) de "tentou informar a data mas não reconhecemos" — para
        # não repetir a mesma mensagem e sinalizar que o dado não foi entendido.
        if birth is None:
            if extracted_cpf and not attempted_date:
                return {
                    "active_agent": "triage",
                    "cpf": cpf,
                    "messages": speak(
                        deps.composer,
                        MessageSpec(
                            goal="agradecer o CPF e pedir a data de nascimento para concluir "
                            "a autenticação",
                            fallback=(
                                "Obrigado. Agora informe sua data de nascimento "
                                "(formato DD/MM/AAAA ou AAAA-MM-DD)."
                            ),
                            ask="informe a data de nascimento (formato DD/MM/AAAA ou AAAA-MM-DD)",
                        ),
                    ),
                }
            return {
                "active_agent": "triage",
                "cpf": cpf,
                "messages": speak(
                    deps.composer,
                    MessageSpec(
                        goal="informar de forma cordial que não foi possível reconhecer a "
                        "data de nascimento e pedir novamente em um formato válido, com "
                        "um exemplo concreto",
                        fallback=(
                            "Não consegui reconhecer a data de nascimento. "
                            "Poderia informar no formato DD/MM/AAAA? Por exemplo: 19/09/1991."
                        ),
                        ask="informe a data de nascimento no formato DD/MM/AAAA (ex.: 19/09/1991)",
                    ),
                ),
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
            auth_update["messages"] = speak(
                deps.composer,
                MessageSpec(
                    goal="cumprimentar o cliente pelo nome, confirmar a autenticação e "
                    "apresentar como pode ajudar",
                    fallback=(
                        f"Olá, {nome}! Autenticação concluída. Posso te ajudar com:\n"
                        "- **Limite de crédito** — consultar seu limite atual ou "
                        "solicitar um aumento\n"
                        "- **Câmbio** — cotação de moedas (dólar, euro, etc.)\n\n"
                        "Sobre qual desses assuntos você gostaria de falar?"
                    ),
                    facts={"nome do cliente": str(nome)},
                    options=(
                        "Limite de crédito — consultar o limite atual ou solicitar aumento",
                        "Câmbio — cotação de moedas (dólar, euro, etc.)",
                    ),
                    ask="sobre qual desses assuntos deseja falar",
                ),
            )
        else:
            attempts = auth_update["auth_attempts"]
            remaining = max(0, 3 - attempts)
            if remaining == 0:
                auth_update["messages"] = speak(
                    deps.composer,
                    MessageSpec(
                        goal="informar que não foi possível autenticar e que o atendimento "
                        "será encerrado por segurança",
                        fallback=(
                            "Não consegui validar seus dados. "
                            "Por segurança, este atendimento será encerrado."
                        ),
                    ),
                )
            else:
                auth_update["messages"] = speak(
                    deps.composer,
                    MessageSpec(
                        goal="informar falha na autenticação, quantas tentativas restam e "
                        "pedir CPF e data novamente",
                        fallback=(
                            "Não foi possível autenticar com esses dados. "
                            f"Você ainda tem {remaining} tentativa(s). "
                            "Informe novamente CPF e data de nascimento."
                        ),
                        facts={"tentativas restantes": str(remaining)},
                        ask="informe novamente o CPF e a data de nascimento",
                    ),
                )
                # limpa CPF para nova tentativa completa
                auth_update["cpf"] = None

        return auth_update

    return triage_node


def _has_date_signal(text: str, cpf_in_text: str | None) -> bool:
    """Indica se a mensagem traz sinal de data além do CPF.

    Remove o CPF encontrado no turno e verifica se ainda restam dígitos — o ano
    de nascimento sempre os contém. Serve para acionar o fallback de data
    (inclusive quando CPF e data vêm juntos) sem gastar chamadas em turnos que
    contêm apenas o CPF.

    Args:
        text: Mensagem do usuário.
        cpf_in_text: CPF extraído neste turno (ou ``None``).

    Returns:
        ``True`` se houver dígitos fora do CPF.
    """
    remainder = text.replace(cpf_in_text, " ") if cpf_in_text else text
    return any(char.isdigit() for char in remainder)


def _nlu_birth_date(nlu: LlmExtractor, text: str) -> date | None:
    """Interpreta a data de nascimento via LLM e converte para ``date``.

    Args:
        nlu: Extrator de linguagem natural.
        text: Mensagem do usuário.

    Returns:
        ``date`` válida ou ``None`` (sem data ou falha de parse/rede).
    """
    iso = nlu.birth_date(text)
    if not iso:
        return None
    try:
        return parse_flexible_date(iso)
    except (ValueError, TypeError):
        return None
