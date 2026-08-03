"""Nó de câmbio: consulta cotação via FxClient."""

from __future__ import annotations

from typing import Any

from banco_agil.deps import AppDeps
from banco_agil.domain.errors import FxPairNotFoundError, FxUnavailableError
from banco_agil.graph.nodes._compose import speak
from banco_agil.graph.state import SessionState
from banco_agil.llm.composer import MessageSpec
from banco_agil.tools.exchange_tools import get_exchange_rate_update
from banco_agil.tools.session_tools import end_conversation_update
from banco_agil.utils.conversation import (
    extract_currency_code,
    last_user_text,
    looks_like_end,
)


def make_exchange_node(deps: AppDeps):
    """Factory do nó de câmbio.

    Args:
        deps: Dependências da aplicação.

    Returns:
        Função de nó LangGraph.
    """

    def exchange_node(state: SessionState) -> dict[str, Any]:
        """Consulta e apresenta a cotação solicitada.

        Args:
            state: Estado da sessão.

        Returns:
            Update com cotação ou mensagem de erro controlada.
        """
        text = last_user_text(state.get("messages", []))
        if looks_like_end(text):
            update = end_conversation_update()
            update["active_agent"] = "exchange"
            update["messages"] = speak(
                deps.composer,
                MessageSpec(
                    goal="encerrar o atendimento de câmbio de forma cordial",
                    fallback="Atendimento encerrado. Até logo!",
                ),
            )
            return update

        currency = extract_currency_code(text)
        if currency is None and deps.nlu is not None:
            currency = deps.nlu.currency(text)
        if currency is None:
            return _ask_currency(deps)

        try:
            obs = get_exchange_rate_update(deps.fx, currency)
            rate = obs["last_tool_calls"][0]["result"]
            assert isinstance(rate, dict)
            bid = float(rate["bid"])
            ts = rate.get("timestamp", "")
            bid_str = f"R$ {bid:.4f}".replace(".", ",")
            msg = (
                f"Cotação atual {currency}/BRL: compra {bid_str}"
                + (f" (atualizado em {ts})." if ts else ".")
                + " Posso ajudar com mais alguma coisa?"
            )
            facts = {f"cotação de compra {currency}/BRL": bid_str}
            if ts:
                facts["atualizado em"] = str(ts)
            return {
                "active_agent": "exchange",
                **obs,
                "messages": speak(
                    deps.composer,
                    MessageSpec(
                        goal=f"apresentar a cotação atual de {currency}/BRL e oferecer mais ajuda",
                        fallback=msg,
                        facts=facts,
                        ask="se posso ajudar com mais alguma coisa",
                    ),
                ),
            }
        except FxPairNotFoundError as exc:
            return {
                "active_agent": "exchange",
                "error": str(exc),
                "messages": speak(
                    deps.composer,
                    MessageSpec(
                        goal=(
                            f"informar de forma cordial que não há cotação disponível "
                            f"para a moeda {currency} contra o real, e oferecer cotar "
                            "outra moeda ou outro assunto"
                        ),
                        fallback=(
                            f"Não encontrei cotação disponível para {currency}/BRL. "
                            "Se quiser, posso tentar outra moeda ou ajudar com outro assunto."
                        ),
                        facts={"moeda solicitada": currency},
                    ),
                ),
            }
        except (FxUnavailableError, ValueError) as exc:
            return {
                "active_agent": "exchange",
                "error": str(exc),
                "messages": speak(
                    deps.composer,
                    MessageSpec(
                        goal="informar de forma cordial que a cotação está temporariamente "
                        "indisponível e sugerir tentar novamente ou outro assunto",
                        fallback=(
                            "Não consegui obter a cotação agora. "
                            "Tente novamente em instantes ou peça outro assunto."
                        ),
                    ),
                ),
            }

    return exchange_node


def _ask_currency(deps: AppDeps) -> dict[str, Any]:
    """Pede ao cliente qual moeda cotar quando a intenção não indica uma.

    Preferimos perguntar a assumir dólar por padrão: o silêncio levava o câmbio
    a ignorar pedidos como "peso argentino" e devolver USD.

    Args:
        deps: Dependências da aplicação (usa o compositor de mensagens).

    Returns:
        Update mantendo o skill de câmbio ativo e solicitando a moeda.
    """
    return {
        "active_agent": "exchange",
        "messages": speak(
            deps.composer,
            MessageSpec(
                goal="pedir de forma cordial e natural qual moeda o cliente deseja cotar, "
                "deixando claro que atendemos diversas moedas e citando alguns exemplos "
                "SEM dar a entender que são as únicas disponíveis",
                fallback=(
                    "Sobre qual moeda você gostaria da cotação? Atendemos diversas moedas — "
                    "por exemplo, dólar (USD), euro (EUR), libra (GBP), iene (JPY) ou "
                    "peso argentino (ARS)."
                ),
            ),
        ),
    }
