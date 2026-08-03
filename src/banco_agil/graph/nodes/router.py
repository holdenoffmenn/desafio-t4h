"""Nó hub: roteamento de intenção com a LLM como intérprete primária.

Filosofia: a LLM **entende** a intenção (com o contexto da conversa) e o código
**decide** o fluxo. Não há mais classificação por palavras-chave nem menu de bot
como último recurso — se a LLM falhar (após retry), devolvemos uma mensagem de
erro amigável; se a intenção for genuinamente ambígua, pedimos uma clarificação
curta e natural. O roteador semântico (ML) permanece apenas como rede de apoio,
inclusive quando a LLM está desligada.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AnyMessage

from banco_agil.deps import AppDeps
from banco_agil.graph.nodes._compose import speak
from banco_agil.graph.state import Intent, SessionState
from banco_agil.llm.composer import MessageSpec
from banco_agil.llm.intent import IntentFallback
from banco_agil.tools.session_tools import end_conversation_update

if TYPE_CHECKING:
    from banco_agil.llm.composer import MessageComposer
from banco_agil.utils.conversation import (
    extract_money,
    last_user_text,
    looks_like_affirmative,
    looks_like_end,
)

_ACTIONABLE: frozenset[str] = frozenset({"credit", "exchange", "interview", "end"})
_MAX_CONTEXT_MESSAGES = 6


def make_router_node(
    deps: AppDeps,
    *,
    llm_fallback: IntentFallback | None = None,
):
    """Factory do nó router (hub).

    Args:
        deps: Dependências da aplicação.
        llm_fallback: Classificador de intenção via LLM (contexto + retry).
            ``None`` desliga a LLM e usa apenas o roteador semântico.

    Returns:
        Função de nó LangGraph.
    """

    def router_node(state: SessionState) -> dict[str, Any]:
        """Decide a intenção do turno e encaminha para o skill adequado.

        Ordem de decisão:
            1. Guardas de **continuidade de fluxo** (coleta em andamento): a
               próxima mensagem é um dado, não uma nova intenção.
            2. **LLM** como intérprete primária (com contexto + retry). Falha
               após retry → mensagem de erro amigável.
            3. Rede de apoio **semântica** (ML) quando a LLM está ausente ou
               devolve intenção ambígua.
            4. Clarificação curta e natural (sem menu de bot).

        Args:
            state: Estado da sessão.

        Returns:
            Update com intent, confiança e fonte do roteamento.
        """
        messages = state.get("messages", [])
        text = last_user_text(messages)

        continuity = _continuity_route(state, text)
        if continuity is not None:
            return continuity

        if llm_fallback is not None:
            result = llm_fallback(text, _recent_context(messages))
            if result.failed:
                return _llm_error(deps.composer)
            if result.intent in _ACTIONABLE:
                return _routed(result.intent, source="llm_fallback", confidence=None)

        route = deps.intent_router.predict(text)
        if route is not None and route.intent in _ACTIONABLE:
            return _routed(route.intent, source="semantic", confidence=route.confidence)

        return _clarify(deps.composer)

    return router_node


def _continuity_route(state: SessionState, text: str) -> dict[str, Any] | None:
    """Resolve o roteamento por continuidade de fluxo (estado da sessão).

    São situações em que a próxima mensagem do cliente pertence a um fluxo já
    em andamento (encerramento explícito, entrevista/valor em coleta, aceite de
    entrevista) — portanto não devem ser reclassificadas como nova intenção.

    Args:
        state: Estado da sessão.
        text: Última mensagem do usuário.

    Returns:
        Update de roteamento se houver continuidade; ``None`` caso contrário.
    """
    if looks_like_end(text):
        update = end_conversation_update()
        update.update(
            {
                "active_agent": "router",
                "intent": "end",
                "route_source": "context",
                "route_confidence": 1.0,
            }
        )
        return update

    # Entrevista em andamento: "8000"/"formal"/"não" são respostas de campo,
    # não intenções. Sem esta guarda a entrevista nunca completaria.
    if state.get("awaiting_interview") and not state.get("interview_complete"):
        return _routed("interview", source="context", confidence=1.0)

    # Aumento de limite em coleta de valor/confirmação.
    if state.get("awaiting_limit_value") or (
        state.get("awaiting_increase_confirm")
        and (looks_like_affirmative(text) or extract_money(text) is not None)
    ):
        return _routed("credit", source="context", confidence=1.0)

    # Aceite da entrevista oferecida após rejeição de crédito.
    if (
        state.get("offered_interview")
        and not state.get("interview_complete")
        and looks_like_affirmative(text)
    ):
        update = _routed("interview", source="context", confidence=1.0)
        update["interview_accepted"] = True
        return update

    return None


def _recent_context(
    messages: Sequence[AnyMessage],
    max_messages: int = _MAX_CONTEXT_MESSAGES,
) -> str:
    """Monta uma transcrição curta dos turnos recentes (para desambiguar).

    Inclui as mensagens anteriores à última do usuário, que já é enviada
    separadamente como "ÚLTIMA MENSAGEM" ao classificador.

    Args:
        messages: Histórico de mensagens da sessão.
        max_messages: Nº máximo de mensagens de contexto a incluir.

    Returns:
        Transcrição formatada ("Cliente:"/"Assistente:") ou string vazia.
    """
    prior = list(messages)[:-1]
    if not prior:
        return ""

    lines: list[str] = []
    for message in prior[-max_messages:]:
        role = "Cliente" if getattr(message, "type", None) == "human" else "Assistente"
        content = getattr(message, "content", "")
        body = content if isinstance(content, str) else str(content)
        body = body.strip()
        if body:
            lines.append(f"{role}: {body}")
    if not lines:
        return ""
    return "CONTEXTO DA CONVERSA:\n" + "\n".join(lines)


def _routed(intent: str, *, source: str, confidence: float | None) -> dict[str, Any]:
    """Monta o update de roteamento, tratando ``end`` de forma especial.

    Args:
        intent: Intenção acionável (``credit`` | ``exchange`` | ``interview`` | ``end``).
        source: Fonte do roteamento para observabilidade.
        confidence: Confiança quando aplicável.

    Returns:
        Update de estado com a intenção resolvida.
    """
    if intent == "end":
        update = end_conversation_update()
        update.update(
            {
                "active_agent": "router",
                "intent": "end",
                "route_source": source,  # type: ignore[typeddict-item]
                "route_confidence": confidence,
            }
        )
        return update
    return {
        "active_agent": "router",
        "intent": intent,  # type: ignore[typeddict-item]
        "route_source": source,  # type: ignore[typeddict-item]
        "route_confidence": confidence,
    }


def _llm_error(composer: MessageComposer | None) -> dict[str, Any]:
    """Resposta amigável quando a LLM falha após as tentativas de retry.

    Não silencia a falha nem cai em heurística: informa a instabilidade e
    convida o cliente a repetir, preservando o contexto do fluxo atual.
    """
    return {
        "active_agent": "router",
        "intent": "unknown",
        "route_source": "error",
        "route_confidence": None,
        "messages": speak(
            composer,
            MessageSpec(
                goal="informar de forma cordial uma instabilidade momentânea e pedir que "
                "o cliente reenvie a mensagem",
                fallback=(
                    "Tive uma instabilidade momentânea ao processar sua mensagem. "
                    "Pode enviar novamente, por favor?"
                ),
            ),
        ),
    }


def _clarify(composer: MessageComposer | None, *, intent: Intent = "unknown") -> dict[str, Any]:
    """Clarificação curta e natural (sem menu de bot) para mensagem ambígua."""
    return {
        "active_agent": "router",
        "intent": intent,
        "route_source": "context",
        "route_confidence": None,
        "messages": speak(
            composer,
            MessageSpec(
                goal="pedir uma clarificação breve e natural sobre o que o cliente precisa",
                fallback=(
                    "Só para eu te direcionar melhor: você gostaria de falar sobre "
                    "limite de crédito ou cotação de moedas?"
                ),
                options=("Limite de crédito", "Cotação de moedas (câmbio)"),
            ),
        ),
    }
