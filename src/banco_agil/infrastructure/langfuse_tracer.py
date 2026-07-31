"""Tracer Langfuse com degradação graciosa (app funciona sem chaves/SDK)."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from banco_agil.config import Settings
from banco_agil.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TurnTraceResult:
    """Resultado de um turno rastreado.

    Attributes:
        trace_id: Identificador do trace (local ou Langfuse).
        trace_url: URL clicável no Langfuse Cloud, se disponível.
        enabled: Se o SDK Langfuse estava ativo neste turno.
    """

    trace_id: str
    trace_url: str | None = None
    enabled: bool = False


@dataclass
class SessionTracer:
    """Registra traces/spans/eventos de um turno de atendimento.

    Se Langfuse não estiver configurado ou o pacote não estiver instalado,
    opera em modo no-op (apenas logs locais).
    """

    settings: Settings
    _client: Any | None = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Tenta inicializar o client Langfuse."""
        public_key = (self.settings.langfuse_public_key or "").strip()
        secret_key = (self.settings.langfuse_secret_key or "").strip()
        if not public_key or not secret_key:
            logger.info("langfuse_disabled", reason="missing_keys")
            return
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "langfuse_disabled",
                reason="package_missing",
                hint="pip install 'banco-agil[observability]'",
            )
            return
        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=self.settings.langfuse_host,
            )
            self._enabled = True
            logger.info("langfuse_enabled", host=self.settings.langfuse_host)
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_init_failed", error=str(exc))
            self._client = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """True se o client Langfuse está ativo."""
        return self._enabled and self._client is not None

    def record_turn(self, *, session_id: str, state: dict[str, Any]) -> TurnTraceResult:
        """Registra um turno a partir do estado do grafo.

        Emite spans lógicos (guard/router/agent) e eventos de negócio
        (input_blocked, auth_failed, credit_rejected, etc.).

        Args:
            session_id: Identificador da sessão.
            state: Estado pós-``invoke``.

        Returns:
            TurnTraceResult com URL opcional do trace.
        """
        trace_id = str(uuid4())
        active_agent = str(state.get("active_agent") or "unknown")
        cpf_masked = "***" if state.get("cpf") else None

        events: list[str] = []
        if state.get("input_blocked") or active_agent == "safe_reply":
            events.append("input_blocked")
        if state.get("route_source") == "llm_fallback":
            events.append("router_fallback_llm")
        if (
            not state.get("authenticated")
            and int(state.get("auth_attempts") or 0) > 0
            and active_agent in {"triage", "end"}
        ):
            events.append("auth_failed")
        if state.get("last_request_status") == "rejeitado":
            events.append("credit_rejected")

        logger.info(
            "chat_turn",
            session_id=session_id,
            active_agent=active_agent,
            authenticated=bool(state.get("authenticated")),
            intent=state.get("intent"),
            route_source=state.get("route_source"),
            cpf=cpf_masked,
            events=events,
            # data_nascimento nunca deve aparecer — redaction também cobre
        )

        if not self.enabled:
            return TurnTraceResult(trace_id=trace_id, trace_url=None, enabled=False)

        try:
            return self._record_langfuse(
                session_id=session_id,
                state=state,
                events=events,
                cpf_masked=cpf_masked,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_record_failed", error=str(exc))
            return TurnTraceResult(trace_id=trace_id, trace_url=None, enabled=False)

    def _record_langfuse(
        self,
        *,
        session_id: str,
        state: dict[str, Any],
        events: list[str],
        cpf_masked: str | None,
    ) -> TurnTraceResult:
        """Persiste o trace no Langfuse (SDK v3/v4, baseado em OpenTelemetry).

        Usa ``create_trace_id`` + ``start_observation`` (span raiz da sessão e
        span do nó ativo) e anexa os eventos de negócio. ``session_id`` e demais
        atributos vão no nome do trace e na metadata (a v4 não expõe
        ``update_trace`` fora de context manager).
        """
        from langfuse import propagate_attributes

        client = self._client
        assert client is not None

        active_agent = str(state.get("active_agent") or "unknown")
        trace_name = f"session_{session_id}"
        span_name = f"agent:{active_agent}"
        if active_agent in {"guard", "router"}:
            span_name = f"node:{active_agent}"

        # propagate_attributes define atributos de trace (session_id, nome) e os
        # propaga aos spans filhos — habilita o agrupamento nativo por Sessions.
        with (
            propagate_attributes(
                session_id=session_id,
                trace_name=trace_name,
                metadata={
                    "authenticated": bool(state.get("authenticated")),
                    "cpf_masked": cpf_masked,
                    "intent": state.get("intent"),
                },
            ),
            client.start_as_current_observation(
                name=trace_name,
                as_type="span",
                metadata={"active_agent": active_agent},
            ) as root,
        ):
            trace_id: str = root.trace_id
            node_span = client.start_observation(
                trace_context={"trace_id": trace_id, "parent_span_id": root.id},
                name=span_name,
                as_type="span",
                metadata={
                    "route_source": state.get("route_source"),
                    "route_confidence": state.get("route_confidence"),
                    "safety_label": state.get("safety_label"),
                    "safety_score": state.get("safety_score"),
                    "last_request_status": state.get("last_request_status"),
                },
            )
            for event_name in events:
                node_span.create_event(name=event_name)
            node_span.end()

        # Flush best-effort (não bloqueia o request se falhar)
        with suppress(Exception):
            client.flush()

        trace_url = client.get_trace_url(trace_id=trace_id)
        return TurnTraceResult(trace_id=trace_id, trace_url=trace_url, enabled=True)


def build_tracer(settings: Settings) -> SessionTracer:
    """Factory do tracer a partir das Settings.

    Args:
        settings: Configuração tipada.

    Returns:
        SessionTracer (enabled ou no-op).
    """
    return SessionTracer(settings=settings)
