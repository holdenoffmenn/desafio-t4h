"""UI Streamlit — Visão Cliente + Tech for Humans (Backoffice)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st

from banco_agil.config import get_settings

# Velocidade do efeito de digitação (segundos por palavra) no streaming do cliente.
_STREAM_DELAY_S = 0.02


def _api_base() -> str:
    """URL base da API (Settings / env)."""
    return get_settings().api_base_url.rstrip("/")


def _ensure_session() -> None:
    """Inicializa chaves do st.session_state na primeira carga."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = None
    if "ended" not in st.session_state:
        st.session_state.ended = False
    if "pending" not in st.session_state:
        st.session_state.pending = None


def _post_chat(session_id: str, message: str) -> dict[str, Any]:
    """Chama POST /chat.

    Args:
        session_id: ID da sessão.
        message: Texto do usuário.

    Returns:
        JSON da resposta.

    Raises:
        httpx.HTTPError: Em falha de rede/HTTP.
    """
    url = f"{_api_base()}/chat"
    response = httpx.post(
        url,
        json={"session_id": session_id, "message": message},
        timeout=60.0,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def _new_session() -> None:
    """Reinicia a sessão de chat no front."""
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.last_metadata = None
    st.session_state.ended = False
    st.session_state.pending = None


def _stream_reply(text: str) -> Iterator[str]:
    """Gera a resposta em pedaços para um efeito de digitação suave.

    As respostas são determinísticas (geradas no backend), então o streaming
    é apenas visual: emite palavra a palavra com um pequeno atraso.

    Args:
        text: Resposta completa da API.

    Yields:
        Fragmentos de texto (palavra + espaço).
    """
    parts = text.split(" ")
    for index, word in enumerate(parts):
        yield word + (" " if index < len(parts) - 1 else "")
        time.sleep(_STREAM_DELAY_S)


def _render_client_tab() -> None:
    """Aba Visão Cliente — chat limpo."""
    st.subheader("Atendimento Banco Ágil")
    st.caption(f"Sessão: `{st.session_state.session_id}`")

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    # Há uma mensagem do usuário já exibida aguardando resposta: processa agora
    # com feedback visual (spinner) e streaming.
    if st.session_state.pending is not None:
        _process_pending()

    busy = st.session_state.pending is not None
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("Nova sessão", width="stretch", disabled=busy):
            _new_session()
            st.rerun()
    with cols[1]:
        end_clicked = st.button(
            "Encerrar atendimento",
            width="stretch",
            disabled=st.session_state.ended or busy,
        )

    if end_clicked and not st.session_state.ended:
        _queue_message("quero encerrar o atendimento")

    prompt = st.chat_input(
        "Digite sua mensagem…",
        disabled=st.session_state.ended,
    )
    if prompt:
        _queue_message(prompt)


def _queue_message(text: str) -> None:
    """Adiciona a mensagem do usuário e re-renderiza imediatamente.

    O ``rerun`` faz a mensagem aparecer na hora; o processamento (chamada à
    API) acontece no próximo ciclo, em ``_process_pending``.

    Args:
        text: Texto do usuário.
    """
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.pending = text
    st.rerun()


def _process_pending() -> None:
    """Chama a API para a mensagem pendente, com spinner e streaming."""
    text = str(st.session_state.pending)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Processando…"):
                data = _post_chat(st.session_state.session_id, text)
            reply = str(data.get("reply", ""))
            meta = data.get("metadata")
            st.write_stream(_stream_reply(reply))
            st.session_state.last_metadata = meta
            if isinstance(meta, dict) and meta.get("should_end"):
                st.session_state.ended = True
        except httpx.HTTPError as exc:
            reply = (
                "Não foi possível falar com o servidor agora. "
                f"Verifique se a API está no ar ({_api_base()}). "
                f"Detalhe: {exc.__class__.__name__}"
            )
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.pending = None
    st.rerun()


def _render_backoffice_tab() -> None:
    """Aba Tech for Humans — estado, roteamento, tools, score."""
    st.subheader("Tech for Humans — Backoffice")
    st.caption("Visão técnica da sessão (não visível ao cliente).")

    meta = st.session_state.last_metadata
    if not meta:
        st.info("Envie uma mensagem na aba Cliente para popular o painel.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Agente ativo", str(meta.get("active_agent") or "—"))
    with c2:
        st.metric("Autenticado", "sim" if meta.get("authenticated") else "não")
    with c3:
        st.metric("Tentativas auth", str(meta.get("auth_attempts", 0)))

    st.markdown("#### Roteamento")
    route = meta.get("route") or {}
    r1, r2 = st.columns(2)
    with r1:
        st.write(f"**Intent:** `{meta.get('intent')}`")
        st.write(f"**Source:** `{route.get('source')}`")
    with r2:
        conf = route.get("confidence")
        if conf is not None:
            st.progress(min(max(float(conf), 0.0), 1.0), text=f"confidence={conf:.2f}")
        else:
            st.write("**Confidence:** —")

    st.markdown("#### Segurança")
    safety = meta.get("safety") or {}
    blocked = bool(safety.get("blocked"))
    badge = "BLOQUEADO" if blocked else "ok"
    st.write(
        f"**Status:** `{badge}` · label=`{safety.get('label')}` · score=`{safety.get('score')}`"
    )

    st.markdown("#### Tools executadas")
    tools = meta.get("last_tool_calls") or []
    if tools:
        st.dataframe(tools, width="stretch")
    else:
        st.write("_Nenhuma tool neste turno._")

    st.markdown("#### Score (última entrevista)")
    score = meta.get("last_score_calculation")
    if score:
        st.json(score)
    else:
        st.write("_Sem cálculo de score nesta sessão ainda._")

    st.markdown("#### Pedido de crédito")
    st.write(f"**Último status:** `{meta.get('last_request_status')}`")

    st.markdown("#### Langfuse")
    trace = meta.get("langfuse_trace_url")
    if trace:
        st.link_button("Abrir trace", trace)
    else:
        st.caption("Trace Langfuse disponível na Parte 4.")

    st.markdown("#### Preview — solicitações_aumento_limite.csv")
    csv_path = Path(get_settings().data_dir) / "solicitacoes_aumento_limite.csv"
    if csv_path.exists():
        try:
            st.code(csv_path.read_text(encoding="utf-8")[:4000], language="csv")
        except OSError:
            st.warning("Não foi possível ler o CSV de solicitações.")
    else:
        st.write("_Arquivo ainda não gerado._")

    with st.expander("Metadata JSON completo"):
        st.code(json.dumps(meta, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    """Entry point Streamlit."""
    st.set_page_config(
        page_title="Banco Ágil",
        page_icon="🏦",
        layout="wide",
    )
    _ensure_session()
    st.title("Banco Ágil")
    st.caption(f"API: `{_api_base()}`")

    tab_client, tab_back = st.tabs(["Visão Cliente", "Tech for Humans (Backoffice)"])
    with tab_client:
        _render_client_tab()
    with tab_back:
        _render_backoffice_tab()


main()
