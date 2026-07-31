# Parte 3 — Fluxogramas: API + Interface

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 3).

---

## API

### `api.routes.chat.post_chat` (`POST /chat`) — 🟡

```mermaid
flowchart TD
    A[POST /chat: session_id, message] --> V[validar ChatRequest]
    V --> INV[graph.invoke<br/>config thread_id=session_id]
    INV --> ERR{erro de domínio?}
    ERR -- Sim --> T[traduzir p/ mensagem amigável<br/>sem stack trace]
    ERR -- Não --> ST[ler estado resultante do checkpointer]
    ST --> M[montar ChatMetadata:<br/>active_agent, intent, route, safety,<br/>tool_calls mascarados, score, trace_url]
    M --> R[return ChatResponse]
    T --> R
```

### Interação end-to-end (camadas) — 🟡

```mermaid
sequenceDiagram
    participant U as Usuário
    participant S as Streamlit
    participant A as FastAPI
    participant G as LangGraph (+checkpointer)
    U->>S: mensagem
    S->>A: POST /chat {session_id, message}
    A->>G: invoke(thread_id=session_id)
    G-->>A: estado atualizado
    A-->>S: {reply, metadata}
    S-->>U: render Tab Cliente
    S-->>U: render Tab Backoffice (route/safety/tools)
```

---

## UI

### `ui.streamlit_app` — ciclo de interação — 🟡

```mermaid
flowchart TD
    A[carregar app] --> SID{session_id existe?}
    SID -- Não --> GEN[gerar uuid4 em st.session_state]
    SID -- Sim --> USE[reusar session_id]
    GEN --> IN[input do usuário / botão Encerrar]
    USE --> IN
    IN --> CALL[httpx POST /chat]
    CALL --> HERR{erro HTTP?}
    HERR -- Sim --> MSG[exibir aviso amigável]
    HERR -- Não --> R1[Tab Cliente: append reply]
    R1 --> R2[Tab Backoffice: render metadata]
```
