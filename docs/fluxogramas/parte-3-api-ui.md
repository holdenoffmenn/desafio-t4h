# Parte 3 — Fluxogramas: API + Interface

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 3).
>
> **Parte 3 implementada e validada** (`test_api` verde).

---

## API

### `api.routes.chat.chat` (`POST /chat`) — ✅

```mermaid
flowchart TD
    A[POST /chat: session_id, message] --> V[validar ChatRequest]
    V --> G{grafo pronto?}
    G -- Não --> E503[HTTP 503]
    G -- Sim --> ST[get_state thread_id]
    ST --> NEW{sessão existe?}
    NEW -- Não --> INIT[initial_state + HumanMessage]
    NEW -- Sim --> MSG[apenas HumanMessage]
    INIT --> INV[graph.invoke]
    MSG --> INV
    INV --> ERR{DomainError?}
    ERR -- Sim --> T[mensagem amigável + metadata parcial]
    ERR -- Não --> M[build_metadata + last_ai_text]
    M --> R[ChatResponse 200]
    T --> R
```

### Interação end-to-end (camadas) — ✅

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

### `ui.streamlit_app` — ciclo de interação — ✅

```mermaid
flowchart TD
    A[carregar app] --> SID{session_id existe?}
    SID -- Não --> GEN[gerar uuid4 em st.session_state]
    SID -- Sim --> USE[reusar session_id]
    GEN --> IN[input / Encerrar / Nova sessão]
    USE --> IN
    IN --> CALL[httpx POST /chat]
    CALL --> HERR{erro HTTP?}
    HERR -- Sim --> MSG[exibir aviso amigável]
    HERR -- Não --> R1[Tab Cliente: append reply]
    R1 --> R2[Tab Backoffice: render metadata]
```
