# Parte 4 — Fluxogramas: Observabilidade, Hardening e Entrega

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 4).
>
> **Parte 4 implementada e validada** (redaction + tracer no-op + suíte verde).

---

## Observabilidade

### `infrastructure.langfuse_tracer.SessionTracer.record_turn` — ✅

```mermaid
flowchart TD
    A[invoke do grafo / chat] --> H{Langfuse keys + SDK?}
    H -- Não --> DEG[no-op + structlog local]
    H -- Sim --> TR[Trace session_id, cpf_masked]
    TR --> SP[Span node/agent ativo]
    SP --> EV[Eventos: input_blocked, auth_failed,<br/>credit_rejected, router_fallback_llm]
    EV --> URL[trace_url -> metadata.langfuse_trace_url]
    DEG --> LOG[chat_turn log com PII redigida]
    URL --> LOG
```

### `observability.logging.redact_pii` — ✅

```mermaid
flowchart TD
    A[evento de log] --> P[processador structlog]
    P --> C{campo sensível?}
    C -- cpf --> M[mascarar ***]
    C -- data_nascimento --> D[remover / REDACTED]
    C -- string com CPF --> S[regex substitute ***]
    C -- demais --> K[manter]
    M --> OUT[emitir log estruturado]
    D --> OUT
    S --> OUT
    K --> OUT
```

---

## Infraestrutura de entrega

### `docker-compose` — ordem de subida — ✅

```mermaid
flowchart TD
    A[docker compose up] --> B[build imagem multi-stage]
    B --> API[serviço api: uvicorn]
    API --> HC{healthcheck /health OK?}
    HC -- Não --> RETRY[retries até start_period]
    HC -- Sim --> UI[serviço ui: streamlit<br/>depends_on service_healthy]
    UI --> READY[UI :8501 + API :8000 prontos]
```
