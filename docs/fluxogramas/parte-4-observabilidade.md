# Parte 4 — Fluxogramas: Observabilidade, Hardening e Entrega

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 4).

---

## Observabilidade

### `infrastructure.langfuse_tracer` — trace/spans/eventos — 🟡

```mermaid
flowchart TD
    A[invoke do grafo] --> H{Langfuse disponível?}
    H -- Não --> DEG[seguir sem trace<br/>log local structlog]
    H -- Sim --> TR[Trace session_id, cpf_masked]
    TR --> SP[Spans por nó:<br/>node:guard, node:router, agent:*, tool:*]
    SP --> EV[Eventos:<br/>input_blocked, router_fallback_llm,<br/>auth_failed, credit_rejected]
    EV --> URL[get_trace_url -> metadata.langfuse_trace_url]
```

### `observability.logging` — redaction de PII — 🟡

```mermaid
flowchart TD
    A[evento de log] --> P[processador structlog]
    P --> C{campo sensível?}
    C -- cpf --> M[mascarar ***]
    C -- data_nascimento --> D[remover do log]
    C -- demais --> K[manter]
    M --> OUT[emitir log estruturado]
    D --> OUT
    K --> OUT
```

---

## Infraestrutura de entrega

### `docker-compose` — ordem de subida — 🟡

```mermaid
flowchart TD
    A[docker compose up] --> B[build imagem única multi-stage]
    B --> API[serviço api: uvicorn]
    API --> HC{healthcheck /health OK?}
    HC -- Não --> RETRY[retries até start_period]
    HC -- Sim --> UI[serviço ui: streamlit<br/>depends_on service_healthy]
    UI --> READY[UI :8501 + API :8000 prontos]
```
