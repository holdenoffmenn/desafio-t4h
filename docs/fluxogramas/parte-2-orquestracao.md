# Parte 2 — Fluxogramas: Orquestração (LangGraph)

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 2).

---

## Topologia geral

### `graph.workflow.build_graph` — 🟡

```mermaid
flowchart TD
    START((START)) --> guard
    guard --> rg{route_after_guard}
    rg -- input_blocked --> safe_reply --> END((END))
    rg -- não autenticado --> triage
    rg -- autenticado --> router
    triage --> rt{route_after_triage}
    rt -- 3 falhas --> end
    rt -- turno aguarda input --> END
    rt -- autenticado --> router
    router --> rr{route_after_router}
    rr -- credit --> credit
    rr -- exchange --> exchange
    rr -- interview --> interview
    rr -- end --> end
    rr -- incerto --> END
    credit --> rc{route_after_credit}
    rc -- rejeitado + aceite --> interview
    rc -- senão --> router
    interview --> router
    exchange --> router
    end --> END
```

---

## Funções de roteamento (edges — puras)

### `graph.edges.route_after_guard` — 🟡

```mermaid
flowchart TD
    A[state] --> B{input_blocked?}
    B -- Sim --> S[safe_reply]
    B -- Não --> C{authenticated?}
    C -- Não --> T[triage]
    C -- Sim --> R[router]
```

### `graph.edges.route_after_triage` — 🟡

```mermaid
flowchart TD
    A[state] --> E{should_end?}
    E -- Sim --> N1[end]
    E -- Não --> AU{authenticated?}
    AU -- Sim --> R[router]
    AU -- Não --> AT{auth_attempts >= 3?}
    AT -- Sim --> N2[end]
    AT -- Não --> W[END<br/>encerra turno, aguarda input]
```

### `graph.edges.route_after_router` — 🟡

```mermaid
flowchart TD
    A[state] --> E{should_end?}
    E -- Sim --> N[end]
    E -- Não --> I{intent}
    I -- credit --> C[credit]
    I -- exchange --> X[exchange]
    I -- interview --> V[interview]
    I -- end --> N
    I -- unknown --> W[END<br/>clarificar no próximo turno]
```

### `graph.edges.route_after_credit` — 🟡

```mermaid
flowchart TD
    A[state] --> E{should_end?}
    E -- Sim --> N[end]
    E -- Não --> J{last_request_status == rejeitado<br/>E offered_interview?}
    J -- Sim --> V[interview]
    J -- Não --> R[router]
```

### `graph.edges.route_after_interview` — 🟡

```mermaid
flowchart TD
    A[state] --> R[router<br/>reanálise com score atualizado]
```

---

## Nós com ramificação

### `graph.nodes.guard` — 🟡

```mermaid
flowchart TD
    A[nó guard: última mensagem do usuário] --> EN{SAFETY_ENABLED?}
    EN -- Não --> PASS[input_blocked=False]
    EN -- Sim --> C[SafetyClassifier.check]
    C --> B{blocked?}
    B -- Sim --> SET[input_blocked=True, safety_label, safety_score]
    B -- Não --> PASS
```

### `graph.nodes.triage` — 🔲

```mermaid
flowchart TD
    A[nó triage] --> TODO[a detalhar na implementação:<br/>coleta CPF/data multi-turno,<br/>authenticate_customer,<br/>incremento de auth_attempts só em par completo]
```

### `graph.nodes.router` — 🟡

```mermaid
flowchart TD
    A[nó router] --> S[SemanticIntentRouter.predict]
    S --> R{resultado?}
    R -- RouteResult --> SET1[intent, route_confidence,<br/>route_source=semantic]
    R -- None --> L[identify_intent via LLM]
    L --> SET2[intent, route_source=llm_fallback]
```

---

## Tools

### `tools.credit_tools.request_limit_increase` — 🟡

```mermaid
flowchart TD
    A[request_limit_increase cpf, new_limit] --> CR[repo.create -> status pendente, request_id]
    CR --> EV[CreditLimitService.evaluate_request]
    EV --> UP[repo.update_status aprovado/rejeitado]
    UP --> AP{aprovado?}
    AP -- Sim --> UL[customer_repo.update_limit]
    AP -- Não --> SK[não altera limite]
    UL --> CMD[return Command update: last_request_id, last_request_status]
    SK --> CMD
```
