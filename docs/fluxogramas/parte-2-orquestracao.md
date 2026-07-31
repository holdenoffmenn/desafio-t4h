# Parte 2 — Fluxogramas: Orquestração (LangGraph)

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 2).
>
> **Parte 2 implementada e validada** (`test_routing` + `test_graph_flow` verdes).

---

## Topologia geral

### `graph.workflow.build_graph` — ✅

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
    rc -- interview_accepted --> interview
    rc -- senão --> END
    interview --> ri{route_after_interview}
    ri -- complete --> credit
    ri -- incompleta --> END
    exchange --> END
    end --> END
```

> **Nota:** skills encerram o **turno** com `END`. O hub (`router`) é retomado no próximo `invoke` (`guard → router`), evitando loop no mesmo tick.

---

## Funções de roteamento (edges — puras)

### `graph.edges.route_after_guard` — ✅

```mermaid
flowchart TD
    A[state] --> B{input_blocked?}
    B -- Sim --> S[safe_reply]
    B -- Não --> C{authenticated?}
    C -- Não --> T[triage]
    C -- Sim --> R[router]
```

### `graph.edges.route_after_triage` — ✅

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

### `graph.edges.route_after_router` — ✅

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

### `graph.edges.route_after_credit` — ✅

```mermaid
flowchart TD
    A[state] --> E{should_end?}
    E -- Sim --> N[end]
    E -- Não --> J{interview_accepted?}
    J -- Sim --> V[interview]
    J -- Não --> W[END]
```

### `graph.edges.route_after_interview` — ✅

```mermaid
flowchart TD
    A[state] --> C{interview_complete?}
    C -- Sim --> CR[credit<br/>reanálise]
    C -- Não --> W[END]
```

---

## Nós com ramificação

### `graph.nodes.guard` — ✅

```mermaid
flowchart TD
    A[nó guard: última mensagem do usuário] --> EN{SAFETY_ENABLED?}
    EN -- Não --> PASS[input_blocked=False]
    EN -- Sim --> C[SafetyClassifier.check]
    C --> B{blocked?}
    B -- Sim --> SET[input_blocked=True, safety_label, safety_score]
    B -- Não --> PASS
```

### `graph.nodes.triage` — ✅

```mermaid
flowchart TD
    A[nó triage] --> ENDQ{pedido de fim?}
    ENDQ -- Sim --> E[should_end]
    ENDQ -- Não --> CPF{tem CPF?}
    CPF -- Não --> P1[pedir CPF]
    CPF -- Sim --> DT{tem data?}
    DT -- Não --> P2[pedir data nascimento]
    DT -- Sim --> AUTH[authenticate_customer]
    AUTH --> OK{ok?}
    OK -- Sim --> R[authenticated + saudação]
    OK -- Não --> ATT[auth_attempts += 1]
```

### `graph.nodes.router` — ✅

```mermaid
flowchart TD
    A[nó router] --> AFF{oferta entrevista + sim?}
    AFF -- Sim --> IV[intent=interview]
    AFF -- Não --> S[SemanticIntentRouter.predict]
    S --> R{resultado?}
    R -- RouteResult --> SET1[intent semantic]
    R -- None --> H[heuristic_intent]
    H --> L{achou?}
    L -- Sim --> SET2[intent heuristic]
    L -- Não --> FB[llm_fallback / clarificar]
```

---

## Tools

### `tools.credit_tools.request_limit_increase_update` — ✅

```mermaid
flowchart TD
    A[request_limit_increase] --> CR[repo.create -> status pendente]
    CR --> EV[CreditLimitService.evaluate_request]
    EV --> UP[repo.update_status aprovado/rejeitado]
    UP --> AP{aprovado?}
    AP -- Sim --> UL[customer_repo.update_limit]
    AP -- Não --> SK[não altera limite]
    UL --> OUT[update: last_request_id, last_request_status]
    SK --> OUT
```
