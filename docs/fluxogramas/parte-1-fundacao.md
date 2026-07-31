# Parte 1 — Fluxogramas: Fundação (Domain + Infra + ML)

> Status: 🟡 = rascunho da estratégia · ✅ = validado no código · 🔲 = pendente.
> Referência de escopo: [`../../PROMPT-IMPLEMENTACAO.md`](../../PROMPT-IMPLEMENTACAO.md) (Parte 1).
>
> **Parte 1 implementada e validada** (`ruff` / `pyright` / `pytest` verdes).

---

## Domínio

### `domain.auth.AuthService.authenticate` — ✅

```mermaid
flowchart TD
    A[authenticate cpf, birth_date] --> N[normalize_cpf]
    N --> F[repo.find_by_cpf]
    F --> E{cliente existe?}
    E -- Não --> R0[return None]
    E -- Sim --> D{data_nascimento confere?}
    D -- Não --> R1[return None]
    D -- Sim --> OK[return Customer]
```

### `domain.scoring.ScoringService.calculate_score` — ✅

```mermaid
flowchart TD
    A[calculate_score data] --> P[peso_dep = PESO_DEPENDENTES.get<br/>n, PESO_DEPENDENTES_3_MAIS]
    P --> C["raw = renda/(despesas+1)*PESO_RENDA<br/>+ peso_emprego + peso_dep + peso_dividas"]
    C --> CL["clamp: max(0, min(1000, int(raw)))"]
    CL --> OUT[return score 0..1000]
```

### `domain.credit_limit.CreditLimitService.evaluate_request` — ✅

```mermaid
flowchart TD
    A[Início] --> B{new_limit > limite_atual?}
    B -- Não --> R1[rejeitado<br/>reason=limite_menor_que_atual]
    B -- Sim --> C[score_repo.get_max_limit_for_score]
    C --> D{new_limit <= max_allowed?}
    D -- Não --> R2[rejeitado<br/>reason=score_insuficiente]
    D -- Sim --> A1[aprovado<br/>approved_limit=new_limit]
```

---

## Infraestrutura

### `infrastructure.csv_repository.AtomicCsvStore.write_all` — ✅

```mermaid
flowchart TD
    A[write rows] --> L[adquirir FileLock]
    L --> T[abrir arquivo .tmp no MESMO diretório]
    T --> W[escrever todas as linhas no .tmp]
    W --> R[os.replace .tmp -> destino<br/>rename atômico]
    R --> U[liberar FileLock]
    W -. exceção .-> C[remover .tmp / propagar PersistenceError]
    C --> U
```

### `infrastructure.customer_repository.CsvCustomerRepository.update_score` / `update_limit` — ✅

```mermaid
flowchart TD
    A[update_score cpf, valor] --> RD[ler todos os clientes]
    RD --> F{cpf encontrado?}
    F -- Não --> E[raise CustomerNotFoundError]
    F -- Sim --> M[modificar campo em memória]
    M --> WR[AtomicCsvStore.write<br/>temp + replace]
    WR --> OK[retorno]
```

### `infrastructure.credit_request_repository` — ciclo de vida — ✅

```mermaid
flowchart TD
    A[create request] --> G[gerar request_id UUID]
    G --> P[gravar linha status=pendente]
    P --> RID[return request_id]
    RID --> EV[CreditLimitService.evaluate_request]
    EV --> DEC{decisão}
    DEC -- aprovado --> UA[update_status aprovado]
    DEC -- rejeitado --> UR[update_status rejeitado]
```

### `infrastructure.fx_client.FxClient.get_rate` — ✅

```mermaid
flowchart TD
    A[get_rate currency] --> MOCK{FX_MOCK=true?}
    MOCK -- Sim --> RM[retornar cotação mock]
    MOCK -- Não --> CACHE{cache válido < 60s?}
    CACHE -- Sim --> RC[retornar cache]
    CACHE -- Não --> H[httpx GET fx_api_url + timeout]
    H --> OKQ{resposta OK?}
    OKQ -- Sim --> S[parse + atualizar cache -> ExchangeRate]
    OKQ -- Não --> RT{retries restantes?}
    RT -- Sim --> H
    RT -- Não --> ERR[raise FxUnavailableError]
```

---

## Machine Learning

### `ml.intent_router.SemanticIntentRouter.predict` — ✅

```mermaid
flowchart TD
    A[predict text] --> ART{artefato carregado?}
    ART -- Não --> N0[return None<br/>-> fallback LLM]
    ART -- Sim --> K{kind?}
    K -- sklearn_pipeline --> P1[pipeline.predict_proba text]
    K -- embedding_clf --> E[encoder.encode + clf.predict_proba]
    P1 --> M[idx = argmax; conf = probs idx]
    E --> M
    M --> T{conf >= threshold?}
    T -- Não --> N1[return None<br/>-> fallback LLM]
    T -- Sim --> R[return RouteResult intent, conf, source=semantic]
```

### `ml.safety_classifier.SafetyClassifier.check` — ✅

```mermaid
flowchart TD
    A[check text] --> RGX[avaliar denylist regex]
    A --> MDL{artefato carregado?}
    MDL -- Não --> M0[model_label=ok, score=0]
    MDL -- Sim --> MP[model_predict -> label, score]
    RGX --> J[regex_hit?]
    J --> DEC{regex_hit OU<br/>label!=ok e score>=threshold?}
    M0 --> DEC
    MP --> DEC
    DEC -- Sim --> B[SafetyResult blocked=True]
    DEC -- Não --> OK[SafetyResult blocked=False, label=ok]
```
