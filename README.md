# Banco Ágil — Agente Bancário Inteligente

Sistema de atendimento bancário com agentes de IA especializados.

> **Status:** Partes 1–2 implementadas. Ver [`PROMPT-IMPLEMENTACAO.md`](PROMPT-IMPLEMENTACAO.md).
> Arquitetura: [`docs/ESTRATEGIA-IMPLEMENTACAO.md`](docs/ESTRATEGIA-IMPLEMENTACAO.md).

## Progresso

| Parte | Conteúdo | Status |
|---|---|---|
| 1 | Domain + CSV + ML classifiers | ✅ |
| 2 | LangGraph (guard, triage, router, skills, CLI) | ✅ |
| 3 | FastAPI + Streamlit (Cliente + Backoffice) | ✅ |
| 4 | Langfuse + Docker + README final | 🔲 |

## Setup rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

python scripts/seed_data.py
python scripts/train_router.py
python scripts/train_safety.py

ruff check . && pyright
pytest tests/unit tests/integration -v
```

## CLI (Parte 2)

```bash
# cliente seed: Ana — CPF 529.982.247-25 / nascimento 15/05/1990
python -m banco_agil.cli
```

## API + UI (Parte 3)

```bash
# terminal 1 — API
uvicorn banco_agil.main:app --reload --port 8000

# terminal 2 — Streamlit
streamlit run src/banco_agil/ui/streamlit_app.py
```

- API docs: http://localhost:8000/docs  
- UI: http://localhost:8501 (abas **Visão Cliente** e **Tech for Humans**)

Fluxo sugerido: autenticação → consulta limite → `sim` → valor → (rejeição/aprovação) → câmbio → encerrar.

### Estrutura

```
src/banco_agil/
  config.py / deps.py / cli.py
  domain/           # regras de negócio puras
  infrastructure/   # CSV atômico, FX, checkpointer
  ml/               # intent router + safety
  graph/            # state, edges, nodes, workflow
  agents/           # persona + prompts
  tools/            # tools tipadas
  utils/
data/  models/  scripts/  tests/  docs/
```
