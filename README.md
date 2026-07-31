# Banco Ágil — Agente Bancário Inteligente

Sistema de atendimento bancário com agentes de IA especializados.

> **Status:** Partes 1–2 implementadas. Ver [`PROMPT-IMPLEMENTACAO.md`](PROMPT-IMPLEMENTACAO.md).
> Arquitetura: [`docs/ESTRATEGIA-IMPLEMENTACAO.md`](docs/ESTRATEGIA-IMPLEMENTACAO.md).

## Progresso

| Parte | Conteúdo | Status |
|---|---|---|
| 1 | Domain + CSV + ML classifiers | ✅ |
| 2 | LangGraph (guard, triage, router, skills, CLI) | ✅ |
| 3 | FastAPI + Streamlit | 🔲 |
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

Fluxo sugerido: autenticação → consulta limite → aumento (ex.: 6000) → rejeição → entrevista → câmbio → encerrar.

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
