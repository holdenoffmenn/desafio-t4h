# Banco Ágil — Agente Bancário Inteligente

Sistema de atendimento bancário com agentes de IA especializados.

> **Status:** Parte 1 (Fundação) implementada. Ver [`PROMPT-IMPLEMENTACAO.md`](PROMPT-IMPLEMENTACAO.md) para o plano completo.
> Arquitetura detalhada em [`docs/ESTRATEGIA-IMPLEMENTACAO.md`](docs/ESTRATEGIA-IMPLEMENTACAO.md).

## Parte 1 — Fundação (já disponível)

Camada de domínio, repositórios CSV atômicos, classifiers ML (roteamento + safety) e utilitários — **sem LLM**.

### Setup rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

python scripts/seed_data.py
python scripts/train_router.py
python scripts/train_safety.py

ruff check . && ruff format --check .
pyright
pytest tests/unit tests/integration -v --cov=src/banco_agil --cov-report=term-missing
```

### Estrutura (Parte 1)

```
src/banco_agil/
  config.py
  domain/           # regras de negócio puras
  infrastructure/   # CSV atômico, FX client
  ml/               # intent router + safety classifier
  utils/            # CPF, moeda, datas
data/               # seeds + datasets ML
models/             # artefatos .joblib (gerados)
scripts/            # seed_data, train_router, train_safety
tests/
```
