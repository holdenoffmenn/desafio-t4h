#!/bin/sh
# Entrypoint: valida artefatos necessários e delega ao CMD.
set -eu

DATA_DIR="${DATA_DIR:-/app/data}"
MODELS_DIR="${MODELS_DIR:-/app/models}"

if [ ! -f "${MODELS_DIR}/intent_router.joblib" ] || [ ! -f "${MODELS_DIR}/safety_clf.joblib" ]; then
  echo "aviso: modelos .joblib ausentes em ${MODELS_DIR}." >&2
  echo "       No host: python scripts/train_router.py && python scripts/train_safety.py" >&2
fi

if [ ! -f "${DATA_DIR}/clientes.csv" ] || [ ! -f "${DATA_DIR}/score_limite.csv" ]; then
  echo "aviso: CSVs de seed ausentes em ${DATA_DIR}." >&2
  echo "       No host: python scripts/seed_data.py" >&2
fi

# Falha cedo se o volume montado não for gravável (UID do container ≠ dono do host).
if [ ! -w "${DATA_DIR}" ]; then
  echo "erro: ${DATA_DIR} não é gravável pelo usuário $(id -u):$(id -g)." >&2
  echo "      Ajuste APP_UID/APP_GID no build ou: chmod ug+rw ${DATA_DIR}" >&2
  exit 1
fi

exec "$@"
