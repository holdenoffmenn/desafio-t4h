# syntax=docker/dockerfile:1
# Multi-stage build — imagem slim, usuário não-root alinhado ao host (bind mounts).

FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Extras opcionais (ex.: llm-gemini,observability). Vazio = só dependências base.
ARG EXTRAS=llm-gemini,observability

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && if [ -n "$EXTRAS" ]; then \
         pip wheel --no-cache-dir --wheel-dir /wheels ".[${EXTRAS}]"; \
       else \
         pip wheel --no-cache-dir --wheel-dir /wheels .; \
       fi


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# UID/GID padrão 1000 batem com o usuário típico do host (WSL/Linux),
# evitando EACCES em volumes bind-montados (data/, models/).
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --create-home --uid "${APP_UID}" --gid appuser --shell /usr/sbin/nologin appuser

COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models ./models
COPY scripts ./scripts
COPY docker/entrypoint.sh /entrypoint.sh

RUN pip install --no-cache-dir /wheels/* \
    && chmod +x /entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]
# Default: API. Override no compose para o serviço UI.
CMD ["uvicorn", "banco_agil.main:app", "--host", "0.0.0.0", "--port", "8000"]
