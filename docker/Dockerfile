# syntax=docker/dockerfile:1
# Multi-stage build — imagem slim, usuário não-root.

FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels .
# Langfuse é opcional: pip install langfuse no runtime ou extra [observability]


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models ./models
COPY scripts ./scripts

RUN pip install --no-cache-dir /wheels/* \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Default: API. Override no compose para o serviço UI.
CMD ["uvicorn", "banco_agil.main:app", "--host", "0.0.0.0", "--port", "8000"]
