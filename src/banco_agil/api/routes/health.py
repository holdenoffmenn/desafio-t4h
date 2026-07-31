"""Health check da API."""

from __future__ import annotations

from fastapi import APIRouter

from banco_agil.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Retorna status da API para probes e Docker healthcheck.

    Returns:
        HealthResponse com status ok.
    """
    return HealthResponse()
