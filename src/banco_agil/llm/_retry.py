"""Retry com backoff para chamadas de LLM (transitório: rede/quota/timeout).

Centraliza a política de "tentar novamente antes de desistir" para todas as
chamadas ``model.invoke`` da camada LLM. Falhas transitórias (rede, 429,
timeout) são reavaliadas algumas vezes; se todas as tentativas falharem, a
última exceção é propagada para o chamador decidir a degradação (texto
canônico, ``None`` ou mensagem de erro amigável).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = get_logger(__name__)

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5


def invoke_with_retry(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    event: str = "llm_invoke",
) -> Any:
    """Invoca o chat model com retry exponencial em falhas transitórias.

    Args:
        model: Chat model LangChain já instanciado.
        messages: Mensagens (System/Human) a enviar.
        attempts: Nº total de tentativas (>= 1).
        base_delay: Atraso base em segundos; cresce como ``base_delay * 2**i``.
        event: Nome do evento de log para observabilidade.

    Returns:
        A resposta do modelo (``BaseMessage``) na primeira tentativa bem-sucedida.

    Raises:
        Exception: Repropaga a última exceção se todas as tentativas falharem.
    """
    total = max(1, attempts)
    last_exc: Exception | None = None
    for index in range(total):
        try:
            return model.invoke(list(messages))
        # Falha transitória (rede/quota/timeout): tenta novamente antes de desistir.
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            remaining = total - index - 1
            logger.warning(
                event,
                error=str(exc),
                attempt=index + 1,
                attempts=total,
                will_retry=remaining > 0,
            )
            if remaining > 0:
                time.sleep(base_delay * (2**index))

    assert last_exc is not None  # total >= 1 garante ao menos uma tentativa
    raise last_exc
