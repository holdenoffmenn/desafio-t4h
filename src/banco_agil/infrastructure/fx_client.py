"""Cliente HTTP para cotação de câmbio (AwesomeAPI) com cache e mock."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from banco_agil.domain.errors import FxUnavailableError
from banco_agil.domain.models import ExchangeRate

# Pares comuns suportados pela AwesomeAPI (ampliável).
SUPPORTED_PAIRS: frozenset[str] = frozenset(
    {
        "USD-BRL",
        "EUR-BRL",
        "GBP-BRL",
        "JPY-BRL",
        "ARS-BRL",
        "CAD-BRL",
        "AUD-BRL",
        "CHF-BRL",
        "CNY-BRL",
        "BTC-BRL",
    }
)


@dataclass
class _CacheEntry:
    """Entrada de cache em memória com TTL."""

    rate: ExchangeRate
    expires_at: float


class FxClient:
    """Consulta cotação de moedas com timeout, retry, cache TTL e modo mock.

    Colaboradores:
        httpx: cliente HTTP.
    """

    def __init__(
        self,
        api_url_template: str,
        *,
        mock: bool = False,
        timeout: float = 5.0,
        max_retries: int = 2,
        cache_ttl_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Inicializa o cliente de câmbio.

        Args:
            api_url_template: URL com placeholder ``{pair}`` (ex.: USD-BRL).
            mock: Se True, retorna cotação fixa sem rede.
            timeout: Timeout por request em segundos.
            max_retries: Tentativas adicionais após falha.
            cache_ttl_seconds: TTL do cache em memória.
            client: Cliente httpx injetável (testes).
        """
        self._url_template = api_url_template
        self._mock = mock
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache_ttl = cache_ttl_seconds
        self._client = client
        self._cache: dict[str, _CacheEntry] = {}

    def get_rate(self, currency: str, base: str = "BRL") -> ExchangeRate:
        """Obtém cotação da moeda contra a base (default BRL).

        Args:
            currency: Código ISO da moeda (ex.: ``USD``, ``eur``).
            base: Moeda base (default ``BRL``).

        Returns:
            ExchangeRate com bid/ask/timestamp.

        Raises:
            FxUnavailableError: Se a API falhar após retries ou par inválido.
            ValueError: Se o código de moeda for inválido.
        """
        code = currency.strip().upper()
        if len(code) < 3 or not code.isalpha():
            raise ValueError(f"Código de moeda inválido: {currency!r}")

        pair = f"{code}-{base.upper()}"
        if pair not in SUPPORTED_PAIRS:
            raise FxUnavailableError(
                f"Par {pair} não suportado. Disponíveis: {sorted(SUPPORTED_PAIRS)}"
            )

        if self._mock:
            return ExchangeRate(
                currency=code,
                bid=5.1234,
                ask=5.1250,
                timestamp="mock",
            )

        cached = self._cache.get(pair)
        now = time.monotonic()
        if cached is not None and cached.expires_at > now:
            return cached.rate

        rate = self._fetch_with_retry(pair, code)
        self._cache[pair] = _CacheEntry(rate=rate, expires_at=now + self._cache_ttl)
        return rate

    def _fetch_with_retry(self, pair: str, code: str) -> ExchangeRate:
        """Busca a cotação com retries e timeout."""
        url = self._url_template.format(pair=pair)
        last_error: Exception | None = None
        attempts = self._max_retries + 1

        for _ in range(attempts):
            try:
                return self._fetch_once(url, code)
            except (httpx.HTTPError, FxUnavailableError, KeyError, ValueError) as exc:
                last_error = exc

        raise FxUnavailableError(
            f"API de câmbio indisponível para {pair}: {last_error}"
        ) from last_error

    def _fetch_once(self, url: str, code: str) -> ExchangeRate:
        """Executa um GET e faz parse da resposta AwesomeAPI."""
        if self._client is not None:
            response = self._client.get(url, timeout=self._timeout)
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url)

        response.raise_for_status()
        payload: Any = response.json()

        # AwesomeAPI retorna dict keyed por "USDBRL" ou lista em alguns endpoints.
        if isinstance(payload, list):
            if not payload:
                raise FxUnavailableError("Resposta de câmbio vazia.")
            item = payload[0]
        elif isinstance(payload, dict):
            # chave típica: USDBRL
            if not payload:
                raise FxUnavailableError("Resposta de câmbio vazia.")
            item = next(iter(payload.values()))
        else:
            raise FxUnavailableError("Formato de resposta de câmbio inesperado.")

        return ExchangeRate(
            currency=code,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp=str(item.get("create_date", item.get("timestamp", ""))),
        )
