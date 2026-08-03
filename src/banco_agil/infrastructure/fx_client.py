"""Cliente HTTP para cotação de câmbio (AwesomeAPI) com cache e mock."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from banco_agil.domain.errors import FxPairNotFoundError, FxUnavailableError
from banco_agil.domain.models import ExchangeRate


@dataclass
class _CacheEntry:
    """Entrada de cache em memória com TTL."""

    rate: ExchangeRate
    expires_at: float


class FxClient:
    """Consulta cotação de moedas com timeout, retry, cache TTL e modo mock.

    Não há whitelist local de pares: qualquer código ISO válido é pedido à API.
    Quando a fonte responde que a moeda não existe, sobe ``FxPairNotFoundError``.

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
        backoff_base_seconds: float = 0.2,
        client: httpx.Client | None = None,
    ) -> None:
        """Inicializa o cliente de câmbio.

        Args:
            api_url_template: URL com placeholder ``{pair}`` (ex.: USD-BRL).
            mock: Se True, retorna cotação fixa sem rede.
            timeout: Timeout por request em segundos.
            max_retries: Tentativas adicionais após falha.
            cache_ttl_seconds: TTL do cache em memória.
            backoff_base_seconds: Base do backoff exponencial entre tentativas.
            client: Cliente httpx injetável (testes).
        """
        self._url_template = api_url_template
        self._mock = mock
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache_ttl = cache_ttl_seconds
        self._backoff_base = backoff_base_seconds
        self._client = client
        self._cache: dict[str, _CacheEntry] = {}

    def get_rate(self, currency: str, base: str = "BRL") -> ExchangeRate:
        """Obtém cotação da moeda contra a base (default BRL).

        Args:
            currency: Código ISO da moeda (ex.: ``USD``, ``eur``, ``KRW``).
            base: Moeda base (default ``BRL``).

        Returns:
            ExchangeRate com bid/ask/timestamp.

        Raises:
            FxPairNotFoundError: Se a API indicar que o par não existe.
            FxUnavailableError: Se a API falhar após retries.
            ValueError: Se o código de moeda for inválido.
        """
        code = currency.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError(f"Código de moeda inválido: {currency!r}")

        pair = f"{code}-{base.upper()}"

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
        """Busca a cotação com retries e timeout.

        ``FxPairNotFoundError`` não é retentado: a moeda simplesmente não existe
        na fonte, independente de novas tentativas.
        """
        url = self._url_template.format(pair=pair)
        last_error: Exception | None = None
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            try:
                return self._fetch_once(url, code, pair)
            except FxPairNotFoundError:
                raise
            except (httpx.HTTPError, FxUnavailableError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(self._backoff_base * (2**attempt))

        raise FxUnavailableError(
            f"API de câmbio indisponível para {pair}: {last_error}"
        ) from last_error

    def _fetch_once(self, url: str, code: str, pair: str) -> ExchangeRate:
        """Executa um GET e faz parse da resposta AwesomeAPI."""
        if self._client is not None:
            response = self._client.get(url, timeout=self._timeout)
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url)

        if response.status_code == 404 or self._is_coin_not_exists(response):
            raise FxPairNotFoundError(
                f"Cotação indisponível para o par {pair}: moeda não encontrada na fonte."
            )

        response.raise_for_status()
        payload: Any = response.json()

        # AwesomeAPI retorna dict keyed por "USDBRL" ou lista em alguns endpoints.
        if isinstance(payload, list):
            if not payload:
                raise FxUnavailableError("Resposta de câmbio vazia.")
            item = payload[0]
        elif isinstance(payload, dict):
            if self._payload_is_coin_not_exists(payload):
                raise FxPairNotFoundError(
                    f"Cotação indisponível para o par {pair}: moeda não encontrada na fonte."
                )
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

    @staticmethod
    def _is_coin_not_exists(response: httpx.Response) -> bool:
        """Detecta corpo AwesomeAPI ``CoinNotExists`` sem depender só do status."""
        try:
            payload: Any = response.json()
        except ValueError:
            return False
        return isinstance(payload, dict) and FxClient._payload_is_coin_not_exists(payload)

    @staticmethod
    def _payload_is_coin_not_exists(payload: dict[str, Any]) -> bool:
        """True quando o JSON indica moeda inexistente."""
        code = str(payload.get("code", ""))
        message = str(payload.get("message", "")).lower()
        return code == "CoinNotExists" or "moeda nao encontrada" in message
