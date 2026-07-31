"""Factory provider-agnostic para instanciar um chat model LangChain.

Centraliza a escolha do provedor de LLM num único ponto, de modo que o
restante da aplicação dependa apenas da interface ``BaseChatModel`` e nunca
de um provedor específico. Trocar de Gemini para Groq/OpenAI/OpenRouter é
uma mudança de configuração (`.env`), não de código.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from banco_agil.config import Settings, get_settings
from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

# Aliases amigáveis → identificador aceito pelo ``init_chat_model``.
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google_genai",
    "google": "google_genai",
    "google_genai": "google_genai",
    "groq": "groq",
    "openai": "openai",
    "together": "together",
    "togetherai": "together",
    # OpenRouter expõe um endpoint OpenAI-compatível.
    "openrouter": "openai",
}

# Valores que desligam o LLM (modo determinístico/heurístico).
_DISABLED: frozenset[str] = frozenset({"", "none", "fake", "off", "disabled"})


class LLMProviderError(RuntimeError):
    """Provedor desconhecido ou mal configurado."""


@dataclass(frozen=True)
class _ProviderConfig:
    """Parâmetros resolvidos para ``init_chat_model``."""

    model_provider: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve(settings: Settings) -> _ProviderConfig | None:
    """Resolve o provedor e as credenciais a partir das Settings.

    Args:
        settings: Configuração da aplicação.

    Returns:
        ``_ProviderConfig`` pronto para o factory, ou ``None`` se o LLM
        estiver desligado (provider vazio/``none``) ou sem chave de API.

    Raises:
        LLMProviderError: Se o nome do provedor não for reconhecido.
    """
    provider = settings.llm_provider.strip().lower()
    if provider in _DISABLED:
        return None

    model_provider = _PROVIDER_ALIASES.get(provider)
    if model_provider is None:
        raise LLMProviderError(
            f"Provedor de LLM desconhecido: {settings.llm_provider!r}. "
            f"Suportados: {', '.join(sorted(_PROVIDER_ALIASES))} ou 'none'."
        )

    match provider:
        case "gemini" | "google" | "google_genai":
            key = settings.gemini_api_key
            kwargs: dict[str, Any] = {"google_api_key": key} if key else {}
        case "groq":
            key = settings.groq_api_key
            kwargs = {"api_key": key} if key else {}
        case "openai":
            key = settings.openai_api_key
            kwargs = {"api_key": key} if key else {}
        case "together" | "togetherai":
            key = settings.together_api_key
            kwargs = {"api_key": key} if key else {}
        case "openrouter":
            key = settings.openrouter_api_key
            kwargs = {"api_key": key, "base_url": settings.openrouter_base_url}
        case _:  # pragma: no cover - guardado por _PROVIDER_ALIASES
            raise LLMProviderError(settings.llm_provider)

    if not key:
        logger.warning(
            "llm_disabled_missing_key",
            provider=model_provider,
            hint="defina a chave de API no .env ou use LLM_PROVIDER=none",
        )
        return None

    return _ProviderConfig(model_provider=model_provider, kwargs=kwargs)


def build_chat_model(settings: Settings | None = None) -> BaseChatModel | None:
    """Instancia o chat model do provedor configurado (ou ``None``).

    Degrada graciosamente: retorna ``None`` (o sistema opera em modo
    determinístico/heurístico) quando o LLM está desligado, sem chave, com o
    pacote do provedor ausente ou em qualquer falha de inicialização. Nunca
    levanta em runtime — apenas registra ``warning`` para auditoria.

    Args:
        settings: Configuração opcional (default: ``get_settings()``).

    Returns:
        ``BaseChatModel`` pronto para ``invoke`` ou ``None``.
    """
    cfg_settings = settings or get_settings()
    try:
        provider_cfg = _resolve(cfg_settings)
    except LLMProviderError as exc:
        logger.warning("llm_provider_invalid", error=str(exc))
        return None
    if provider_cfg is None:
        return None

    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        logger.warning(
            "llm_langchain_missing",
            hint="instale o pacote do provedor, ex: pip install 'banco-agil[llm-gemini]'",
        )
        return None

    try:
        model = init_chat_model(
            cfg_settings.llm_model,
            model_provider=provider_cfg.model_provider,
            temperature=cfg_settings.llm_temperature,
            **provider_cfg.kwargs,
        )
    # Captura ampla é intencional: provider package ausente, credencial
    # inválida ou incompatibilidade de versão não devem derrubar a app.
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm_init_failed",
            provider=provider_cfg.model_provider,
            model=cfg_settings.llm_model,
            error=str(exc),
        )
        return None

    logger.info(
        "llm_ready",
        provider=provider_cfg.model_provider,
        model=cfg_settings.llm_model,
    )
    return model
