"""Configuração central da aplicação via variáveis de ambiente."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações tipadas carregadas de `.env` e do ambiente.

    Attributes:
        llm_provider: Provedor de LLM (`gemini`, `groq`, `openai`,
            `together`, `openrouter` ou `none`/`fake` para desligar).
        llm_model: Nome do modelo no provedor escolhido.
        llm_temperature: Temperatura de amostragem (0 = determinístico).
        groq_api_key: Chave da API Groq (opcional na Parte 1).
        gemini_api_key: Chave da API Gemini / Google AI Studio (opcional).
        openai_api_key: Chave da API OpenAI (opcional).
        together_api_key: Chave da API TogetherAI (opcional).
        openrouter_api_key: Chave da API OpenRouter (opcional).
        openrouter_base_url: Endpoint OpenAI-compatível do OpenRouter.
        langfuse_public_key: Chave pública Langfuse.
        langfuse_secret_key: Chave secreta Langfuse.
        langfuse_host: Host do Langfuse Cloud/self-hosted.
        fx_api_url: Template de URL da API de câmbio (`{pair}` substituível).
        fx_mock: Se True, FxClient retorna cotação mockada.
        router_confidence_threshold: Limiar mínimo para aceitar intent semântico.
        safety_enabled: Liga/desliga o filtro de intenções maliciosas.
        safety_threshold: Limiar mínimo para bloquear por score do modelo.
        checkpointer_db: Caminho do SQLite do checkpointer LangGraph.
        data_dir: Diretório dos CSVs e datasets.
        models_dir: Diretório dos artefatos `.joblib`.
        api_base_url: Base URL da API (consumida pela UI).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "gemini"
    llm_model: str = "gemini-flash-latest"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    together_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    fx_api_url: str = "https://economia.awesomeapi.com.br/json/last/{pair}"
    fx_mock: bool = False

    router_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    safety_enabled: bool = True
    safety_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    checkpointer_db: Path = Path("data/sessions.sqlite")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    api_base_url: str = "http://localhost:8000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância cacheada de Settings.

    Returns:
        Settings carregadas do ambiente / `.env`.
    """
    return Settings()
