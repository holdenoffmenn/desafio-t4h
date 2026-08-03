"""Camada de LLM provider-agnostic (Gemini, Groq, OpenAI, Together, OpenRouter)."""

from banco_agil.llm.composer import MessageComposer, MessageSpec, make_message_composer
from banco_agil.llm.extract import LlmExtractor, make_llm_extractor
from banco_agil.llm.factory import LLMProviderError, build_chat_model
from banco_agil.llm.intent import (
    IntentFallback,
    IntentResult,
    LlmIntentClassifier,
    make_llm_intent_fallback,
)

__all__ = [
    "IntentFallback",
    "IntentResult",
    "LLMProviderError",
    "LlmExtractor",
    "LlmIntentClassifier",
    "MessageComposer",
    "MessageSpec",
    "build_chat_model",
    "make_llm_extractor",
    "make_llm_intent_fallback",
    "make_message_composer",
]
