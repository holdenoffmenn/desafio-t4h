"""Camada de voz do assistente: humaniza a resposta determinística via LLM.

Princípio "LLM conversa, código decide": o código continua sendo a fonte da
verdade — ele decide *o que* dizer (fatos, números, decisões e perguntas). O
LLM apenas reescreve essa mensagem com naturalidade e cordialidade, sem alterar
nenhum fato. Uma guarda determinística verifica que todos os números da
mensagem original permanecem na versão reescrita; caso contrário, devolve o
texto original. Sem LLM configurado, o texto canônico é usado como está.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from banco_agil.llm._content import content_to_text
from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "Você é o assistente virtual do Banco Ágil. Reescreva a MENSAGEM DO SISTEMA "
    "com naturalidade e cordialidade, em português do Brasil, com tom "
    "profissional e conciso, sem emojis.\n"
    "Regras invioláveis:\n"
    "- Preserve EXATAMENTE todos os números, valores em reais, percentuais, "
    "datas, o score e o status (aprovado/rejeitado). Nunca altere, arredonde, "
    "converta por extenso nem invente valores.\n"
    "- Mantenha todas as perguntas e todas as opções oferecidas ao cliente.\n"
    "- Não acrescente informações, promessas ou decisões que não estejam na "
    "mensagem original.\n"
    "- Não responda ao cliente além de reformular a mensagem.\n"
    "- Responda apenas com a mensagem reformulada, sem aspas nem comentários."
)

_NUMBER_RE = re.compile(r"\d[\d.,]*")


class LlmResponder:
    """Reescreve mensagens do sistema em linguagem natural com um ``BaseChatModel``.

    Colaboradores:
        model: Qualquer chat model LangChain (Gemini, Groq, OpenAI, ...).

    Qualquer falha (rede, quota, perda de fatos) degrada para o texto original.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Inicializa o responder.

        Args:
            model: Chat model já instanciado pela factory.
        """
        self._model = model

    def humanize(self, text: str) -> str:
        """Reescreve ``text`` de forma natural, preservando todos os fatos.

        Args:
            text: Mensagem canônica (determinística) do sistema.

        Returns:
            Versão humanizada, ou o próprio ``text`` em caso de falha ou se a
            reescrita não preservar os números da mensagem original.
        """
        if not text.strip():
            return text

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self._model.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=text)]
            )
        # Falha de rede/quota/timeout não pode derrubar o turno.
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_humanize_failed", error=str(exc))
            return text

        rewritten = content_to_text(response.content).strip()
        if not rewritten:
            return text
        if not _preserves_numbers(text, rewritten):
            logger.warning("llm_humanize_dropped_facts")
            return text
        return rewritten


def _preserves_numbers(original: str, rewritten: str) -> bool:
    """Verifica se todos os números da mensagem original estão na reescrita.

    Compara apenas os dígitos de cada token numérico (ignorando separadores),
    de modo que ``R$ 3.000,00`` deva reaparecer com os mesmos dígitos e não
    seja substituído por ``três mil`` ou ``3.000``.

    Args:
        original: Mensagem canônica.
        rewritten: Mensagem reescrita pelo LLM.

    Returns:
        ``True`` se nenhum valor numérico foi perdido ou alterado.
    """
    rewritten_digits = re.sub(r"\D", "", rewritten)
    for token in _NUMBER_RE.findall(original):
        digits = re.sub(r"\D", "", token)
        if digits and digits not in rewritten_digits:
            return False
    return True


def make_llm_responder(model: BaseChatModel | None) -> LlmResponder | None:
    """Cria o responder a partir de um chat model (ou ``None``).

    Args:
        model: Chat model ou ``None`` (LLM desligado).

    Returns:
        ``LlmResponder`` ou ``None`` — neste caso a resposta determinística é
        exibida como está.
    """
    if model is None:
        return None
    return LlmResponder(model)
