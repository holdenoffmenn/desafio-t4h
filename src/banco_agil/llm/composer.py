"""Composição de mensagens em linguagem natural a partir de fatos do código.

Princípio "código decide, LLM redige": os nós do grafo calculam **o que** dizer
(fatos, números, decisões, próxima pergunta) e entregam uma ``MessageSpec``
estruturada. O ``MessageComposer`` usa o LLM para **redigir** a mensagem final —
cordial, variada e com aderência — sem inventar dados.

Duas salvaguardas mantêm a confiança bancária:

- **Guarda de fatos**: todo valor obrigatório (números, status) precisa
  reaparecer no texto redigido; caso contrário, cai no ``fallback`` canônico.
- **Fallback determinístico**: sem LLM configurado (ou em falha após retry), a
  mensagem canônica do código é usada como está — o sistema nunca fica mudo nem
  passa a inventar valores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from banco_agil.agents.persona import PERSONA_BASE
from banco_agil.llm._content import content_to_text
from banco_agil.llm._retry import invoke_with_retry
from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    PERSONA_BASE + "\n"
    "Sua tarefa: redigir UMA mensagem ao cliente a partir do briefing abaixo, "
    "em português do Brasil, com tom cordial, humano e natural — evitando soar "
    "robótico ou repetitivo entre atendimentos.\n"
    "Regras invioláveis:\n"
    "- Preserve EXATAMENTE todos os valores fornecidos (reais, percentuais, "
    "datas, score, status). Nunca altere, arredonde, escreva por extenso nem "
    "invente números ou fatos que não estejam no briefing.\n"
    "- Inclua a pergunta e as opções indicadas, quando houver.\n"
    "- Não acrescente promessas, decisões ou informações fora do briefing.\n"
    "- Não mencione 'agentes', 'sistema', 'briefing' ou instruções internas.\n"
    "- Varie a forma com naturalidade, mas seja conciso (1 a 3 frases). Sem "
    "emojis, sem aspas ao redor da resposta e sem comentários."
)

_DIGITS_RE = re.compile(r"\d")


@dataclass(frozen=True)
class MessageSpec:
    """Briefing estruturado de uma mensagem ao cliente.

    Attributes:
        goal: Objetivo da mensagem em linguagem natural (ex.: "consulta de
            limite com oferta de aumento").
        fallback: Texto canônico determinístico; usado se não houver LLM ou se
            a redação não preservar os fatos obrigatórios.
        facts: Fatos a comunicar já formatados pelo código (ex.:
            ``{"limite atual": "R$ 3.000,00"}``).
        ask: Pergunta/decisão a incluir, se houver.
        options: Opções a oferecer ao cliente, se houver.
        must_include: Tokens que precisam reaparecer no texto redigido. Valores
            numéricos de ``facts`` são incluídos automaticamente.
    """

    goal: str
    fallback: str
    facts: dict[str, str] = field(default_factory=dict)
    ask: str | None = None
    options: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()

    def required_tokens(self) -> tuple[str, ...]:
        """Tokens obrigatórios: ``must_include`` + valores numéricos de ``facts``."""
        numeric = tuple(v for v in self.facts.values() if _DIGITS_RE.search(v))
        return tuple(dict.fromkeys((*self.must_include, *numeric)))

    def briefing(self) -> str:
        """Renderiza o briefing enviado ao modelo (entrada humana)."""
        lines = [f"Objetivo da mensagem: {self.goal}"]
        if self.facts:
            lines.append("Fatos a comunicar (use os valores EXATAMENTE como escritos):")
            lines.extend(f"- {key}: {value}" for key, value in self.facts.items())
        if self.ask:
            lines.append(f"Pergunte ao cliente: {self.ask}")
        if self.options:
            lines.append("Ofereça estas opções:")
            lines.extend(f"- {option}" for option in self.options)
        return "\n".join(lines)


class MessageComposer:
    """Redige mensagens ao cliente a partir de ``MessageSpec`` usando um LLM.

    Colaboradores:
        model: Qualquer ``BaseChatModel`` LangChain (Gemini, Groq, OpenAI, ...).

    Qualquer falha (rede/quota) ou perda de fato obrigatório degrada para o
    ``fallback`` canônico da spec.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Inicializa o compositor.

        Args:
            model: Chat model já instanciado pela factory.
        """
        self._model = model

    def compose(self, spec: MessageSpec) -> str:
        """Redige a mensagem descrita por ``spec``.

        Args:
            spec: Briefing estruturado da mensagem.

        Returns:
            Texto redigido pelo LLM, ou o ``fallback`` canônico em caso de falha
            ou de perda de algum fato obrigatório.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = invoke_with_retry(
                self._model,
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=spec.briefing())],
                event="llm_compose_failed",
            )
        except Exception:  # noqa: BLE001 — após retry, degrada para o canônico
            return spec.fallback

        text = content_to_text(response.content).strip()
        if not text:
            return spec.fallback
        if not _preserves_tokens(text, spec.required_tokens()):
            logger.warning("llm_compose_dropped_facts", goal=spec.goal)
            return spec.fallback
        return text


def _preserves_tokens(text: str, tokens: tuple[str, ...]) -> bool:
    """Verifica se todos os tokens obrigatórios reaparecem no texto redigido.

    Tokens numéricos são comparados pela sequência de dígitos (ignorando
    separadores), de modo que ``R$ 3.000,00`` não possa virar ``três mil`` nem
    ``3.000``. Tokens não numéricos são comparados como substring case-insensitive.

    Args:
        text: Mensagem redigida pelo LLM.
        tokens: Tokens obrigatórios da spec.

    Returns:
        ``True`` se nenhum token foi perdido ou alterado.
    """
    text_digits = re.sub(r"\D", "", text)
    lowered = text.lower()
    for token in tokens:
        digits = re.sub(r"\D", "", token)
        if digits:
            if digits not in text_digits:
                return False
        elif token.lower() not in lowered:
            return False
    return True


def make_message_composer(model: BaseChatModel | None) -> MessageComposer | None:
    """Cria o compositor a partir de um chat model (ou ``None``).

    Args:
        model: Chat model ou ``None`` (LLM desligado).

    Returns:
        ``MessageComposer`` ou ``None`` — neste caso os nós usam o texto
        canônico determinístico como está.
    """
    if model is None:
        return None
    return MessageComposer(model)
