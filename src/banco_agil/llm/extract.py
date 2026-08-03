"""Extração de dados em linguagem natural via LLM (entrevista e valores).

Segue o princípio "LLM conversa e **interpreta**, código decide": a LLM apenas
normaliza a resposta do cliente (ex.: "sete mil" → ``7000``, "sou CLT" →
``formal``); a validação de tipos e toda decisão financeira permanecem no código
determinístico. É um **fallback**: só é acionado quando a heurística de
``utils.conversation`` não conseguiu interpretar a resposta, preservando a
degradação graciosa quando não há LLM configurado.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from banco_agil.llm._content import content_to_text
from banco_agil.llm._retry import invoke_with_retry
from banco_agil.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

Emprego = Literal["formal", "autônomo", "desempregado"]
MoneyField = Literal["renda_mensal", "despesas_fixas", "limite_credito"]

_NULL_TOKENS: frozenset[str] = frozenset({"null", "none", "nao informado", "não informado", ""})

_MONEY_LABEL: dict[str, str] = {
    "renda_mensal": "renda mensal bruta",
    "despesas_fixas": "despesas fixas mensais",
    "limite_credito": "novo limite de crédito desejado",
}

_MONEY_PROMPT = (
    "Você extrai valores monetários em reais (BRL) de mensagens de clientes. "
    "O usuário está informando: {label}. Converta a resposta para um número. "
    "Exemplos: 'sete mil' -> 7000; 'R$ 3.500,50' -> 3500.50; 'quinze mil reais' "
    "-> 15000; 'uns 2 mil e quinhentos' -> 2500. "
    "Responda APENAS com o número (use ponto como separador decimal, sem "
    "símbolos nem separador de milhar) ou a palavra null se não houver valor."
)

_EMPREGO_PROMPT = (
    "Classifique o tipo de vínculo empregatício citado pelo usuário. "
    "'formal' = CLT, carteira assinada, servidor, empregado registrado. "
    "'autônomo' = autônomo, freelancer, PJ, MEI, empreendedor, conta própria. "
    "'desempregado' = sem emprego, desempregado. "
    "Responda APENAS uma palavra: formal, autônomo, desempregado, ou null."
)

_DEPENDENTES_PROMPT = (
    "Quantos dependentes o usuário informou possuir? "
    "Exemplos: 'nenhum' -> 0; 'não tenho' -> 0; 'dois filhos' -> 2; 'três' -> 3. "
    "Responda APENAS com um número inteiro (>= 0) ou null."
)

_DIVIDAS_PROMPT = (
    "O usuário possui dívidas ativas? Interprete a resposta em linguagem natural. "
    "'sem dívidas', 'quitei tudo', 'nada devo' -> não. "
    "'tenho dívidas', 'devo no cartão', 'estou negativado' -> sim. "
    "Responda APENAS: sim, não, ou null."
)

_CURRENCY_PROMPT = (
    "Identifique a moeda estrangeira que o cliente quer cotar contra o real (BRL). "
    "Responda APENAS com o código ISO 4217 de 3 letras em maiúsculas — qualquer moeda "
    "válida (comum ou menos usual). "
    "Exemplos: 'dólar' -> USD; 'euro' -> EUR; 'peso argentino' -> ARS; "
    "'iene japonês' -> JPY; 'libra esterlina' -> GBP; 'franco suíço' -> CHF; "
    "'dólar canadense' -> CAD; 'dólar australiano' -> AUD; 'iuan' -> CNY; "
    "'won sul-coreano' / 'moeda da Coreia do Sul' -> KRW; "
    "'baht tailandês' -> THB; 'zloty' / 'Polônia' -> PLN; 'bitcoin' -> BTC. "
    "Se não houver moeda clara, responda null."
)

_DATE_PROMPT = (
    "Extraia a data de nascimento informada pelo cliente e normalize-a. "
    "Interprete formatos variados e com erros de digitação, assumindo a ordem "
    "dia-mês-ano do padrão brasileiro. "
    "Exemplos: '19-091991' -> 1991-09-19; '19/09/1991' -> 1991-09-19; "
    "'19 de setembro de 1991' -> 1991-09-19; '1991-09-19' -> 1991-09-19. "
    "Responda APENAS no formato AAAA-MM-DD, ou null se não houver data."
)

_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


class LlmExtractor:
    """Interpreta respostas do cliente em linguagem natural usando um LLM.

    Colaboradores:
        model: Qualquer ``BaseChatModel`` LangChain (Gemini, Groq, OpenAI, ...).

    Todas as falhas (rede, quota, parsing) degradam para ``None`` — o chamador
    mantém o comportamento heurístico/determinístico.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Inicializa o extrator.

        Args:
            model: Chat model já instanciado pela factory.
        """
        self._model = model

    def money(self, text: str, field: MoneyField = "limite_credito") -> float | None:
        """Extrai um valor monetário em reais a partir de texto livre.

        Args:
            text: Mensagem do usuário (ex.: "sete mil").
            field: Campo em coleta, usado para contextualizar o prompt.

        Returns:
            Valor em reais (>= 0) ou ``None`` se não houver/erro.
        """
        raw = self._ask(_MONEY_PROMPT.format(label=_MONEY_LABEL[field]), text)
        if raw is None:
            return None
        normalized = raw.replace(" ", "").replace("r$", "")
        match = re.search(r"-?\d+(?:\.\d+)?", normalized)
        if match is None:
            return None
        try:
            value = float(match.group(0))
        except ValueError:
            return None
        return value if value >= 0 else None

    def tipo_emprego(self, text: str) -> Emprego | None:
        """Classifica o tipo de emprego a partir de texto livre.

        Args:
            text: Mensagem do usuário.

        Returns:
            ``formal`` | ``autônomo`` | ``desempregado`` ou ``None``.
        """
        raw = self._ask(_EMPREGO_PROMPT, text)
        if raw is None:
            return None
        if "desempreg" in raw:
            return "desempregado"
        if "autô" in raw or "auto" in raw:
            return "autônomo"
        if "formal" in raw:
            return "formal"
        return None

    def num_dependentes(self, text: str) -> int | None:
        """Extrai a quantidade de dependentes a partir de texto livre.

        Args:
            text: Mensagem do usuário.

        Returns:
            Inteiro (>= 0) ou ``None``.
        """
        raw = self._ask(_DEPENDENTES_PROMPT, text)
        if raw is None:
            return None
        match = re.search(r"\d+", raw)
        if match is None:
            return None
        value = int(match.group(0))
        return value if value >= 0 else None

    def tem_dividas(self, text: str) -> Literal["sim", "não"] | None:
        """Determina se o usuário possui dívidas ativas.

        Args:
            text: Mensagem do usuário.

        Returns:
            ``"sim"`` | ``"não"`` ou ``None``.
        """
        raw = self._ask(_DIVIDAS_PROMPT, text)
        if raw is None:
            return None
        if "não" in raw or "nao" in raw:
            return "não"
        if "sim" in raw:
            return "sim"
        return None

    def currency(self, text: str) -> str | None:
        """Extrai o código ISO da moeda a cotar a partir de texto livre.

        Fallback do heurístico ``extract_currency_code`` para nomes que a rede
        determinística não cobre (ex.: variações regionais). A disponibilidade
        real da cotação fica a cargo do ``FxClient`` / da API de câmbio.

        Args:
            text: Mensagem do usuário (ex.: "quanto está o peso argentino").

        Returns:
            Código ISO em maiúsculas (3 letras) ou ``None`` se não houver/erro.
        """
        raw = self._ask(_CURRENCY_PROMPT, text)
        if raw is None:
            return None
        # Exige o código como token isolado ("usd", "ars.") para não capturar as
        # 3 primeiras letras de uma frase qualquer ("moeda" -> "moe").
        match = re.search(r"\b([a-z]{3})\b", raw)
        if match is None:
            return None
        return match.group(1).upper()

    def birth_date(self, text: str) -> str | None:
        """Extrai a data de nascimento e normaliza para ``AAAA-MM-DD``.

        Fallback do regex ``extract_date`` para formatos fora do padrão ou com
        erros de digitação (ex.: ``"19-091991"``). A validação da data em si (e
        a conferência com o cadastro) permanece no código determinístico.

        Args:
            text: Mensagem do usuário.

        Returns:
            Data ISO ``AAAA-MM-DD`` ou ``None`` se não houver/erro.
        """
        raw = self._ask(_DATE_PROMPT, text)
        if raw is None:
            return None
        match = _ISO_DATE_RE.search(raw)
        return match.group(1) if match is not None else None

    def _ask(self, system_prompt: str, text: str) -> str | None:
        """Invoca o modelo e devolve a resposta em minúsculas (ou ``None``).

        Args:
            system_prompt: Instrução de extração específica do campo.
            text: Mensagem do usuário.

        Returns:
            Texto normalizado (lower/strip) ou ``None`` se vazio/nulo/erro.
        """
        if not text.strip():
            return None

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = invoke_with_retry(
                self._model,
                [SystemMessage(content=system_prompt), HumanMessage(content=text)],
                event="llm_extract_failed",
            )
        # Após as tentativas, falha de rede/quota degrada para ``None``: o nó
        # re-pergunta o campo de forma amigável (sem inventar valores).
        except Exception:  # noqa: BLE001
            return None

        answer = content_to_text(response.content).strip().lower()
        if answer in _NULL_TOKENS:
            return None
        return answer


def make_llm_extractor(model: BaseChatModel | None) -> LlmExtractor | None:
    """Cria o extrator NL a partir de um chat model (ou ``None``).

    Args:
        model: Chat model ou ``None`` (LLM desligado).

    Returns:
        ``LlmExtractor`` ou ``None`` — neste caso os nós usam apenas a
        interpretação heurística determinística.
    """
    if model is None:
        return None
    return LlmExtractor(model)
