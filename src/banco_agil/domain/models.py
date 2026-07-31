"""Modelos de domínio (Pydantic v2) sem I/O."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from banco_agil.utils.currency import normalize_brazilian_currency


class Customer(BaseModel):
    """Cliente do Banco Ágil.

    Attributes:
        cpf: CPF normalizado (somente dígitos).
        data_nascimento: Data de nascimento.
        nome: Nome completo.
        limite_atual: Limite de crédito atual em reais.
        score: Score de crédito (0–1000).
    """

    cpf: str
    data_nascimento: date
    nome: str
    limite_atual: float = Field(ge=0)
    score: int = Field(ge=0, le=1000)


class CreditRequest(BaseModel):
    """Pedido formal de aumento de limite.

    Attributes:
        request_id: Identificador único do pedido.
        cpf_cliente: CPF do cliente autenticado.
        data_hora_solicitacao: Timestamp ISO 8601.
        limite_atual: Limite no momento da solicitação.
        novo_limite_solicitado: Valor pedido pelo cliente.
        status_pedido: ``pendente`` | ``aprovado`` | ``rejeitado``.
    """

    request_id: str
    cpf_cliente: str
    data_hora_solicitacao: datetime
    limite_atual: float
    novo_limite_solicitado: float
    status_pedido: Literal["pendente", "aprovado", "rejeitado"] = "pendente"


class InterviewInput(BaseModel):
    """Dados financeiros coletados na entrevista de crédito.

    Attributes:
        renda_mensal: Renda mensal em reais (> 0).
        tipo_emprego: formal | autônomo | desempregado.
        despesas_fixas: Despesas fixas mensais (>= 0).
        num_dependentes: Número de dependentes (0–20).
        tem_dividas: sim | não.
    """

    renda_mensal: float = Field(gt=0, description="Renda mensal em reais")
    tipo_emprego: Literal["formal", "autônomo", "desempregado"]
    despesas_fixas: float = Field(ge=0)
    num_dependentes: int = Field(ge=0, le=20)
    tem_dividas: Literal["sim", "não"]

    @field_validator("renda_mensal", "despesas_fixas", mode="before")
    @classmethod
    def parse_currency(cls, value: object) -> float:
        """Normaliza strings monetárias BR antes da validação numérica."""
        return normalize_brazilian_currency(value)


class CreditDecision(BaseModel):
    """Resultado da avaliação de aumento de limite.

    Attributes:
        status: ``aprovado`` ou ``rejeitado``.
        reason: Motivo estruturado (ex.: ``score_insuficiente``).
        approved_limit: Limite aprovado, se aplicável.
    """

    status: Literal["aprovado", "rejeitado"]
    reason: str | None = None
    approved_limit: float | None = None


class ScoreBand(BaseModel):
    """Faixa de score → limite máximo permitido.

    Attributes:
        score_min: Limite inferior inclusivo.
        score_max: Limite superior inclusivo.
        limite_max_permitido: Maior limite aprovável na faixa.
    """

    score_min: int = Field(ge=0, le=1000)
    score_max: int = Field(ge=0, le=1000)
    limite_max_permitido: float = Field(ge=0)


class ExchangeRate(BaseModel):
    """Cotação de câmbio retornada pelo FxClient.

    Attributes:
        currency: Código ISO da moeda (ex.: USD).
        bid: Preço de compra.
        ask: Preço de venda.
        timestamp: Momento da cotação (string da API ou ISO).
    """

    currency: str
    bid: float
    ask: float
    timestamp: str
