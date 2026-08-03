"""Hierarquia de exceções de domínio do Banco Ágil."""


class DomainError(Exception):
    """Erro base da camada de domínio.

    Camadas externas devem traduzir esta hierarquia em mensagens
    amigáveis ao cliente, sem expor stack traces.
    """


class CustomerNotFoundError(DomainError):
    """Cliente não encontrado na base (por CPF)."""


class ScoreTableError(DomainError):
    """Faixa de score inexistente ou tabela ``score_limite`` inconsistente."""


class PersistenceError(DomainError):
    """Falha de leitura/escrita em persistência (CSV, lock, etc.)."""


class FxUnavailableError(DomainError):
    """API de câmbio indisponível ou resposta inválida."""


class FxPairNotFoundError(FxUnavailableError):
    """Par de moedas inexistente na fonte de câmbio (ex.: CoinNotExists)."""
