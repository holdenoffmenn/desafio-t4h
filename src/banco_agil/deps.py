"""Container de dependências injetáveis no grafo (DI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from banco_agil.config import Settings, get_settings
from banco_agil.domain.auth import AuthService
from banco_agil.domain.credit_limit import CreditLimitService
from banco_agil.domain.scoring import ScoringService
from banco_agil.infrastructure.credit_request_repository import CsvCreditRequestRepository
from banco_agil.infrastructure.customer_repository import CsvCustomerRepository
from banco_agil.infrastructure.fx_client import FxClient
from banco_agil.infrastructure.score_limit_repository import CsvScoreLimitRepository
from banco_agil.ml.intent_router import SemanticIntentRouter
from banco_agil.ml.safety_classifier import SafetyClassifier


@dataclass
class AppDeps:
    """Dependências de infraestrutura e domínio para os nós do grafo.

    Attributes:
        settings: Configuração tipada.
        customers: Repositório de clientes.
        credit_requests: Repositório de solicitações.
        score_limits: Tabela score → limite.
        fx: Cliente de câmbio.
        auth: Serviço de autenticação.
        scoring: Serviço de score.
        credit_limit: Serviço de avaliação de limite.
        intent_router: Classificador de intenções.
        safety: Filtro de segurança.
    """

    settings: Settings
    customers: CsvCustomerRepository
    credit_requests: CsvCreditRequestRepository
    score_limits: CsvScoreLimitRepository
    fx: FxClient
    auth: AuthService
    scoring: ScoringService
    credit_limit: CreditLimitService
    intent_router: SemanticIntentRouter
    safety: SafetyClassifier


def build_deps(
    settings: Settings | None = None,
    *,
    data_dir: Path | None = None,
    models_dir: Path | None = None,
) -> AppDeps:
    """Monta o container de dependências a partir das Settings.

    Args:
        settings: Settings opcional (default: ``get_settings()``).
        data_dir: Sobrescreve diretório de dados (testes).
        models_dir: Sobrescreve diretório de modelos (testes).

    Returns:
        AppDeps pronto para injeção no workflow.
    """
    cfg = settings or get_settings()
    root_data = data_dir or cfg.data_dir
    root_models = models_dir or cfg.models_dir

    return AppDeps(
        settings=cfg,
        customers=CsvCustomerRepository(root_data / "clientes.csv"),
        credit_requests=CsvCreditRequestRepository(root_data / "solicitacoes_aumento_limite.csv"),
        score_limits=CsvScoreLimitRepository(root_data / "score_limite.csv"),
        fx=FxClient(cfg.fx_api_url, mock=cfg.fx_mock),
        auth=AuthService(),
        scoring=ScoringService(),
        credit_limit=CreditLimitService(),
        intent_router=SemanticIntentRouter(
            root_models / "intent_router.joblib",
            threshold=cfg.router_confidence_threshold,
        ),
        safety=SafetyClassifier(
            root_models / "safety_clf.joblib",
            threshold=cfg.safety_threshold,
        ),
    )
