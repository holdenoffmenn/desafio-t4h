"""Roteador semântico de intenções com limiar de confiança e degradação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RouteResult:
    """Resultado do roteamento semântico.

    Attributes:
        intent: Classe predita.
        confidence: Probabilidade da classe vencedora (0–1).
        source: Sempre ``semantic`` neste componente (fallback LLM é externo).
    """

    intent: str
    confidence: float
    source: Literal["semantic"] = "semantic"


class SemanticIntentRouter:
    """Classifica intenção com um Pipeline sklearn e limiar de confiança.

    Carrega um artefato ``sklearn_pipeline`` (Tfidf + LogisticRegression) e
    retorna ``None`` quando a confiança é baixa ou o artefato está ausente —
    sinal para o caller acionar o fallback LLM.
    """

    def __init__(self, artifact_path: Path, threshold: float) -> None:
        """Inicializa o roteador.

        Args:
            artifact_path: Caminho do ``.joblib`` treinado.
            threshold: Confiança mínima para aceitar a predição.
        """
        self._threshold = threshold
        self._bundle: dict[str, Any] | None = None
        if artifact_path.exists():
            loaded = joblib.load(artifact_path)
            if isinstance(loaded, dict):
                self._bundle = loaded

    @property
    def available(self) -> bool:
        """True se um artefato válido foi carregado."""
        return self._bundle is not None

    def predict(self, text: str) -> RouteResult | None:
        """Prediz a intenção ou retorna None (fallback LLM).

        Args:
            text: Mensagem do usuário.

        Returns:
            RouteResult se confiança >= threshold; None caso contrário
            ou se o artefato estiver ausente.
        """
        if self._bundle is None:
            return None

        pipeline = self._bundle.get("pipeline")
        if pipeline is None:
            return None
        labels = list(self._bundle.get("labels", pipeline.classes_))
        probs: NDArray[np.floating] = np.asarray(
            pipeline.predict_proba([text])[0], dtype=np.float64
        )

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if confidence < self._threshold:
            return None
        return RouteResult(intent=str(labels[idx]), confidence=confidence)
