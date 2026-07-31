"""Roteador semântico de intenções com limiar de confiança e degradação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from numpy.typing import NDArray

from banco_agil.ml.embeddings import EmbeddingsProvider

IntentLabel = Literal["credit", "exchange", "interview", "end", "unknown"]


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
    """Classifica intenção com artefato sklearn e limiar de confiança.

    Suporta dois formatos de artefato:
        - ``sklearn_pipeline``: Pipeline (Tfidf/Hashing + LogisticRegression)
        - ``embedding_clf``: classificador que espera vetores do EmbeddingsProvider

    Retorna ``None`` quando a confiança é baixa ou o artefato está ausente —
    sinal para o caller acionar fallback LLM.
    """

    def __init__(
        self,
        artifact_path: Path,
        threshold: float,
        encoder: EmbeddingsProvider | None = None,
    ) -> None:
        """Inicializa o roteador.

        Args:
            artifact_path: Caminho do ``.joblib`` treinado.
            threshold: Confiança mínima para aceitar a predição.
            encoder: Encoder opcional (obrigatório se artefato for embedding_clf).
        """
        self._threshold = threshold
        self._encoder = encoder
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

        kind = self._bundle.get("kind", "sklearn_pipeline")
        if kind == "sklearn_pipeline":
            probs, labels = self._predict_pipeline(text)
        elif kind == "embedding_clf":
            probs, labels = self._predict_embedding(text)
        else:
            return None

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if confidence < self._threshold:
            return None
        return RouteResult(intent=str(labels[idx]), confidence=confidence)

    def _predict_pipeline(self, text: str) -> tuple[NDArray[np.floating], list[str]]:
        """Predição via Pipeline sklearn completo."""
        pipeline = self._bundle["pipeline"]  # type: ignore[index]
        labels = list(self._bundle.get("labels", pipeline.classes_))  # type: ignore[union-attr]
        probs = pipeline.predict_proba([text])[0]
        return np.asarray(probs, dtype=np.float64), labels

    def _predict_embedding(self, text: str) -> tuple[NDArray[np.floating], list[str]]:
        """Predição via encoder externo + LogisticRegression."""
        if self._encoder is None:
            raise RuntimeError("encoder é obrigatório para artefato embedding_clf")
        clf = self._bundle["clf"]  # type: ignore[index]
        labels = list(self._bundle.get("labels", clf.classes_))  # type: ignore[union-attr]
        vec = self._encoder.encode([text], normalize=True)
        probs = clf.predict_proba(vec)[0]
        return np.asarray(probs, dtype=np.float64), labels
