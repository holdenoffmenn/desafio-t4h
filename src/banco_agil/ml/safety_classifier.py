"""Filtro de intenções maliciosas (denylist + modelo sklearn)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(as\s+|todas\s+as\s+)?instru[çc][õo]es",
    r"(reveal|mostre|revele).*(system\s*prompt|prompt\s+do\s+sistema)",
    r"aja\s+como\s+.*sem\s+restri[çc][õo]es",
    r"desconsidere\s+.*regras",
    r"jailbreak",
    r"ignore\s+previous\s+instructions",
    r"you\s+are\s+now\s+DAN",
]


@dataclass(frozen=True)
class SafetyResult:
    """Resultado da checagem de segurança.

    Attributes:
        blocked: Se a mensagem deve ser bloqueada.
        label: ``ok`` | ``injection`` | ``abuse`` | ``off_topic``.
        score: Score de risco (0–1); 1.0 em hit de regex.
    """

    blocked: bool
    label: str
    score: float


class SafetyClassifier:
    """Combina denylist regex + classificador sklearn (defesa em profundidade).

    Degrada para regex-only se o artefato estiver ausente. Não é garantia
    contra prompt injection — reduz superfície de ataque; a proteção real
    vem do least privilege das tools.
    """

    def __init__(
        self,
        artifact_path: Path,
        threshold: float,
    ) -> None:
        """Inicializa o classificador.

        Args:
            artifact_path: Caminho do ``.joblib`` treinado.
            threshold: Limiar para bloquear por score do modelo.
        """
        self._threshold = threshold
        self._bundle: dict[str, Any] | None = None
        if artifact_path.exists():
            loaded = joblib.load(artifact_path)
            if isinstance(loaded, dict):
                self._bundle = loaded
        self._regex = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]

    @property
    def available(self) -> bool:
        """True se um artefato de modelo foi carregado."""
        return self._bundle is not None

    def check(self, text: str) -> SafetyResult:
        """Avalia se a mensagem deve ser bloqueada.

        Args:
            text: Mensagem crua do usuário.

        Returns:
            SafetyResult com blocked/label/score.
        """
        regex_hit = any(pattern.search(text) for pattern in self._regex)
        model_label, model_score = self._model_predict(text)

        if regex_hit:
            return SafetyResult(blocked=True, label="injection", score=1.0)

        blocked = model_label != "ok" and model_score >= self._threshold
        return SafetyResult(
            blocked=blocked,
            label=model_label if blocked else "ok",
            score=model_score,
        )

    def _model_predict(self, text: str) -> tuple[str, float]:
        """Predição do modelo; retorna (ok, 0.0) se sem artefato."""
        if self._bundle is None:
            return "ok", 0.0

        kind = self._bundle.get("kind", "sklearn_pipeline")
        if kind != "sklearn_pipeline":
            return "ok", 0.0

        pipeline = self._bundle["pipeline"]
        labels = list(self._bundle.get("labels", pipeline.classes_))
        probs = pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        label = str(labels[idx])
        score = float(probs[idx])

        # Score de risco: probabilidade da classe não-ok vencedora,
        # ou 1 - P(ok) se ok for a vencedora (para threshold de bloqueio).
        if label == "ok":
            return "ok", 1.0 - score
        return label, score
