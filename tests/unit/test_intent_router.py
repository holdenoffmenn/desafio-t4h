"""Testes do SemanticIntentRouter."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from banco_agil.ml.intent_router import SemanticIntentRouter


def _build_artifact(path: Path) -> None:
    texts = [
        "quero aumentar meu limite de crédito",
        "consultar limite do cartão",
        "cotação do dólar hoje",
        "quanto está o euro",
        "pode encerrar a conversa",
        "quero finalizar o atendimento",
        "aceito a entrevista financeira",
        "atualizar meu score",
        "oi tudo bem",
        "não sei",
    ]
    labels = [
        "credit",
        "credit",
        "exchange",
        "exchange",
        "end",
        "end",
        "interview",
        "interview",
        "unknown",
        "unknown",
    ]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(
        {
            "kind": "sklearn_pipeline",
            "pipeline": pipeline,
            "labels": list(pipeline.named_steps["clf"].classes_),
        },
        path,
    )


def test_predict_returns_none_without_artifact(tmp_path: Path) -> None:
    router = SemanticIntentRouter(tmp_path / "missing.joblib", threshold=0.5)
    assert router.predict("quero crédito") is None
    assert router.available is False


def test_predict_high_confidence(tmp_path: Path) -> None:
    artifact = tmp_path / "intent_router.joblib"
    _build_artifact(artifact)
    router = SemanticIntentRouter(artifact, threshold=0.3)
    result = router.predict("quero aumentar meu limite de crédito")
    assert result is not None
    assert result.intent == "credit"
    assert result.source == "semantic"
    assert 0.0 <= result.confidence <= 1.0


def test_predict_low_confidence_returns_none(tmp_path: Path) -> None:
    artifact = tmp_path / "intent_router.joblib"
    _build_artifact(artifact)
    router = SemanticIntentRouter(artifact, threshold=0.999)
    # Texto ambíguo/fora do vocabulário tende a baixa confiança
    result = router.predict("xyzabc qwerty")
    assert result is None
