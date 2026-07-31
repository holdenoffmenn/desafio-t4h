#!/usr/bin/env python3
"""Treina e persiste o filtro de intenções maliciosas (sklearn Pipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "safety_samples.jsonl"
OUT_PATH = ROOT / "models" / "safety_clf.joblib"


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Carrega dataset rotulado de segurança.

    Args:
        path: Arquivo JSONL com campos ``text`` e ``label``.

    Returns:
        Tupla (texts, labels).
    """
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(str(row["text"]))
            labels.append(str(row["label"]))
    return texts, labels


def train(texts: list[str], labels: list[str]) -> Pipeline:
    """Treina Pipeline Tfidf + LogisticRegression multiclasse.

    Args:
        texts: Frases rotuladas.
        labels: Classes (ok, injection, abuse, off_topic).

    Returns:
        Pipeline treinado.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"),
            ),
        ]
    )

    if len(set(labels)) < 2 or len(texts) < 8:
        pipeline.fit(texts, labels)
        return pipeline

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=42,
    )
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    print(classification_report(y_test, preds, zero_division=0))  # type: ignore[arg-type]
    pipeline.fit(texts, labels)
    return pipeline


def main() -> int:
    """Treina o safety classifier e salva ``models/safety_clf.joblib``."""
    if not DATA_PATH.exists():
        print(f"Dataset não encontrado: {DATA_PATH}", file=sys.stderr)
        return 1

    texts, labels = load_dataset(DATA_PATH)
    print(f"Loaded {len(texts)} samples from {DATA_PATH}")
    pipeline = train(texts, labels)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kind": "sklearn_pipeline",
        "pipeline": pipeline,
        "labels": list(pipeline.named_steps["clf"].classes_),
    }
    joblib.dump(artifact, OUT_PATH)
    print(f"Saved artifact -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
