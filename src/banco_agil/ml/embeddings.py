"""Providers de embeddings para roteamento semântico e safety."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Contrato de encoder de textos → vetores densos.

    Injetável para testabilidade (mock) e para trocar o backend
    (sentence-transformers, hashing, etc.) sem alterar os classifiers.
    """

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> NDArray[np.floating]:
        """Codifica uma lista de textos.

        Args:
            texts: Textos de entrada.
            normalize: Se True, normaliza L2 cada vetor.

        Returns:
            Matriz ``(n_texts, dim)``.
        """
        ...


class HashingEmbeddings:
    """Encoder local determinístico via hashing trick (sem download).

    Útil para testes e treino offline. Não é semanticamente equivalente
    a sentence-transformers, mas preserva a interface EmbeddingsProvider.
    """

    def __init__(self, dim: int = 128) -> None:
        """Inicializa o encoder.

        Args:
            dim: Dimensionalidade do vetor de saída.
        """
        self._dim = dim

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> NDArray[np.floating]:
        """Gera vetores determinísticos a partir de tokens.

        Args:
            texts: Textos de entrada.
            normalize: Se True, normaliza L2.

        Returns:
            Matriz float64 ``(n, dim)``.
        """
        vectors = np.zeros((len(texts), self._dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                idx = hash(token) % self._dim
                vectors[i, idx] += 1.0
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        return vectors


class SentenceTransformerEmbeddings:
    """Wrapper de sentence-transformers com carga lazy (opcional).

    Requires:
        Pacote opcional ``sentence-transformers`` (extra ``embeddings``).
    """

    def __init__(self, model_name: str) -> None:
        """Inicializa sem carregar o modelo ainda.

        Args:
            model_name: Identificador HuggingFace / sentence-transformers.
        """
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        """Carrega o modelo na primeira chamada."""
        if self._model is None:
            try:
                from sentence_transformers import (  # type: ignore[import-not-found]
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers não instalado. "
                    "Use: pip install 'banco-agil[embeddings]' "
                    "ou injete HashingEmbeddings / Pipeline sklearn."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> NDArray[np.floating]:
        """Codifica textos com sentence-transformers.

        Args:
            texts: Textos de entrada.
            normalize: Encaminhado como ``normalize_embeddings``.

        Returns:
            Matriz de embeddings.
        """
        model = self._ensure_model()
        # sentence_transformers.SentenceTransformer.encode
        result = model.encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float64)
