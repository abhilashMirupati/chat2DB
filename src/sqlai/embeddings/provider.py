"""
Embedding provider abstractions for Graph-RAG retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence

import numpy as np
from huggingface_hub import InferenceClient

from sqlai.config import EmbeddingConfig

LOGGER = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Return embeddings for the supplied texts."""

    def embed(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class NullEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [np.zeros(1)] * len(texts)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        if not model:
            raise ValueError("Hugging Face embedding provider requires a model name.")
        self.model = model
        kwargs = {"token": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        try:
            self.client = InferenceClient(model=model, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialise Hugging Face client: {exc}") from exc

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for text in texts:
            try:
                response = self.client.text_embeddings(
                    inputs=text,
                    model=self.model,
                )
            except Exception:  # noqa: BLE001
                LOGGER.debug("text_embeddings not available, falling back to feature extraction.")
                response = self.client.feature_extraction(text)
            vector = _coerce_embedding(response)
            vectors.append(vector)
        return vectors


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Placeholder for future Ollama embedding integration.
    """

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model or "nomic-embed-text"
        self.base_url = base_url or "http://localhost:11434"

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        raise NotImplementedError(
            "Ollama embedding provider is not yet implemented. "
            "Compose a PR or switch to Hugging Face in the meantime."
        )


def _coerce_embedding(response: object) -> np.ndarray:
    """
    Convert various HF inference responses into numpy vectors.
    """

    if isinstance(response, dict):
        if "embeddings" in response:
            response = response["embeddings"]
        elif "data" in response and isinstance(response["data"], list):
            response = response["data"][0]
    if isinstance(response, list):
        if response and isinstance(response[0], list):
            response = response[0]
        return np.array(response, dtype=float)
    raise ValueError(f"Unexpected embedding response: {response!r}")


def create_embedding_provider(config: Optional[EmbeddingConfig]) -> Optional[EmbeddingProvider]:
    if not config or config.provider == "none":
        return None
    provider = config.provider.lower()
    if provider == "huggingface":
        if not config.model or not config.api_key:
            raise ValueError("Hugging Face embeddings require both model and api key.")
        return HuggingFaceEmbeddingProvider(model=config.model, api_key=config.api_key, base_url=config.base_url)
    if provider == "ollama":
        return OllamaEmbeddingProvider(model=config.model or "nomic-embed-text", base_url=config.base_url)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")

