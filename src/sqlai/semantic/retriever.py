"""
Semantic retrieval utilities for Graph-RAG selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests
import os
from huggingface_hub import InferenceClient

from sqlai.config import EmbeddingConfig
from sqlai.graph.context import ColumnCard, GraphContext, RelationshipCard, TableCard
from sqlai.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


class _SimilarityProvider:
    def similarities(self, query: str, texts: Sequence[str]) -> List[float]:
        raise NotImplementedError


class _HuggingFaceSimilarity(_SimilarityProvider):
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None) -> None:
        parameters: Dict[str, str] = {}
        if base_url:
            parameters["base_url"] = base_url.rstrip("/")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        os.environ["HF_TOKEN"] = api_key
        self.client = InferenceClient(provider="hf-inference", api_key=api_key, **parameters)
        self.model = model

    def similarities(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        sentences = list(texts)
        try:
            response = self.client.sentence_similarity(query, sentences, model=self.model)
        except TypeError:
            payload = {"source_sentence": query, "sentences": sentences}
            LOGGER.debug("Hugging Face similarity fallback", extra={"payload": payload, "model": self.model})
            response = self.client.sentence_similarity(payload, model=self.model)
        return list(response)


class _OllamaSimilarity(_SimilarityProvider):
    def __init__(self, model: str, base_url: Optional[str] = None) -> None:
        self.model = model
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._cache: Dict[str, np.ndarray] = {}

    def similarities(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        query_vec = self._embed(query)
        sims: List[float] = []
        for text in texts:
            vec = self._embed(text)
            sims.append(_cosine_similarity(query_vec, vec))
        return sims

    def _embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            vector = np.array(data["embedding"], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            vector = np.zeros(1, dtype=np.float32)
        self._cache[text] = vector
        return vector


@dataclass
class RetrievalResult:
    tables: List[TableCard]
    columns: List[ColumnCard]
    relationships: List[RelationshipCard]
    details: Dict[str, List[Tuple[str, float]]]


class SemanticRetriever:
    """
    Hybrid heuristic + embedding-based retriever for graph cards.
    """

    def __init__(self, config: EmbeddingConfig | None) -> None:
        self.config = config or EmbeddingConfig(provider="none")
        self.provider: Optional[_SimilarityProvider] = None
        if self.config.provider == "huggingface" and self.config.model and self.config.api_key:
            self.provider = _HuggingFaceSimilarity(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        elif self.config.provider == "ollama" and self.config.model:
            self.provider = _OllamaSimilarity(
                model=self.config.model,
                base_url=self.config.base_url,
            )

    def select_cards(
        self,
        graph: GraphContext,
        question: str,
        *,
        max_tables: int = 6,
        max_columns: int = 10,
    ) -> RetrievalResult:
        heuristic_tables = graph.rank_tables(question, max_cards=max_tables)
        heuristic_columns = graph.rank_columns(question, heuristic_tables, max_cards=max_columns)

        sim_tables: List[Tuple[TableCard, float]] = []
        sim_columns: List[Tuple[ColumnCard, float]] = []

        if self.provider:
            try:
                table_texts = [card.render() for card in graph.tables]
                table_scores = self.provider.similarities(question, table_texts)
                sim_tables = sorted(
                    zip(graph.tables, table_scores),
                    key=lambda item: item[1],
                    reverse=True,
                )
                column_texts = [card.render() for card in graph.column_cards]
                column_scores = self.provider.similarities(question, column_texts)
                sim_columns = sorted(
                    zip(graph.column_cards, column_scores),
                    key=lambda item: item[1],
                    reverse=True,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Semantic similarity call failed",
                    extra={
                        "provider": type(self.provider).__name__,
                        "model": getattr(self.provider, "model", None),
                        "question": question,
                        "error": repr(exc),
                    },
                )
                sim_tables = []
                sim_columns = []

        selected_tables = self._merge_tables(heuristic_tables, sim_tables, max_tables)
        selected_columns = self._merge_columns(
            heuristic_columns,
            sim_columns,
            selected_tables,
            max_columns,
        )
        selected_relationships = graph.relationships_for_tables(selected_tables)
        details = {
            "heuristic_tables": [(card.name, 1.0) for card in heuristic_tables],
            "semantic_tables": [(card.name, float(score)) for card, score in sim_tables[:max_tables]],
            "heuristic_columns": [(f"{card.table}.{card.column.name}", 1.0) for card in heuristic_columns],
            "semantic_columns": [
                (f"{card.table}.{card.column.name}", float(score)) for card, score in sim_columns[:max_columns]
            ],
        }
        return RetrievalResult(
            tables=selected_tables,
            columns=selected_columns,
            relationships=selected_relationships,
            details=details,
        )

    def _merge_tables(
        self,
        heuristic: List[TableCard],
        semantic: List[Tuple[TableCard, float]],
        limit: int,
    ) -> List[TableCard]:
        ordered: List[TableCard] = []
        seen: Set[str] = set()

        for card in heuristic:
            if card.name not in seen:
                ordered.append(card)
                seen.add(card.name)

        for card, _score in semantic:
            if card.name not in seen:
                ordered.append(card)
                seen.add(card.name)
            if len(ordered) >= limit:
                break

        return ordered[:limit]

    def _merge_columns(
        self,
        heuristic: List[ColumnCard],
        semantic: List[Tuple[ColumnCard, float]],
        tables: List[TableCard],
        limit: int,
    ) -> List[ColumnCard]:
        table_names = {card.name for card in tables}
        ordered: List[ColumnCard] = []
        seen: Set[Tuple[str, str]] = set()

        for card in heuristic:
            key = (card.table, card.column.name)
            if key not in seen and card.table in table_names:
                ordered.append(card)
                seen.add(key)

        for card, _score in semantic:
            key = (card.table, card.column.name)
            if key not in seen and card.table in table_names:
                ordered.append(card)
                seen.add(key)
            if len(ordered) >= limit:
                break

        return ordered[:limit]

