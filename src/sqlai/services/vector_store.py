from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import json

try:
    import chromadb
    from chromadb.api import ClientAPI as ChromaClient
    from chromadb.api.models.Collection import Collection
except ImportError:  # pragma: no cover - handled dynamically when dependency missing
    chromadb = None
    ChromaClient = Any  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment]

from sqlai.config import EmbeddingConfig, VectorStoreConfig
from sqlai.services.graph_cache import GraphCache, GraphCardRecord
from sqlai.utils.vector_store_utils import build_vector_store_namespace

LOGGER = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Persists graph card embeddings and serves similarity queries.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        cache_dir: Path,
        embedding_config: EmbeddingConfig,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self._client: Optional[ChromaClient] = None
        self._collection: Optional[Collection] = None
        self.enabled = True
        self.embedding_config = embedding_config
        self.namespace = self._build_namespace()
        self.store_path = self._compute_store_path()
        if chromadb is None:
            LOGGER.warning(
                "Vector store requires 'chromadb' to be installed. Install it or remove vector store usage.",
            )
            raise RuntimeError("Chromadb is required for persistent embeddings. Install it with `pip install chromadb`.")
        self._write_metadata()

    def refresh_tables(
        self,
        schema: str,
        tables: Iterable[str],
        graph_cache: GraphCache,
        embedder: Optional["EmbeddingProvider"],  # noqa: F821
    ) -> None:
        if not self._is_ready(embedder):
            return
        for table in tables:
            records = graph_cache.get_cards_for_table(schema, table)
            self._replace_records(schema, table, records, embedder)  # type: ignore[arg-type]
            LOGGER.info(
                "Vector store: embedded %s card(s) for table %s",
                len(records),
                table,
            )

    def delete_tables(self, schema: str, tables: Iterable[str]) -> None:
        if not self.enabled:
            return
        tables = list(tables)
        if not tables:
            return
        collection = self._ensure_collection()
        try:
            collection.delete(where={"schema": schema, "table": {"$in": tables}})
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to delete vectors for tables %s: %s", tables, exc)

    def query(
        self,
        question: str,
        embedder: Optional["EmbeddingProvider"],  # noqa: F821
        *,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar embeddings.
        
        Args:
            question: The query text to search for
            embedder: The embedding provider to use
            top_k: Number of results to return
            where: Optional metadata filter (ChromaDB where clause)
                   Example: {"card_type": "table"} or {"table": {"$in": ["test_sets", "executions"]}}
        """
        if not self._is_ready(embedder):
            return []
        try:
            embedding = embedder.embed_many([question])[0]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to embed query for vector search: %s", exc)
            return []
        collection = self._ensure_collection()
        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": top_k,
                "include": ["metadatas", "distances", "documents"],
            }
            if where is not None:
                query_kwargs["where"] = where
            result = collection.query(**query_kwargs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vector store query failed: %s", exc)
            return []
        hits: List[Dict[str, Any]] = []
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        for idx, metadata in enumerate(metadatas[0]):
            if not metadata:
                continue
            distance = float(distances[0][idx]) if distances and distances[0] else 0.0
            score = 1.0 / (1.0 + distance)
            hits.append(
                {
                    "vector_id": ids[0][idx] if ids and ids[0] else "",
                    "metadata": metadata,
                    "document": documents[0][idx] if documents and documents[0] else "",
                    "score": score,
                }
            )
        return hits

    def list_vectors_by_table(
        self,
        schema: Optional[str] = None,
        *,
        limit: int = 100_000,
    ) -> Dict[str, Set[str]]:
        """
        Return a mapping of table_name -> set(vector_id) currently stored in ChromaDB.

        Args:
            schema: Optional schema filter. When provided, only vectors with matching
                schema metadata are returned.
            limit: Maximum number of vectors to retrieve from Chroma. Defaults to
                100k which is sufficient for typical installations. Increase if needed.
        """
        if not self.enabled:
            return {}
        collection = self._ensure_collection()
        get_kwargs: Dict[str, Any] = {
            "include": ["metadatas"],
            "limit": limit,
        }
        if schema:
            get_kwargs["where"] = {"schema": schema}
        try:
            result = collection.get(**get_kwargs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Vector store listing failed for schema '%s': %s",
                schema or "*",
                exc,
            )
            return {}

        ids: List[str] = result.get("ids") or []
        metadatas: List[Dict[str, Any]] = result.get("metadatas") or []
        table_map: Dict[str, Set[str]] = {}

        for idx, vector_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            table_name = metadata.get("table") or "__unknown__"
            table_map.setdefault(table_name, set()).add(vector_id)

        return table_map

    def _ensure_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection
        client = self._ensure_client()
        self._collection = client.get_or_create_collection(
            name=self.config.collection,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _ensure_client(self) -> ChromaClient:
        if self._client is not None:
            return self._client
        if chromadb is None:  # pragma: no cover - guarded earlier
            raise RuntimeError("chromadb is not available.")
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.store_path))
        return self._client

    def _replace_records(
        self,
        schema: str,
        table: str,
        records: Sequence[GraphCardRecord],
        embedder: "EmbeddingProvider",  # noqa: F821
    ) -> None:
        collection = self._ensure_collection()
        try:
            collection.delete(where={"schema": schema, "table": table})
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Unable to clear existing vectors for table %s: %s", table, exc)
        if not records:
            return
        documents = [record.text for record in records]
        try:
            embeddings = embedder.embed_many(documents)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to embed graph cards for table '%s': %s", table, exc)
            raise RuntimeError(
                f"Embedding provider failed while processing table '{table}'. "
                "Resolve the embedding issue and rerun hydration."
            ) from exc
        ids = [record.vector_id for record in records]
        metadatas: List[Dict[str, Any]] = []
        for record in records:
            meta = {
                "schema": schema,
                "table": table,
                "card_type": record.card_type,
                "identifier": record.identifier,
                "schema_hash": record.schema_hash,
            }
            meta.update(record.metadata or {})
            metadatas.append(self._normalise_metadata(meta))
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to upsert vectors for table '%s': %s", table, exc)
            raise RuntimeError(
                f"Vector store failed while upserting table '{table}'. "
                "Investigate Chroma storage before proceeding."
            ) from exc

    def _is_ready(self, embedder: Optional["EmbeddingProvider"]) -> bool:  # noqa: F821
        if not self.enabled or embedder is None:
            return False
        if not getattr(embedder, "supports_embeddings", False):
            LOGGER.debug("Embedding provider does not support vector extraction; skipping vector store.")
            return False
        return True

    def _normalise_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalised: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalised[key] = value
            else:
                try:
                    normalised[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    normalised[key] = str(value)
        return normalised

    def _build_namespace(self) -> str:
        """Build namespace using shared utility function for consistency."""
        return build_vector_store_namespace(self.embedding_config)

    def _compute_store_path(self) -> Path:
        base = Path(self.config.path) if self.config.path else (self.cache_dir / "vector_store")
        return base / self.namespace

    def _metadata_path(self) -> Path:
        return self.store_path / "metadata.json"

    def _write_metadata(self) -> None:
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)
            metadata = {
                "provider": self.embedding_config.provider,
                "model": self.embedding_config.model,
                "vector_provider": self.config.provider,
                "collection": self.config.collection,
            }
            self._metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to write vector store metadata: %s", exc)


class EmbeddingProviderProtocol:
    """
    Protocol used for static typing where mypy is not available in runtime.
    """

    supports_embeddings: bool

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


