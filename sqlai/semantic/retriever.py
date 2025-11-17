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
from sqlai.services.vector_store import VectorStoreManager
from sqlai.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


class _SimilarityProvider:
    supports_embeddings: bool = False

    def similarities(self, query: str, texts: Sequence[str]) -> List[float]:
        raise NotImplementedError

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


class _HuggingFaceSimilarity(_SimilarityProvider):
    supports_embeddings = True

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

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        embeddings: List[Sequence[float]] = []
        for text in texts:
            output = self.client.feature_extraction(text, model=self.model)
            array = np.array(output, dtype=np.float32)
            if array.ndim > 1:
                array = array.mean(axis=0)
            embeddings.append(array.astype(np.float32).tolist())
        return embeddings


class _OllamaSimilarity(_SimilarityProvider):
    supports_embeddings = True

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

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._embed(text).astype(np.float32).tolist() for text in texts]


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

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        vector_store: Optional[VectorStoreManager] = None,
    ) -> None:
        if config is None:
            raise ValueError("Embedding configuration is required.")
        self.config = config
        self.provider: Optional[_SimilarityProvider] = None
        self.vector_store = vector_store
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
        max_columns: Optional[int] = None,  # None = no limit, include ALL columns
    ) -> RetrievalResult:
        """
        Select relevant graph cards using semantic search on tables only.
        Once top K tables are selected, get ALL columns and relationships for those tables.
        """
        LOGGER.info("=" * 80)
        LOGGER.info("GRAPH CARD RETRIEVAL - Starting selection for query: %s", question)
        LOGGER.info("=" * 80)
        
        # Step 1: Heuristic table ranking
        heuristic_tables = graph.rank_tables(question, max_cards=max_tables)
        LOGGER.info("Step 1 - Heuristic table ranking: %d tables", len(heuristic_tables))
        for idx, card in enumerate(heuristic_tables, 1):
            LOGGER.info("  %d. %s (heuristic score: 1.0)", idx, card.name)

        sim_tables: List[Tuple[TableCard, float]] = []
        vector_tables: List[Tuple[TableCard, float]] = []

        # Step 2: On-the-fly semantic similarity for tables (fallback if vector store not available)
        if self.provider:
            try:
                LOGGER.info("Step 2a - On-the-fly semantic similarity for tables (provider: %s)", 
                           type(self.provider).__name__)
                table_texts = [card.render() for card in graph.tables]
                table_scores = self.provider.similarities(question, table_texts)
                sim_tables = sorted(
                    zip(graph.tables, table_scores),
                    key=lambda item: item[1],
                    reverse=True,
                )
                LOGGER.info("  Found %d tables with semantic scores", len(sim_tables))
                for idx, (card, score) in enumerate(sim_tables[:max_tables], 1):
                    LOGGER.info("  %d. %s (semantic score: %.4f)", idx, card.name, score)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "On-the-fly semantic similarity call failed: %s",
                    exc,
                    extra={
                        "provider": type(self.provider).__name__,
                        "model": getattr(self.provider, "model", None),
                        "question": question,
                        "error": repr(exc),
                    },
                )
                sim_tables = []

        # Step 3: Vector store semantic search for tables only
        if self.vector_store and self.provider:
            LOGGER.info("Step 2b - Vector store semantic search for table cards only")
            vector_tables = self._vector_hits(
                question,
                graph,
                max_tables=max_tables,
            )
            LOGGER.info("  Found %d tables from vector store", len(vector_tables))
            for idx, (card, score) in enumerate(vector_tables, 1):
                LOGGER.info("  %d. %s (vector score: %.4f)", idx, card.name, score)
        else:
            LOGGER.info("Step 2b - Vector store not available (vector_store=%s, provider=%s)", 
                       self.vector_store is not None, self.provider is not None)

        # Step 4: Merge heuristic + semantic + vector table results
        LOGGER.info("Step 3 - Merging table results (heuristic + semantic + vector)")
        selected_tables = self._merge_tables(
            heuristic_tables,
            sim_tables + vector_tables,
            max_tables,
            graph,  # Pass graph for schema preference logic
        )
        LOGGER.info("  Selected %d tables after merging:", len(selected_tables))
        for idx, card in enumerate(selected_tables, 1):
            schema_prefix = f"{card.schema}." if card.schema else ""
            LOGGER.info("  %d. %s%s (columns: %d)", idx, schema_prefix, card.name, len(card.columns))
        
        # Step 4b: Expand selection to include FK-referenced tables (1-hop expansion)
        # This ensures we have complete join paths even if referenced tables weren't semantically matched
        LOGGER.info("Step 3b - Expanding table selection to include FK-referenced tables (1-hop)")
        try:
            expanded_tables = self._expand_tables_with_fk_references(
                selected_tables, 
                graph,
                max_expansion=5,  # Safety limit: max 5 additional tables
            )
            if len(expanded_tables) > len(selected_tables):
                new_tables = [t for t in expanded_tables if t not in selected_tables]
                LOGGER.info("  Expanded from %d to %d tables (+%d FK-referenced tables)", 
                           len(selected_tables), len(expanded_tables), len(new_tables))
                for card in new_tables:
                    schema_prefix = f"{card.schema}." if card.schema else ""
                    LOGGER.info("    + %s%s (added via FK reference)", schema_prefix, card.name)
                selected_tables = expanded_tables
            else:
                LOGGER.info("  No additional tables needed (all FK-referenced tables already selected)")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "FK expansion failed, using original table selection: %s", 
                exc,
                exc_info=True
            )
            # Fallback: continue with original selection
            LOGGER.info("  Continuing with %d originally selected tables", len(selected_tables))

        # Step 5: Get ALL columns for selected tables (no semantic search on columns)
        LOGGER.info("Step 4 - Getting ALL columns for selected tables (no semantic search)")
        selected_table_names = {card.name.lower() for card in selected_tables}  # Normalize to lowercase
        selected_table_names_original = {card.name for card in selected_tables}  # Keep original for logging
        LOGGER.info("  Looking for columns in tables: %s", sorted(selected_table_names_original))
        
        # Case-insensitive matching: normalize both sides to lowercase
        selected_columns = [
            card for card in graph.column_cards
            if card.table.lower() in selected_table_names
        ]
        LOGGER.info("  Found %d total columns for selected tables", len(selected_columns))
        
        # Group columns by table for logging
        columns_by_table: Dict[str, List[ColumnCard]] = {}
        for col in selected_columns:
            if col.table not in columns_by_table:
                columns_by_table[col.table] = []
            columns_by_table[col.table].append(col)
        
        for table_name in sorted(columns_by_table.keys()):
            cols = columns_by_table[table_name]
            LOGGER.info("  Table '%s': %d columns - %s", 
                       table_name, len(cols), 
                       ", ".join([c.column.name for c in cols]))
        
        # Apply max_columns limit only if explicitly set (None = no limit, include ALL columns)
        original_column_count = len(selected_columns)
        if max_columns is not None and len(selected_columns) > max_columns:
            LOGGER.warning(
                "  Column limit exceeded: %d columns found, limiting to %d (max_columns)",
                len(selected_columns), max_columns
            )
            # If we have too many columns, prioritize by heuristic ranking
            heuristic_columns = graph.rank_columns(question, selected_tables, max_cards=max_columns)
            heuristic_column_keys = {(card.table, card.column.name) for card in heuristic_columns}
            # Keep heuristic columns first, then add others up to limit
            prioritized = [c for c in selected_columns if (c.table, c.column.name) in heuristic_column_keys]
            remaining = [c for c in selected_columns if (c.table, c.column.name) not in heuristic_column_keys]
            selected_columns = prioritized + remaining[:max_columns - len(prioritized)]
            LOGGER.info("  After limiting: %d columns (prioritized: %d, remaining: %d)",
                       len(selected_columns), len(prioritized), len(remaining))
        else:
            if max_columns is None:
                LOGGER.info("  All %d columns included (no limit set)", len(selected_columns))
            else:
                LOGGER.info("  All %d columns included (within limit of %d)", 
                           len(selected_columns), max_columns)

        # Step 6: Get ALL relationships for selected tables
        LOGGER.info("Step 5 - Getting ALL relationships for selected tables")
        selected_relationships = graph.relationships_for_tables(selected_tables)
        LOGGER.info("  Found %d relationships", len(selected_relationships))
        for idx, rel in enumerate(selected_relationships, 1):
            LOGGER.info("  %d. %s", idx, rel.render())

        # Validation checks
        LOGGER.info("Step 6 - Validation checks")
        if not selected_tables:
            LOGGER.error("  ⚠️  CRITICAL: No tables selected! This will cause issues.")
            LOGGER.error("     Heuristic tables: %d", len(heuristic_tables))
            LOGGER.error("     Semantic tables: %d", len(sim_tables))
            LOGGER.error("     Vector tables: %d", len(vector_tables))
            LOGGER.error("     Available tables in graph: %s", 
                        [t.name for t in graph.tables[:10]])
            # Don't raise exception - let caller handle it, but log clearly
        else:
            LOGGER.info("  ✅ Tables: %d selected", len(selected_tables))
        
        if not selected_columns:
            LOGGER.warning("  ⚠️  WARNING: No columns found for selected tables!")
            LOGGER.warning("     Selected tables: %s", [t.name for t in selected_tables])
            LOGGER.warning("     Available tables in graph: %s", 
                          [t.name for t in graph.tables[:5]] + 
                          (["..."] if len(graph.tables) > 5 else []))
        else:
            LOGGER.info("  ✅ Columns: %d selected (%d before limit)", 
                       len(selected_columns), original_column_count)
        
        if not selected_relationships:
            LOGGER.info("  ℹ️  No relationships found (this is OK if tables are not connected)")
        else:
            LOGGER.info("  ✅ Relationships: %d selected", len(selected_relationships))

        # Summary
        LOGGER.info("=" * 80)
        LOGGER.info("RETRIEVAL SUMMARY - Passing to LLM:")
        LOGGER.info("  Tables: %d", len(selected_tables))
        LOGGER.info("  Columns: %d", len(selected_columns))
        LOGGER.info("  Relationships: %d", len(selected_relationships))
        LOGGER.info("=" * 80)

        details = {
            "heuristic_tables": [(card.name, 1.0) for card in heuristic_tables],
            "semantic_tables": [(card.name, float(score)) for card, score in sim_tables[:max_tables]],
            "vector_tables": [(card.name, float(score)) for card, score in vector_tables[:max_tables]],
            "selected_columns": [(f"{card.table}.{card.column.name}", 1.0) for card in selected_columns],
            "selected_relationships": [rel.render() for rel in selected_relationships],
        }
        return RetrievalResult(
            tables=selected_tables,
            columns=selected_columns,
            relationships=selected_relationships,
            details=details,
        )

    def _vector_hits(
        self,
        question: str,
        graph: GraphContext,
        *,
        max_tables: int,
    ) -> List[Tuple[TableCard, float]]:
        """
        Vector retrieval for table cards only:
        Query table cards to identify relevant tables.
        Once we have top K tables, we get ALL columns and relationships for those tables.
        """
        if not self.vector_store or not self.provider:
            LOGGER.debug("Vector store or provider not available, skipping vector search")
            return []
        
        # Build case-insensitive lookup: lowercase name -> TableCard
        table_map = {card.name.lower(): card for card in graph.tables}
        table_map_original = {card.name: card for card in graph.tables}  # Keep original for exact match first
        LOGGER.debug("Querying vector store for table cards (max_tables=%d, available tables in graph: %d)", 
                    max_tables, len(table_map))
        
        # Query table cards only
        table_hits: List[Tuple[TableCard, float]] = []
        try:
            table_results = self.vector_store.query(
                question,
                self.provider,
                top_k=max_tables * 2,  # Get more candidates to account for filtering
                where={"card_type": "table"},
            )
            LOGGER.debug("Vector store returned %d table results", len(table_results))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vector store query failed: %s", exc, exc_info=True)
            return []
        
        matched_count = 0
        unmatched_tables = []
        for hit in table_results:
            metadata = hit.get("metadata") or {}
            score = float(hit.get("score", 0.0))
            table_name = metadata.get("table")
            if not table_name:
                LOGGER.warning("Vector store hit missing 'table' in metadata: %s", metadata)
                continue
            
            # Try exact match first, then case-insensitive
            card = table_map_original.get(table_name) or table_map.get(table_name.lower())
            if card:
                table_hits.append((card, score))
                matched_count += 1
            else:
                unmatched_tables.append(table_name)
                LOGGER.debug("Vector store returned table '%s' not found in graph (available: %s)", 
                           table_name, list(table_map_original.keys())[:5])
        
        if unmatched_tables:
            LOGGER.warning("Vector store returned %d table(s) not found in graph: %s", 
                          len(unmatched_tables), unmatched_tables[:5])
        
        table_hits.sort(key=lambda item: item[1], reverse=True)
        table_hits = table_hits[:max_tables]
        
        LOGGER.debug("Matched %d/%d vector results to graph tables, returning top %d", 
                    matched_count, len(table_results), len(table_hits))
        
        return table_hits

    def _expand_tables_with_fk_references(
        self,
        selected_tables: List[TableCard],
        graph: GraphContext,
        max_expansion: int = 5,
    ) -> List[TableCard]:
        """
        Expand table selection to include all tables referenced by foreign keys (1-hop expansion).
        This ensures we have complete join paths even if referenced tables weren't semantically matched.
        
        Handles both:
        - Outgoing FKs: Selected table has FK -> other table (add other table) [PRIORITY]
        - Incoming FKs: Other table has FK -> selected table (add other table) [SECONDARY]
        
        Safety features:
        - Max expansion limit to prevent context explosion
        - Only 1-hop expansion (no recursive expansion)
        - Validates tables exist in graph before adding
        
        Example:
        - Selected: [table1, table2]
        - table2 has FK -> table3
        - table4 has FK -> table1
        - Returns: [table1, table2, table3, table4] (if within limit)
        """
        if not selected_tables:
            return selected_tables
        
        selected_table_names = {card.name for card in selected_tables}
        table_map = {card.name: card for card in graph.tables}
        expanded = list(selected_tables)  # Start with selected tables
        expanded_names = set(selected_table_names)
        expansion_count = 0
        
        # Find all relationships involving selected tables
        try:
            relationships = graph.relationships_for_tables(selected_tables)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to get relationships for FK expansion: %s", exc)
            return selected_tables  # Return original if relationships fail
        
        # Priority 1: Outgoing FKs (selected table -> other table)
        # These are critical for join completeness
        for rel in relationships:
            if expansion_count >= max_expansion:
                LOGGER.warning(
                    "FK expansion limit reached (%d), stopping expansion. "
                    "Some FK-referenced tables may be missing.",
                    max_expansion
                )
                break
            
            # Case 1: Selected table has FK -> referred_table (outgoing FK)
            if rel.table in selected_table_names:
                referred_table = rel.detail.referred_table
                if referred_table and referred_table not in expanded_names:
                    if referred_table in table_map:
                        expanded.append(table_map[referred_table])
                        expanded_names.add(referred_table)
                        expansion_count += 1
                        LOGGER.debug("  Adding FK-referenced table: %s (referenced by %s via FK)", 
                                   referred_table, rel.table)
                    else:
                        LOGGER.debug("  FK-referenced table '%s' not found in graph", referred_table)
        
        # Priority 2: Incoming FKs (other table -> selected table)
        # These are less critical but help with reverse joins
        for rel in relationships:
            if expansion_count >= max_expansion:
                break
            
            # Case 2: referred_table has FK -> selected table (incoming FK)
            if rel.detail.referred_table in selected_table_names:
                source_table = rel.table
                if source_table and source_table not in expanded_names:
                    if source_table in table_map:
                        expanded.append(table_map[source_table])
                        expanded_names.add(source_table)
                        expansion_count += 1
                        LOGGER.debug("  Adding FK-referencing table: %s (references %s via FK)", 
                                   source_table, rel.detail.referred_table)
                    else:
                        LOGGER.debug("  FK-referencing table '%s' not found in graph", source_table)
        
        if expansion_count > 0:
            LOGGER.info("  FK expansion: added %d table(s) (limit: %d)", expansion_count, max_expansion)
        
        return expanded

    def _merge_tables(
        self,
        heuristic: List[TableCard],
        semantic: List[Tuple[TableCard, float]],
        limit: int,
        graph: GraphContext,
    ) -> List[TableCard]:
        """
        Merge heuristic and semantic table results, prioritizing tables from the default schema
        that participate in FK relationships when conflicts exist (e.g., ext_* vs canonical tables).
        """
        ordered: List[TableCard] = []
        seen: Set[str] = set()
        default_schema = graph.schema

        # Step 1: Identify tables that participate in FK relationships
        tables_with_fks: Set[str] = set()
        for rel in graph.relationships:
            tables_with_fks.add(rel.table)
            tables_with_fks.add(rel.detail.referred_table)

        # Step 2: Collect all candidate tables (heuristic + semantic)
        all_candidates: List[TableCard] = []
        for card in heuristic:
            if card.name not in seen:
                all_candidates.append(card)
                seen.add(card.name)

        for card, _score in semantic:
            if card.name not in seen:
                all_candidates.append(card)
                seen.add(card.name)

        # Step 3: Schema preference: prioritize default schema tables with FKs
        # Group tables by base name to detect conflicts (same table name in different schemas,
        # or ext_* vs canonical names)
        by_base_name: Dict[str, List[TableCard]] = {}
        for card in all_candidates:
            # Normalize base name: remove ext_ prefix and schema qualification
            base_name = card.name.lower()
            # Remove ext_ prefix if present (e.g., "ext_test_cases" -> "test_cases")
            if base_name.startswith("ext_"):
                base_name = base_name[4:]
            # Remove schema prefix if embedded in name (shouldn't happen, but handle it)
            if "." in base_name:
                base_name = base_name.split(".", 1)[1]
            
            if base_name not in by_base_name:
                by_base_name[base_name] = []
            by_base_name[base_name].append(card)

        # Step 4: For each base name, prefer default schema tables with FKs
        prioritized: List[TableCard] = []
        seen_after_priority: Set[str] = set()

        for base_name, candidates in by_base_name.items():
            if len(candidates) == 1:
                # No conflict, add directly
                card = candidates[0]
                if card.name not in seen_after_priority:
                    prioritized.append(card)
                    seen_after_priority.add(card.name)
            else:
                # Conflict detected: multiple tables with same base name
                # Priority order:
                # 1. Default schema + has FK relationships (highest priority)
                # 2. Default schema (even without FKs)
                # 3. Has FK relationships (but not default schema)
                # 4. First candidate (fallback)
                preferred = None
                for card in candidates:
                    is_default_schema = (
                        default_schema and card.schema == default_schema
                    ) or (not default_schema and not card.schema)
                    # Check if table participates in FK relationships
                    has_fk = card.name in tables_with_fks or any(
                        rel.table == card.name or rel.detail.referred_table == card.name
                        for rel in graph.relationships
                    )

                    # Priority 1: Default schema + FK
                    if is_default_schema and has_fk:
                        preferred = card
                        break
                    # Priority 2: Default schema (even without FK)
                    elif is_default_schema and preferred is None:
                        preferred = card
                    # Priority 3: Has FK (but not default schema) - only if we haven't found default schema yet
                    elif has_fk and preferred is None:
                        preferred = card

                # Fallback: use first candidate if no preference found
                if preferred is None:
                    preferred = candidates[0]

                if preferred.name not in seen_after_priority:
                    prioritized.append(preferred)
                    seen_after_priority.add(preferred.name)

                    # Log schema preference decision
                    if len(candidates) > 1:
                        other_schemas = [
                            f"{c.schema}.{c.name}" if c.schema else c.name
                            for c in candidates
                            if c.name != preferred.name
                        ]
                        preferred_name = (
                            f"{preferred.schema}.{preferred.name}"
                            if preferred.schema
                            else preferred.name
                        )
                        LOGGER.info(
                            "  Schema preference: selected %s (default schema with FK) over %s",
                            preferred_name,
                            other_schemas,
                        )

        # Step 5: Ensure we haven't missed any tables (safety check)
        # All candidates should have been processed in Step 4, but this handles edge cases
        for card in all_candidates:
            if card.name not in seen_after_priority and len(prioritized) < limit:
                # Only add if it doesn't conflict with already prioritized tables
                base_name = card.name.lower()
                if base_name.startswith("ext_"):
                    base_name = base_name[4:]
                if "." in base_name:
                    base_name = base_name.split(".", 1)[1]
                
                # Check if we already have a table with this base name
                already_has_base_name = any(
                    (c.name.lower().startswith("ext_") and c.name.lower()[4:] == base_name) or
                    (not c.name.lower().startswith("ext_") and c.name.lower() == base_name) or
                    (c.name.lower().split(".", 1)[-1] == base_name if "." in c.name.lower() else False)
                    for c in prioritized
                )
                
                if not already_has_base_name:
                    prioritized.append(card)
                    seen_after_priority.add(card.name)

        return prioritized[:limit]

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

