"""
Semantic retrieval utilities for Graph-RAG selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
        fk_expansion_max_depth: int = 3,  # Maximum hop depth for FK expansion
        fk_expansion_max_tables: int = 20,  # Maximum additional tables to add via FK expansion
        table_filter: Optional[List[str]] = None,  # Optional list of table names to filter vector search
    ) -> RetrievalResult:
        """
        Select relevant graph cards using semantic search on tables only.
        Once top K tables are selected, get ALL columns and relationships for those tables.
        """
        LOGGER.info("=" * 80)
        LOGGER.info("GRAPH CARD RETRIEVAL - Starting selection for query: %s", question)
        LOGGER.info("=" * 80)
        
        # Using only vector search (embeddings are pre-computed via prewarm_metadata.py)
        # On-the-fly semantic search removed since all embeddings are already in vector store

        vector_tables: List[Tuple[TableCard, float]] = []

        # Vector store semantic search for tables only (with similarity threshold)
        if self.vector_store and self.provider:
            if table_filter:
                LOGGER.info("Step 1 - Vector store semantic search for table cards (similarity threshold: 0.5, filtered to %d tables)", len(table_filter))
            else:
                LOGGER.info("Step 1 - Vector store semantic search for table cards (similarity threshold: 0.5)")
            vector_tables = self._vector_hits(
                question,
                graph,
                max_tables=max_tables,
                similarity_threshold=0.5,
                table_filter=table_filter,
            )
            LOGGER.info("  Found %d tables from vector store (score >= 0.5)", len(vector_tables))
            for idx, (card, score) in enumerate(vector_tables, 1):
                LOGGER.info("  %d. %s (vector score: %.4f)", idx, card.name, score)
        else:
            LOGGER.warning("Vector store not available (vector_store=%s, provider=%s) - cannot retrieve tables", 
                       self.vector_store is not None, self.provider is not None)
            if not self.vector_store:
                LOGGER.warning("  Run 'python scripts/prewarm_metadata.py' to generate embeddings")

        # Use vector search results directly (no merging needed since we only use vector search)
        LOGGER.info("Step 2 - Using vector search results")
        selected_tables = self._merge_tables(
            [],  # No heuristic tables
            vector_tables,
            max_tables,
            graph,  # Pass graph for schema preference logic
        )
        LOGGER.info("  Selected %d tables after merging:", len(selected_tables))
        for idx, card in enumerate(selected_tables, 1):
            schema_prefix = f"{card.schema}." if card.schema else ""
            LOGGER.info("  %d. %s%s (columns: %d)", idx, schema_prefix, card.name, len(card.columns))
        
        # Step 4b: Expand selection to include FK-referenced tables (multi-hop expansion)
        # This ensures we have complete join paths even if intermediate tables weren't semantically matched
        LOGGER.info(
            "Step 3b - Expanding table selection to include FK-referenced tables (multi-hop, max_depth=%d, max_expansion=%d)",
            fk_expansion_max_depth, fk_expansion_max_tables
        )
        try:
            expanded_tables = self._expand_tables_with_fk_references(
                selected_tables, 
                graph,
                max_expansion=fk_expansion_max_tables,
                max_depth=fk_expansion_max_depth,
            )
            if len(expanded_tables) > len(selected_tables):
                new_tables = [t for t in expanded_tables if t not in selected_tables]
                LOGGER.info("  Expanded from %d to %d tables (+%d FK-referenced tables)", 
                           len(selected_tables), len(expanded_tables), len(new_tables))
                for card in new_tables:
                    schema_prefix = f"{card.schema}." if card.schema else ""
                    LOGGER.info("    + %s%s (added via FK reference)", schema_prefix, card.name)
                
                # Log if FK expansion added tables not in filter (for accuracy)
                if table_filter:
                    filter_set = {t.lower() for t in table_filter}
                    tables_not_in_filter = [t for t in new_tables if t.name.lower() not in filter_set]
                    if tables_not_in_filter:
                        LOGGER.info("  FK expansion added %d table(s) not in filter (required for join accuracy): %s",
                                   len(tables_not_in_filter),
                                   [t.name for t in tables_not_in_filter[:5]])
                
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
            LOGGER.error("     Vector tables (score >= 0.5): %d", len(vector_tables))
            LOGGER.error("     Available tables in graph: %s", 
                        [t.name for t in graph.tables[:10]])
            LOGGER.error("     Run 'python scripts/prewarm_metadata.py' if vector store is empty")
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
        similarity_threshold: float = 0.5,
        table_filter: Optional[List[str]] = None,  # Optional list of table names to filter vector search
    ) -> List[Tuple[TableCard, float]]:
        """
        Vector retrieval for table cards only:
        Query table cards to identify relevant tables.
        Only returns tables with similarity score >= threshold.
        Once we have top K tables, we get ALL columns and relationships for those tables.
        
        Args:
            question: User's natural language question
            graph: GraphContext containing all tables
            max_tables: Maximum number of tables to return
            similarity_threshold: Minimum similarity score (0.0-1.0) to include a table. Default: 0.5
            table_filter: Optional list of table names to filter vector search. If provided, only these tables will be searched.
        """
        if not self.vector_store or not self.provider:
            LOGGER.debug("Vector store or provider not available, skipping vector search")
            return []
        
        # Extract table names mentioned in the question (for exact matching)
        # Look for patterns like "run table", "the run table", "table run", etc.
        question_lower = question.lower()
        mentioned_tables = []
        for table_card in graph.tables:
            table_name_lower = table_card.name.lower()
            # Check if table name appears in question (with word boundaries)
            # Patterns: "run table", "the run table", "table run", "about run", "run's", etc.
            patterns = [
                f" {table_name_lower} ",
                f" {table_name_lower}'",
                f" {table_name_lower},",
                f" {table_name_lower}.",
                f" {table_name_lower}?",
                f" {table_name_lower}!",
                f"the {table_name_lower} ",
                f"the {table_name_lower}'",
                f"table {table_name_lower}",
                f"about {table_name_lower}",
                f"for {table_name_lower}",
            ]
            # Also check if question starts or ends with table name
            if question_lower.startswith(table_name_lower + " ") or question_lower.endswith(" " + table_name_lower):
                patterns.append("")  # Will match via startswith/endswith check
            if any(pattern in question_lower for pattern in patterns) or \
               question_lower.startswith(table_name_lower + " ") or \
               question_lower.endswith(" " + table_name_lower):
                mentioned_tables.append(table_card.name)
                LOGGER.info("Detected table name '%s' mentioned in question (exact name matching)", table_card.name)
        
        # Filter tables if table_filter is provided
        tables_to_search = graph.tables
        if table_filter:
            # Filter to only selected tables (case-insensitive matching)
            filter_set = {t.lower() for t in table_filter}
            tables_to_search = [t for t in graph.tables if t.name.lower() in filter_set]
            LOGGER.info("Filtering vector search to %d selected tables (from %d total tables)", 
                       len(tables_to_search), len(graph.tables))
            if not tables_to_search:
                LOGGER.warning("Table filter resulted in 0 tables - no tables match the filter. Using all tables instead.")
                tables_to_search = graph.tables
        
        # Build case-insensitive lookup: lowercase name -> TableCard
        table_map = {card.name.lower(): card for card in tables_to_search}
        table_map_original = {card.name: card for card in tables_to_search}  # Keep original for exact match first
        LOGGER.debug("Querying vector store for table cards (max_tables=%d, threshold=%.2f, available tables in graph: %d)", 
                    max_tables, similarity_threshold, len(table_map))
        
        # Query table cards only
        table_hits: List[Tuple[TableCard, float]] = []
        try:
            # Build where clause for vector store query
            where_clause: Dict[str, Any] = {"card_type": "table"}
            if table_filter and tables_to_search:
                # Filter by table names in vector store query
                table_names_in_filter = [t.name for t in tables_to_search]
                where_clause["table"] = {"$in": table_names_in_filter}
                LOGGER.debug("Vector store query filtered to %d tables: %s", len(table_names_in_filter), table_names_in_filter[:5])
            
            table_results = self.vector_store.query(
                question,
                self.provider,
                top_k=max_tables * 3,  # Get more candidates to account for threshold filtering
                where=where_clause,
            )
            LOGGER.debug("Vector store returned %d table results", len(table_results))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vector store query failed: %s", exc, exc_info=True)
            return []
        
        # Extract table names mentioned in the question (for exact matching)
        # Check ALL tables in graph, not just filtered ones, to catch any mention
        question_lower = question.lower()
        mentioned_table_names = set()
        for table_card in graph.tables:
            table_name_lower = table_card.name.lower()
            # Check if table name appears in question (with word boundaries)
            # Patterns: "run table", "the run table", "table run", "about run", "run's", etc.
            patterns = [
                f" {table_name_lower} ",
                f" {table_name_lower}'",
                f" {table_name_lower},",
                f" {table_name_lower}.",
                f" {table_name_lower}?",
                f" {table_name_lower}!",
                f"the {table_name_lower} ",
                f"the {table_name_lower}'",
                f"table {table_name_lower}",
                f"about {table_name_lower}",
                f"for {table_name_lower}",
            ]
            # Also check if question starts or ends with table name
            if any(pattern in question_lower for pattern in patterns) or \
               question_lower.startswith(table_name_lower + " ") or \
               question_lower.endswith(" " + table_name_lower):
                mentioned_table_names.add(table_card.name.lower())
                LOGGER.info("Detected table name '%s' mentioned in question (will bypass similarity threshold)", table_card.name)
        
        matched_count = 0
        filtered_count = 0
        unmatched_tables = []
        # Build set of filtered table names (case-insensitive) for threshold bypass
        # This allows user-selected tables to bypass similarity threshold
        filter_set_lower = {t.lower() for t in table_filter} if table_filter else set()
        
        for hit in table_results:
            metadata = hit.get("metadata") or {}
            score = float(hit.get("score", 0.0))
            table_name = metadata.get("table")
            if not table_name:
                LOGGER.warning("Vector store hit missing 'table' in metadata: %s", metadata)
                continue
            
            # Apply similarity threshold filter, BUT bypass threshold for:
            # 1. Explicitly filtered tables (user selected in UI)
            # 2. Tables mentioned in the question (exact name matching)
            is_in_user_filter = table_name.lower() in filter_set_lower
            is_mentioned_in_question = table_name.lower() in mentioned_table_names
            if score < similarity_threshold and not is_in_user_filter and not is_mentioned_in_question:
                filtered_count += 1
                LOGGER.debug("Filtered out table '%s' (score %.4f < threshold %.2f)", table_name, score, similarity_threshold)
                continue
            elif score < similarity_threshold and (is_in_user_filter or is_mentioned_in_question):
                reason = "user's filter" if is_in_user_filter else "mentioned in question"
                LOGGER.info("Including table '%s' despite low score (%.4f < %.2f) because it's in %s", 
                           table_name, score, similarity_threshold, reason)
            
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
        
        # Ensure all relevant tables are included:
        # 1. Tables in user's filter (even if not in vector results)
        # 2. Tables mentioned in question (even if not in vector results or below threshold)
        included_table_names = {card.name.lower() for card, _ in table_hits}
        missing_tables = []
        
        # Build full graph lookup for mentioned tables (not just filtered tables)
        full_graph_table_map = {card.name.lower(): card for card in graph.tables}
        full_graph_table_map_original = {card.name: card for card in graph.tables}
        
        # Add tables from filter if not already included
        if table_filter:
            for table_name in table_filter:
                if table_name.lower() not in included_table_names:
                    # Find the table card from graph (try filtered map first, then full graph)
                    card = table_map_original.get(table_name) or table_map.get(table_name.lower()) or \
                           full_graph_table_map_original.get(table_name) or full_graph_table_map.get(table_name.lower())
                    if card:
                        # Add with a default low score (0.5) since it wasn't in vector results
                        table_hits.append((card, 0.5))
                        missing_tables.append(table_name)
                        included_table_names.add(table_name.lower())
                        LOGGER.info("Added table '%s' from filter (not found in vector results, using default score 0.5)", table_name)
        
        # Add tables mentioned in question if not already included
        for mentioned_table_lower in mentioned_table_names:
            if mentioned_table_lower not in included_table_names:
                # Find the table card from full graph (mentioned tables might not be in filtered search)
                card = full_graph_table_map_original.get(mentioned_table_lower) or full_graph_table_map.get(mentioned_table_lower)
                if card:
                    # Add with a default score (0.6) since it was mentioned in question
                    table_hits.append((card, 0.6))
                    missing_tables.append(card.name)
                    included_table_names.add(mentioned_table_lower)
                    LOGGER.info("Added table '%s' mentioned in question (not found in vector results or below threshold, using default score 0.6)", card.name)
        
        if missing_tables:
            LOGGER.info("Added %d table(s) that weren't in vector results: %s", 
                       len(missing_tables), missing_tables)
        
        # Sort by score (descending) and limit to max_tables
        table_hits.sort(key=lambda item: item[1], reverse=True)
        table_hits = table_hits[:max_tables]
        
        LOGGER.info("Vector search results: %d matched (score >= %.2f), %d filtered (score < %.2f), returning top %d", 
                    matched_count, similarity_threshold, filtered_count, similarity_threshold, len(table_hits))
        
        return table_hits

    # COMMENTED OUT: Old 1-hop FK expansion method
    # REASON: Replaced with multi-hop recursive expansion (_expand_tables_with_fk_references_multi_hop)
    # to capture entire chains of connected tables (e.g., orders -> customers -> addresses -> regions).
    # The old method only did 1-hop expansion, which could miss intermediate tables in longer join paths.
    # The new method uses BFS traversal to follow FK relationships recursively up to max_depth levels.
    #
    # def _expand_tables_with_fk_references(
    #     self,
    #     selected_tables: List[TableCard],
    #     graph: GraphContext,
    #     max_expansion: int = 5,
    # ) -> List[TableCard]:
    #     """
    #     Expand table selection to include all tables referenced by foreign keys (1-hop expansion).
    #     This ensures we have complete join paths even if referenced tables weren't semantically matched.
    #     
    #     Handles both:
    #     - Outgoing FKs: Selected table has FK -> other table (add other table) [PRIORITY]
    #     - Incoming FKs: Other table has FK -> selected table (add other table) [SECONDARY]
    #     
    #     Safety features:
    #     - Max expansion limit to prevent context explosion
    #     - Only 1-hop expansion (no recursive expansion)
    #     - Validates tables exist in graph before adding
    #     
    #     Example:
    #     - Selected: [table1, table2]
    #     - table2 has FK -> table3
    #     - table4 has FK -> table1
    #     - Returns: [table1, table2, table3, table4] (if within limit)
    #     """
    #     ... (old implementation removed)

    def _expand_tables_with_fk_references(
        self,
        selected_tables: List[TableCard],
        graph: GraphContext,
        max_expansion: int = 20,
        max_depth: int = 3,
    ) -> List[TableCard]:
        """
        Recursively expand table selection to include all tables in FK-connected chains (multi-hop expansion).
        This ensures we have complete join paths even if intermediate tables weren't semantically matched.
        
        Uses BFS (Breadth-First Search) to traverse the graph:
        - Level 0: Initially selected tables
        - Level 1: Tables directly connected via FK (outgoing + incoming)
        - Level 2: Tables connected to Level 1 tables
        - Level N: Up to max_depth levels
        
        Handles both:
        - Outgoing FKs: Selected table has FK -> other table (add other table) [PRIORITY]
        - Incoming FKs: Other table has FK -> selected table (add other table) [SECONDARY]
        
        Safety features:
        - Max depth limit to prevent infinite loops
        - Max expansion limit to prevent context explosion
        - Deduplication using sets to ensure uniqueness
        - Validates tables exist in graph before adding
        
        Example (max_depth=3):
        - Selected: [orders]
        - Level 1: [customers] (orders.customer_id -> customers.id)
        - Level 2: [addresses] (customers.address_id -> addresses.id)
        - Level 3: [regions] (addresses.region_id -> regions.id)
        - Returns: [orders, customers, addresses, regions] (deduplicated)
        
        Args:
            selected_tables: Initially selected tables from semantic/heuristic matching
            graph: GraphContext containing all tables and relationships
            max_expansion: Maximum number of additional tables to add (safety limit)
            max_depth: Maximum hop depth (1=direct only, 2=1-hop, 3=2-hop, etc.)
        
        Returns:
            List of TableCard objects including original selection + FK-connected tables (deduplicated)
        """
        if not selected_tables:
            LOGGER.debug("  [FK Expansion] No tables to expand, returning empty list")
            return selected_tables
        
        LOGGER.debug(
            "  [FK Expansion] Starting multi-hop expansion: %d initial tables, max_depth=%d, max_expansion=%d",
            len(selected_tables), max_depth, max_expansion
        )
        LOGGER.debug("  [FK Expansion] Initial tables: %s", [t.name for t in selected_tables])
        
        table_map = {card.name: card for card in graph.tables}
        visited = {card.name for card in selected_tables}  # Track visited tables to prevent duplicates
        result = list(selected_tables)  # Start with selected tables
        expansion_count = 0
        
        # Build FK adjacency maps for efficient traversal
        # Maps table_name -> set of connected table names
        fk_outgoing: Dict[str, Set[str]] = {}  # table -> {referred_tables}
        fk_incoming: Dict[str, Set[str]] = {}  # table -> {referencing_tables}
        
        LOGGER.debug("  [FK Expansion] Building FK adjacency maps from %d relationships", len(graph.relationships))
        for rel in graph.relationships:
            source = rel.table
            target = rel.detail.referred_table
            
            # Outgoing: source -> target
            if source not in fk_outgoing:
                fk_outgoing[source] = set()
            fk_outgoing[source].add(target)
            
            # Incoming: target <- source
            if target not in fk_incoming:
                fk_incoming[target] = set()
            fk_incoming[target].add(source)
        
        LOGGER.debug(
            "  [FK Expansion] Built adjacency maps: %d tables with outgoing FKs, %d tables with incoming FKs",
            len(fk_outgoing), len(fk_incoming)
        )
        
        # BFS traversal: process level by level
        current_level = {card.name for card in selected_tables}
        actual_depth = 0  # Track actual depth reached
        
        for depth in range(1, max_depth + 1):
            actual_depth = depth
            if expansion_count >= max_expansion:
                LOGGER.warning(
                    "  [FK Expansion] Expansion limit reached (%d) at depth %d, stopping expansion. "
                    "Some FK-connected tables may be missing.",
                    max_expansion, depth
                )
                break
            
            if not current_level:
                LOGGER.debug("  [FK Expansion] No more tables to expand at depth %d, stopping", depth)
                break  # No more tables to expand
            
            LOGGER.debug("  [FK Expansion] Processing depth %d: %d tables", depth, len(current_level))
            next_level = set()
            
            # Process all tables at current depth
            for table_name in current_level:
                if expansion_count >= max_expansion:
                    break
                
                # Priority 1: Outgoing FKs (table -> other tables)
                if table_name in fk_outgoing:
                    for referred_table in fk_outgoing[table_name]:
                        if expansion_count >= max_expansion:
                            break  # Stop if limit reached
                        if referred_table not in visited:
                            if referred_table in table_map:
                                result.append(table_map[referred_table])
                                visited.add(referred_table)
                                next_level.add(referred_table)
                                expansion_count += 1
                                LOGGER.debug(
                                    "  [FK Expansion] [Depth %d] Added FK-referenced table: %s (referenced by %s via FK)",
                                    depth, referred_table, table_name
                                )
                                # Check limit after increment
                                if expansion_count >= max_expansion:
                                    break
                            else:
                                LOGGER.debug(
                                    "  [FK Expansion] [Depth %d] FK-referenced table '%s' not found in graph",
                                    depth, referred_table
                                )
                        # Break outer loop if limit reached
                        if expansion_count >= max_expansion:
                            break
        
                # Priority 2: Incoming FKs (other tables -> table)
                if expansion_count >= max_expansion:
                    break  # Stop if limit reached
                if table_name in fk_incoming:
                    for referencing_table in fk_incoming[table_name]:
                        if expansion_count >= max_expansion:
                            break  # Stop if limit reached
                        if referencing_table not in visited:
                            if referencing_table in table_map:
                                result.append(table_map[referencing_table])
                                visited.add(referencing_table)
                                next_level.add(referencing_table)
                                expansion_count += 1
                                LOGGER.debug(
                                    "  [FK Expansion] [Depth %d] Added FK-referencing table: %s (references %s via FK)",
                                    depth, referencing_table, table_name
                                )
                                # Check limit after increment
                                if expansion_count >= max_expansion:
                                    break
                            else:
                                LOGGER.debug(
                                    "  [FK Expansion] [Depth %d] FK-referencing table '%s' not found in graph",
                                    depth, referencing_table
                                )
                    # Break outer loop if limit reached
            if expansion_count >= max_expansion:
                break
            
            current_level = next_level
            if next_level:
                LOGGER.debug(
                    "  [FK Expansion] [Depth %d] Added %d tables, total expansion: %d/%d",
                    depth, len(next_level), expansion_count, max_expansion
                )
        
        # Final deduplication (safety check - should already be unique, but ensure it)
        seen = set()
        deduplicated = []
        for card in result:
            if card.name not in seen:
                deduplicated.append(card)
                seen.add(card.name)
            else:
                LOGGER.debug("  [FK Expansion] Found duplicate table '%s' during deduplication, removing", card.name)
        
        if len(deduplicated) != len(result):
            LOGGER.warning(
                "  [FK Expansion] Deduplication removed %d duplicate tables",
                len(result) - len(deduplicated)
            )
        
        if len(deduplicated) > len(selected_tables):
            added_count = len(deduplicated) - len(selected_tables)
            LOGGER.info(
                "  [FK Expansion] Multi-hop expansion complete: %d -> %d tables (+%d tables, max_depth=%d, actual_depth=%d)",
                len(selected_tables), len(deduplicated), added_count, max_depth, actual_depth
            )
            LOGGER.debug(
                "  [FK Expansion] Expanded tables: %s",
                [t.name for t in deduplicated if t.name not in {s.name for s in selected_tables}]
            )
        else:
            LOGGER.info(
                "  [FK Expansion] No additional tables needed (all FK-connected tables already selected)"
            )
        
        return deduplicated

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

