#!/usr/bin/env python
"""Migrate existing graph cards to include table descriptions in embedding text.

This script:
1. Reads all table cards from graph cache
2. Fetches descriptions from metadata cache
3. Updates graph card text to include descriptions
4. Re-embeds updated cards in ChromaDB

This is a one-time migration to improve semantic search accuracy without
rebuilding all graph cards from scratch.

Prerequisites
-------------
Set embedding configuration (same as prewarm_metadata.py):

macOS / Linux (bash or zsh):
    export SQLAI_EMBED_PROVIDER="huggingface"
    export SQLAI_EMBED_MODEL="google/embeddinggemma-300m"
    export SQLAI_EMBED_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    export SQLAI_VECTOR_PROVIDER="chroma"

Windows (PowerShell):
    $env:SQLAI_EMBED_PROVIDER = "huggingface"
    $env:SQLAI_EMBED_MODEL = "google/embeddinggemma-300m"
    $env:SQLAI_EMBED_API_KEY = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    $env:SQLAI_VECTOR_PROVIDER = "chroma"

Alternatively, add the same key/value pairs to the project `.env` file.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Set

from dotenv import load_dotenv

# Load config early to check telemetry setting
load_dotenv()
from sqlai.config import load_app_config, load_embedding_config, load_vector_store_config
from sqlai.services.graph_cache import GraphCache, GraphCardRecord
from sqlai.services.metadata_cache import MetadataCache
from sqlai.services.vector_store import VectorStoreManager
from sqlai.semantic.retriever import SemanticRetriever
from sqlai.utils.logging import _disable_telemetry

# Disable telemetry if configured (prevents PostHog SSL errors at root cause)
_disable_telemetry()

# Suppress SSL warnings as fallback (in case telemetry still attempts connection)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*certificate.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*posthog.*", category=UserWarning)

import logging

logging.basicConfig(level=logging.INFO)
# Suppress urllib3 and backoff SSL warnings (harmless telemetry connection failures)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("backoff").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)


def _normalize_schema_name(schema: str) -> str:
    """Normalize schema name to lowercase for consistent cache lookups."""
    if not schema:
        return schema
    return schema.lower().strip()


def main() -> None:
    """
    Migrate graph cards to include table descriptions in embedding text.
    
    This script:
    1. Iterates through all graph cards in cache
    2. For table cards, fetches description from metadata cache
    3. Updates graph card text to prepend description
    4. Updates graph cache with new text
    5. Re-embeds updated cards in ChromaDB
    """
    LOGGER.info("=" * 80)
    LOGGER.info("MIGRATE GRAPH CARDS WITH DESCRIPTIONS")
    LOGGER.info("=" * 80)
    LOGGER.info("This will:")
    LOGGER.info("  1. Read all table cards from graph cache")
    LOGGER.info("  2. Fetch descriptions from metadata cache")
    LOGGER.info("  3. Update graph card text to include descriptions")
    LOGGER.info("  4. Re-embed updated cards in ChromaDB")
    LOGGER.info("")
    LOGGER.info("This is a one-time migration to improve semantic search accuracy.")
    LOGGER.info("=" * 80)
    LOGGER.info("")

    # Load configuration
    app_config = load_app_config()
    embedding_config = load_embedding_config()
    vector_config = load_vector_store_config()

    # Initialize caches and stores
    graph_cache = GraphCache(app_config.cache_dir / "graph_cards.db")
    metadata_cache = MetadataCache(app_config.cache_dir / "table_metadata.db")
    vector_store = VectorStoreManager(
        vector_config,
        app_config.cache_dir,
        embedding_config,
    )
    semantic_retriever = SemanticRetriever(
        embedding_config,
        vector_store=vector_store,
    )
    embedder = semantic_retriever.provider

    if not embedder:
        LOGGER.error("Embedding provider is not configured. Please set SQLAI_EMBED_PROVIDER and related config.")
        return

    LOGGER.info("Collecting all table cards from graph cache...")
    
    # Group table cards by schema and table name
    # schema -> table -> list of cards (table, column, relationship)
    schema_tables: Dict[str, Dict[str, List[GraphCardRecord]]] = defaultdict(lambda: defaultdict(list))
    table_cards_to_update: List[GraphCardRecord] = []
    
    total_cards = 0
    for card in graph_cache.iter_cards():
        total_cards += 1
        # Use normalized schema for consistent grouping
        schema_key = _normalize_schema_name(card.schema) if card.schema else card.schema
        schema_tables[schema_key][card.table].append(card)
        if card.card_type == "table":
            table_cards_to_update.append(card)
    
    LOGGER.info("Found %s total graph cards across all schemas", total_cards)
    LOGGER.info("Found %s table cards to potentially update", len(table_cards_to_update))
    LOGGER.info("")

    # Process each unique (schema, table) pair only once
    updated_count = 0
    skipped_count = 0
    error_count = 0
    tables_to_reembed: Dict[str, Set[str]] = defaultdict(set)  # schema -> set of table names
    processed_tables: Set[tuple[str, str]] = set()  # Track (schema, table) pairs we've processed
    
    for card in table_cards_to_update:
        schema_normalized = _normalize_schema_name(card.schema)
        table_name = card.table
        table_key = (schema_normalized, table_name)
        
        # Skip if we've already processed this table
        if table_key in processed_tables:
            continue
        
        processed_tables.add(table_key)
        
        try:
            # Fetch description from metadata cache
            cached_meta = metadata_cache.fetch(schema_normalized, table_name)
            description = cached_meta.get("description") if cached_meta else None
            
            if not description:
                LOGGER.debug(
                    "Skipping table '%s.%s' - no description in metadata cache",
                    card.schema,
                    table_name,
                )
                skipped_count += 1
                continue
            
            # Get all cards for this table (use normalized schema for lookup)
            schema_key_for_lookup = _normalize_schema_name(card.schema) if card.schema else card.schema
            all_cards_for_table = schema_tables[schema_key_for_lookup][table_name]
            
            # Find the table card (should be only one, but handle multiple)
            table_card = next((c for c in all_cards_for_table if c.card_type == "table"), None)
            if not table_card:
                LOGGER.warning("No table card found for '%s.%s'", card.schema, table_name)
                skipped_count += 1
                continue
            
            # Check if description is already in the text (avoid duplicate)
            # Simple check: if text already starts with description (first 50 chars), skip
            description_prefix = description[:50].strip()
            if table_card.text.startswith(description_prefix):
                LOGGER.debug(
                    "Skipping table '%s.%s' - description already in text",
                    card.schema,
                    table_name,
                )
                skipped_count += 1
                continue
            
            # Update text to include description
            new_text = f"{description}\n{table_card.text}"
            
            # Update metadata to include description
            new_metadata = dict(table_card.metadata)
            new_metadata["description"] = description
            
            # Create updated table card record
            updated_table_card = GraphCardRecord(
                schema=table_card.schema,
                table=table_card.table,
                card_type=table_card.card_type,
                identifier=table_card.identifier,
                schema_hash=table_card.schema_hash,
                text=new_text,
                metadata=new_metadata,
            )
            
            # Replace table card in the list (keep all other cards unchanged)
            updated_cards = []
            for c in all_cards_for_table:
                if c.card_type == "table" and c.identifier == table_card.identifier:
                    updated_cards.append(updated_table_card)
                else:
                    updated_cards.append(c)
            
            # Update graph cache by directly updating the table card's text
            # This avoids the DELETE/INSERT issue with schema case mismatches
            schema_normalized_for_db = _normalize_schema_name(table_card.schema) if table_card.schema else table_card.schema
            with graph_cache._lock:
                # Update the table card's text and metadata directly
                graph_cache.conn.execute(
                    """
                    UPDATE graph_cards
                    SET text = ?,
                        metadata_json = ?,
                        updated_at = ?
                    WHERE schema_name = ? 
                      AND table_name = ?
                      AND card_type = 'table'
                      AND identifier = ?
                    """,
                    (
                        new_text,
                        json.dumps(new_metadata),
                        time.time(),
                        schema_normalized_for_db,
                        table_name,
                        table_card.identifier,
                    ),
                )
                graph_cache.conn.commit()
            
            # Track for re-embedding
            tables_to_reembed[schema_normalized].add(table_name)
            
            updated_count += 1
            if updated_count % 10 == 0:
                LOGGER.info(
                    "Progress: Updated %s/%s tables...",
                    updated_count,
                    len(processed_tables),
                )
                
        except Exception as exc:
            LOGGER.error(
                "Error updating table '%s.%s': %s",
                card.schema,
                table_name,
                exc,
            )
            error_count += 1
    
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("GRAPH CARD UPDATE SUMMARY")
    LOGGER.info("=" * 80)
    LOGGER.info("Total tables processed: %s", len(processed_tables))
    LOGGER.info("  ✓ Updated: %s", updated_count)
    LOGGER.info("  ⊘ Skipped (no description or already updated): %s", skipped_count)
    if error_count > 0:
        LOGGER.info("  ✗ Errors: %s", error_count)
    LOGGER.info("=" * 80)
    LOGGER.info("")
    
    # Re-embed updated tables in ChromaDB
    if tables_to_reembed:
        LOGGER.info("Re-embedding updated table cards in ChromaDB...")
        LOGGER.info("")
        
        total_tables_to_reembed = sum(len(tables) for tables in tables_to_reembed.values())
        reembed_count = 0
        
        for schema_normalized, table_names in tables_to_reembed.items():
            for table_name in sorted(table_names):
                try:
                    vector_store.refresh_tables(
                        schema_normalized,
                        [table_name],
                        graph_cache,
                        embedder,
                    )
                    reembed_count += 1
                    if reembed_count % 10 == 0:
                        LOGGER.info(
                            "Re-embedding progress: %s/%s tables...",
                            reembed_count,
                            total_tables_to_reembed,
                        )
                except Exception as exc:
                    LOGGER.error(
                        "Error re-embedding table '%s.%s': %s",
                        schema_normalized,
                        table_name,
                        exc,
                    )
        
        LOGGER.info("")
        LOGGER.info("=" * 80)
        LOGGER.info("RE-EMBEDDING SUMMARY")
        LOGGER.info("=" * 80)
        LOGGER.info("Total tables re-embedded: %s/%s", reembed_count, total_tables_to_reembed)
        LOGGER.info("=" * 80)
    else:
        LOGGER.info("No tables to re-embed (all were skipped).")
    
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("MIGRATION COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Graph cards have been updated with descriptions and re-embedded.")
    LOGGER.info("Semantic search accuracy should now be improved.")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

