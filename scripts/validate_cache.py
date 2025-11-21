#!/usr/bin/env python
"""Validate that all metadata, graph cards, and embeddings are complete.

This script:
1. Connects to the database and introspects the current schema
2. Compares database tables with cached metadata to find missing/extra tables
3. Validates that all tables have proper graph cards (table, column, relationship/FK)
4. Validates that all graph cards have corresponding embeddings in the vector store

Note: This validates ALL card types for each table:
- Table cards (one per table)
- Column cards (one per column)
- Relationship cards (one per foreign key relationship)

Prerequisites
-------------
Set the core environment variables before running this script.

macOS / Linux (bash or zsh):
    export SQLAI_DB_URL="oracle+oracledb://USER:PASSWORD@HOST:1521/?service_name=AGENT_DEMO"
    export SQLAI_DB_SCHEMA="AGENT_DEMO"
    export SQLAI_EMBED_PROVIDER="huggingface"
    export SQLAI_EMBED_MODEL="google/embeddinggemma-300m"
    export SQLAI_EMBED_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    export SQLAI_VECTOR_PROVIDER="chroma"

Windows (PowerShell):
    $env:SQLAI_DB_URL = "oracle+oracledb://USER:PASSWORD@HOST:1521/?service_name=AGENT_DEMO"
    $env:SQLAI_DB_SCHEMA = "AGENT_DEMO"
    $env:SQLAI_EMBED_PROVIDER = "huggingface"
    $env:SQLAI_EMBED_MODEL = "google/embeddinggemma-300m"
    $env:SQLAI_EMBED_API_KEY = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    $env:SQLAI_VECTOR_PROVIDER = "chroma"

Alternatively, add the same key/value pairs to the project `.env` file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

from sqlai.config import load_app_config, load_database_config, load_embedding_config, load_vector_store_config
from sqlai.database.connectors import create_db_engine, test_connection
from sqlai.database.schema_introspector import introspect_database, TableSummary
from sqlai.services.graph_cache import GraphCache
from sqlai.services.metadata_cache import MetadataCache
from sqlai.services.vector_store import VectorStoreManager

# Load environment variables from .env file (if present)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Validate cache completeness."""
    app_config = load_app_config()
    db_config = load_database_config()
    embedding_config = load_embedding_config()
    vector_config = load_vector_store_config()
    
    schema = db_config.schema or "(default)"
    
    LOGGER.info("=" * 80)
    LOGGER.info("CACHE VALIDATION REPORT")
    LOGGER.info("=" * 80)
    LOGGER.info("Schema: %s", schema)
    LOGGER.info("")
    
    # Step 0: Connect to database and introspect
    LOGGER.info("Step 0: Connecting to database and introspecting schema...")
    try:
        engine = create_db_engine(db_config)
        connectivity_error = test_connection(engine)
        if connectivity_error:
            LOGGER.error("  ✗ Database connection failed: %s", connectivity_error)
            sys.exit(1)
        LOGGER.info("  ✓ Connected to database")
        
        # Introspect database to get actual tables
        schema_summaries = introspect_database(
            engine,
            schema=db_config.schema,
            include_system_tables=db_config.include_system_tables,
        )
        db_table_names = {summary.name for summary in schema_summaries}
        LOGGER.info("  Found %s table(s) in database", len(db_table_names))
        
        # Create a mapping of table name to TableSummary for later use
        db_tables_map: Dict[str, TableSummary] = {summary.name: summary for summary in schema_summaries}
        
        # Dispose engine
        engine.dispose(close=True)
    except Exception as exc:
        LOGGER.error("  ✗ Failed to connect to database: %s", exc)
        LOGGER.error("  Please check your database configuration.")
        sys.exit(1)
    
    # Initialize caches
    metadata_cache = MetadataCache(app_config.cache_dir / "table_metadata.db")
    graph_cache = GraphCache(app_config.cache_dir / "graph_cards.db")
    vector_store = VectorStoreManager(
        vector_config,
        app_config.cache_dir,
        embedding_config,
    )
    
    # Step 1: Compare database tables with metadata cache
    LOGGER.info("")
    LOGGER.info("Step 1: Comparing database with metadata cache...")
    metadata_tables = metadata_cache.fetch_schema(schema)
    metadata_table_names = set(metadata_tables.keys())
    LOGGER.info("  Found %s table(s) in metadata cache", len(metadata_table_names))
    
    # Find tables in database but not in cache
    missing_in_cache = db_table_names - metadata_table_names
    if missing_in_cache:
        LOGGER.warning("  ⚠ %s table(s) in database but NOT in metadata cache:", len(missing_in_cache))
        for table in sorted(missing_in_cache)[:20]:
            summary = db_tables_map[table]
            column_count = len(summary.columns)
            fk_count = len(summary.foreign_keys)
            LOGGER.warning("    - %s (%s columns, %s FKs)", table, column_count, fk_count)
        if len(missing_in_cache) > 20:
            LOGGER.warning("    ... and %s more", len(missing_in_cache) - 20)
    else:
        LOGGER.info("  ✓ All database tables are in metadata cache")
    
    # Find tables in cache but not in database (orphaned)
    orphaned_in_cache = metadata_table_names - db_table_names
    if orphaned_in_cache:
        LOGGER.warning("  ⚠ %s table(s) in metadata cache but NOT in database (orphaned):", len(orphaned_in_cache))
        for table in sorted(orphaned_in_cache)[:20]:
            LOGGER.warning("    - %s", table)
        if len(orphaned_in_cache) > 20:
            LOGGER.warning("    ... and %s more", len(orphaned_in_cache) - 20)
    
    if not metadata_table_names and not db_table_names:
        LOGGER.error("  ✗ No tables found in database or cache!")
        LOGGER.error("  Please run prewarm_metadata.py first to populate the cache.")
        sys.exit(1)
    
    # Step 2: Check which tables have graph cards and verify card types
    LOGGER.info("")
    LOGGER.info("Step 2: Checking graph cards cache...")
    graph_table_names = set(graph_cache.list_tables(schema))
    LOGGER.info("  Found %s table(s) in graph cards cache", len(graph_table_names))
    
    # Check against database tables (what should be there)
    missing_graph_tables = db_table_names - graph_table_names
    if missing_graph_tables:
        LOGGER.warning("  ⚠ Missing graph cards for %s database table(s):", len(missing_graph_tables))
        for table in sorted(missing_graph_tables)[:20]:
            LOGGER.warning("    - %s", table)
        if len(missing_graph_tables) > 20:
            LOGGER.warning("    ... and %s more", len(missing_graph_tables) - 20)
    else:
        LOGGER.info("  ✓ All database tables have graph cards")
    
    # Check each table has all required card types (compare with database structure)
    tables_missing_card_types: Dict[str, Dict[str, int]] = {}
    for table_name in db_table_names & graph_table_names:  # Only check tables that exist in both
        if table_name not in db_tables_map:
            continue
        summary = db_tables_map[table_name]
        cards = graph_cache.get_cards_for_table(schema, table_name)
        card_types = {card.card_type for card in cards}
        
        # Count cards by type
        card_counts = {"table": 0, "column": 0, "relationship": 0}
        for card in cards:
            card_counts[card.card_type] = card_counts.get(card.card_type, 0) + 1
        
        # Check what's missing
        missing_info: Dict[str, int] = {}
        if "table" not in card_types:
            missing_info["table"] = 1
        expected_columns = len(summary.columns)
        if card_counts["column"] < expected_columns:
            missing_info["column"] = expected_columns - card_counts["column"]
        expected_relationships = len(summary.foreign_keys)
        if card_counts["relationship"] < expected_relationships:
            missing_info["relationship"] = expected_relationships - card_counts["relationship"]
        
        if missing_info:
            tables_missing_card_types[table_name] = missing_info
    
    if tables_missing_card_types:
        LOGGER.warning("  ⚠ Some tables are missing required card types:")
        for table, missing in sorted(tables_missing_card_types.items())[:10]:
            missing_str = ", ".join(f"{k}: {v} missing" for k, v in missing.items())
            LOGGER.warning("    - %s: %s", table, missing_str)
        if len(tables_missing_card_types) > 10:
            LOGGER.warning("    ... and %s more tables", len(tables_missing_card_types) - 10)
    else:
        LOGGER.info("  ✓ All tables have complete card types")
    
    # Step 3: Check embeddings for all graph cards
    LOGGER.info("")
    LOGGER.info("Step 3: Checking vector store embeddings...")
    
    # Get all graph cards (table, column, and relationship cards) for database tables
    all_cards: List[Dict[str, str]] = []
    card_counts_by_type: Dict[str, int] = {"table": 0, "column": 0, "relationship": 0}
    for table_name in db_table_names & graph_table_names:  # Only check tables that exist in database
        cards = graph_cache.get_cards_for_table(schema, table_name)
        for card in cards:
            all_cards.append({
                "table": table_name,
                "card_type": card.card_type,
                "identifier": card.identifier,
                "vector_id": card.vector_id,
            })
            card_counts_by_type[card.card_type] = card_counts_by_type.get(card.card_type, 0) + 1
    
    LOGGER.info("  Found %s graph card(s) total", len(all_cards))
    LOGGER.info("    - Table cards: %s", card_counts_by_type.get("table", 0))
    LOGGER.info("    - Column cards: %s", card_counts_by_type.get("column", 0))
    LOGGER.info("    - Relationship/FK cards: %s", card_counts_by_type.get("relationship", 0))
    
    vector_ids_in_store: Set[str] = set()
    missing_embeddings: Set[str] = set()
    card_vector_ids: Set[str] = set()
    
    if not all_cards:
        LOGGER.warning("  ⚠ No graph cards found to check embeddings for")
    else:
        # Get all vector IDs from cards
        card_vector_ids = {card["vector_id"] for card in all_cards}
        
        # Check if vector store is ready
        try:
            collection = vector_store._ensure_collection()
        except Exception as exc:
            LOGGER.error("  ✗ Failed to access vector store: %s", exc)
            LOGGER.error("  Vector store may not be initialized or embeddings may not be generated.")
            LOGGER.error("  Please run prewarm_metadata.py to generate embeddings.")
            sys.exit(1)
        
        # Get all vector IDs from the store
        try:
            # Query all embeddings (with a large limit)
            all_vectors = collection.get(limit=100000)  # Large limit to get all
            vector_ids_in_store = set(all_vectors.get("ids", []))
            LOGGER.info("  Found %s embedding(s) in vector store", len(vector_ids_in_store))
        except Exception as exc:
            LOGGER.error("  ✗ Failed to query vector store: %s", exc)
            sys.exit(1)
        
        # Check which cards are missing embeddings
        missing_embeddings = card_vector_ids - vector_ids_in_store
        
        if missing_embeddings:
            # Group by table
            missing_by_table: Dict[str, List[str]] = {}
            for card in all_cards:
                if card["vector_id"] in missing_embeddings:
                    table = card["table"]
                    if table not in missing_by_table:
                        missing_by_table[table] = []
                    missing_by_table[table].append(
                        f"{card['card_type']}:{card['identifier']}"
                    )
            
            LOGGER.warning("  ⚠ Missing embeddings for %s graph card(s) across %s table(s):", 
                          len(missing_embeddings), len(missing_by_table))
            for table in sorted(missing_by_table.keys())[:10]:
                cards_list = missing_by_table[table]
                LOGGER.warning("    - %s: %s card(s) missing", table, len(cards_list))
                if len(cards_list) <= 5:
                    for card_info in cards_list:
                        LOGGER.warning("      • %s", card_info)
            if len(missing_by_table) > 10:
                LOGGER.warning("    ... and %s more tables", len(missing_by_table) - 10)
        else:
            LOGGER.info("  ✓ All graph cards have embeddings")
        
        # Check for orphaned embeddings (embeddings without corresponding graph cards)
        orphaned_embeddings = vector_ids_in_store - card_vector_ids
        if orphaned_embeddings:
            LOGGER.warning("  ⚠ Found %s orphaned embedding(s) (no corresponding graph card)", 
                          len(orphaned_embeddings))
            # Sample a few orphaned IDs
            sample_orphaned = list(orphaned_embeddings)[:5]
            for vector_id in sample_orphaned:
                LOGGER.warning("    - %s", vector_id)
            if len(orphaned_embeddings) > 5:
                LOGGER.warning("    ... and %s more", len(orphaned_embeddings) - 5)
    
    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("VALIDATION SUMMARY")
    LOGGER.info("=" * 80)
    
    has_issues = bool(
        missing_in_cache or 
        missing_graph_tables or 
        tables_missing_card_types or 
        (all_cards and missing_embeddings)
    )
    
    if not has_issues:
        LOGGER.info("✓ SUCCESS: All caches are complete and match database!")
        LOGGER.info("  - Database: %s table(s)", len(db_table_names))
        LOGGER.info("  - Metadata cache: %s table(s)", len(metadata_table_names))
        LOGGER.info("  - Graph cards cache: %s table(s) with all card types", len(graph_table_names))
        if all_cards:
            LOGGER.info("  - Vector store: %s embedding(s) for %s card(s) (all types: table, column, relationship)", 
                       len(vector_ids_in_store), len(all_cards))
        sys.exit(0)
    else:
        LOGGER.warning("⚠ ISSUES FOUND:")
        if missing_in_cache:
            LOGGER.warning("  - %s table(s) in database but missing from metadata cache", len(missing_in_cache))
        if orphaned_in_cache:
            LOGGER.warning("  - %s orphaned table(s) in cache (not in database)", len(orphaned_in_cache))
        if missing_graph_tables:
            LOGGER.warning("  - %s table(s) missing graph cards entirely", len(missing_graph_tables))
        if tables_missing_card_types:
            LOGGER.warning("  - %s table(s) missing required card types", len(tables_missing_card_types))
        if all_cards and missing_embeddings:
            LOGGER.warning("  - %s graph card(s) missing embeddings", len(missing_embeddings))
        LOGGER.warning("")
        LOGGER.warning("ACTION REQUIRED:")
        LOGGER.warning("  Please run prewarm_metadata.py to fix missing items:")
        LOGGER.warning("    python scripts/prewarm_metadata.py")
        LOGGER.warning("")
        LOGGER.warning("This script only validates and reports - it does not modify or delete any data.")
        sys.exit(1)


if __name__ == "__main__":
    main()

