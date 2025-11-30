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

import os

# Disable all analytics BEFORE any other imports (prevents PostHog SSL errors)
os.environ["LANGCHAIN_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LLAMA_INDEX_ANALYTICS_ENABLED"] = "false"
os.environ["POSTHOG_DISABLE"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["OPENAI_TELEMETRY_OPTOUT"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_TELEMETRY"] = "false"
os.environ["DO_NOT_TRACK"] = "1"

# Hard monkeypatch PostHog so it CANNOT send anything
try:
    import posthog
    posthog.disabled = True
except Exception:
    pass

import warnings

from dotenv import load_dotenv

# Load config early to check telemetry setting
load_dotenv()
from sqlai.config import load_app_config, load_database_config, load_embedding_config, load_vector_store_config
from sqlai.utils.logging import _disable_telemetry

# Disable telemetry if configured (prevents PostHog SSL errors at root cause)
_disable_telemetry()

# Suppress SSL warnings as fallback (in case telemetry still attempts connection)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*certificate.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*posthog.*", category=UserWarning)

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set
from sqlai.database.connectors import create_db_engine, test_connection
from sqlai.database.schema_introspector import introspect_database, TableSummary
from sqlai.services.cache_health import diff_vector_maps, graph_vector_id_map
from sqlai.services.graph_cache import GraphCache
from sqlai.services.metadata_cache import MetadataCache
from sqlai.services.vector_store import VectorStoreManager
from sqlai.semantic.retriever import SemanticRetriever

# Load environment variables from .env file (if present)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Suppress urllib3 and backoff SSL warnings (harmless telemetry connection failures)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("backoff").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Validate cache completeness."""
    parser = argparse.ArgumentParser(
        description="Validate metadata, graph cards, and embeddings cache completeness."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically generate missing embeddings when detected (requires embedding provider configured).",
    )
    args = parser.parse_args()
    
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
    
    # Only exit if database has no tables (nothing to validate)
    # If cache is empty but DB has tables, continue to report missing cache
    if not db_table_names:
        LOGGER.error("  ✗ No tables found in database!")
        LOGGER.error("  Cannot validate cache - database appears to be empty.")
        sys.exit(1)
    
    # If database has tables but cache is empty, this is a validation issue (not an error)
    if not metadata_table_names:
        LOGGER.warning("  ⚠ Metadata cache is empty, but database has %s table(s).", len(db_table_names))
        LOGGER.warning("  Run prewarm_metadata.py to populate the cache.")
    
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
    
    # Always compare DB tables to cache, regardless of cache size
    # Build expected vector map from graph cache, filtered to only DB tables
    expected_vector_map = graph_vector_id_map(graph_cache, schema)
    # Only check tables that exist in the live database (direct comparison)
    expected_vector_map = {
        table: ids
        for table, ids in expected_vector_map.items()
        if table in db_table_names
    }
    
    if not all_cards:
        LOGGER.warning("  ⚠ No graph cards found in cache for database tables")
        LOGGER.warning("  This means embeddings cannot be validated - run prewarm_metadata.py first")
    
    try:
        actual_vector_map = vector_store.list_vectors_by_table(schema)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("  ✗ Failed to read vector store: %s", exc)
        LOGGER.error("  Vector store may not be initialised. Run prewarm_metadata.py to generate embeddings.")
        sys.exit(1)
    
    total_vectors = sum(len(ids) for ids in actual_vector_map.values())
    LOGGER.info("  Found %s embedding(s) in vector store", total_vectors)

    missing_embeddings_map, orphaned_embeddings_map = diff_vector_maps(
        expected_vector_map,
        actual_vector_map,
    )

    if missing_embeddings_map:
        LOGGER.warning(
            "  ⚠ Missing embeddings for %s table(s) (total %s vector(s) absent):",
            len(missing_embeddings_map),
            sum(len(ids) for ids in missing_embeddings_map.values()),
        )
        for table in sorted(missing_embeddings_map.keys())[:10]:
            missing_ids = missing_embeddings_map[table]
            LOGGER.warning("    - %s: %s vector(s) missing", table, len(missing_ids))
            sample_ids = list(sorted(missing_ids))[:5]
            for vector_id in sample_ids:
                LOGGER.warning("      • %s", vector_id)
            if len(missing_ids) > 5:
                LOGGER.warning("      ... and %s more", len(missing_ids) - 5)
        if len(missing_embeddings_map) > 10:
            LOGGER.warning("    ... and %s more tables", len(missing_embeddings_map) - 10)
        
        # Auto-fix if --fix flag is set
        if args.fix:
            LOGGER.info("")
            LOGGER.info("=" * 80)
            LOGGER.info("AUTO-FIX: Generating missing embeddings...")
            LOGGER.info("=" * 80)
            try:
                semantic_retriever = SemanticRetriever(embedding_config, vector_store=vector_store)
                embedder = semantic_retriever.provider
                if not embedder:
                    LOGGER.error(
                        "  ✗ Cannot generate embeddings: embedding provider not configured or unavailable."
                    )
                    LOGGER.error(
                        "  Please ensure SQLAI_EMBED_PROVIDER and SQLAI_EMBED_MODEL are set correctly."
                    )
                    sys.exit(1)
                
                tables_to_fix = sorted(missing_embeddings_map.keys())
                LOGGER.info(
                    "  Generating embeddings for %s table(s): %s",
                    len(tables_to_fix),
                    ", ".join(tables_to_fix[:10]) + (" ..." if len(tables_to_fix) > 10 else ""),
                )
                
                vector_store.refresh_tables(
                    schema,
                    tables_to_fix,
                    graph_cache,
                    embedder,
                )
                
                LOGGER.info("  ✓ Successfully generated embeddings for %s table(s)", len(tables_to_fix))
                
                # Re-check to verify fix
                LOGGER.info("  Verifying embeddings were created...")
                actual_vector_map_after = vector_store.list_vectors_by_table(schema)
                missing_after, _ = diff_vector_maps(expected_vector_map, actual_vector_map_after)
                if missing_after:
                    LOGGER.warning(
                        "  ⚠ Still missing embeddings for %s table(s) after fix attempt.",
                        len(missing_after),
                    )
                    # Update missing_embeddings_map to reflect remaining issues
                    missing_embeddings_map = missing_after
                else:
                    LOGGER.info("  ✓ All missing embeddings have been generated successfully!")
                    # Clear missing_embeddings_map only if ALL embeddings were successfully created
                    missing_embeddings_map = {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("  ✗ Failed to generate embeddings: %s", exc)
                LOGGER.error("  Please check your embedding provider configuration and try again.")
                sys.exit(1)
    else:
        # Only log success if we actually had cards to check
        if expected_vector_map:
            LOGGER.info("  ✓ All graph cards have embeddings")
        elif db_table_names:
            # DB has tables but no graph cards in cache
            LOGGER.warning("  ⚠ No graph cards in cache to validate embeddings for")

    if orphaned_embeddings_map:
        LOGGER.warning(
            "  ⚠ Found %s table(s) with orphaned embeddings (no corresponding graph cards):",
            len(orphaned_embeddings_map),
        )
        for table in sorted(orphaned_embeddings_map.keys())[:5]:
            orphaned_ids = orphaned_embeddings_map[table]
            LOGGER.warning("    - %s: %s vector(s) orphaned", table, len(orphaned_ids))
        if len(orphaned_embeddings_map) > 5:
            LOGGER.warning("    ... and %s more tables", len(orphaned_embeddings_map) - 5)
    
    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("VALIDATION SUMMARY")
    LOGGER.info("=" * 80)
    
    has_issues = bool(
        missing_in_cache
        or missing_graph_tables
        or tables_missing_card_types
        or missing_embeddings_map
    )
    
    if not has_issues:
        LOGGER.info("✓ SUCCESS: All caches are complete and match database!")
        LOGGER.info("  - Database: %s table(s)", len(db_table_names))
        LOGGER.info("  - Metadata cache: %s table(s)", len(metadata_table_names))
        LOGGER.info("  - Graph cards cache: %s table(s) with all card types", len(graph_table_names))
        # Always report vector store status based on actual comparison (DB tables vs cache)
        if expected_vector_map:
            LOGGER.info(
                "  - Vector store: %s embedding(s) for %s card(s) (all types: table, column, relationship)",
                total_vectors,
                sum(len(ids) for ids in expected_vector_map.values()),
            )
        else:
            LOGGER.info("  - Vector store: No graph cards in cache to validate (compare DB tables to cache)")
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
        if missing_embeddings_map:
            LOGGER.warning(
                "  - %s table(s) missing embeddings",
                len(missing_embeddings_map),
            )
        LOGGER.warning("")
        LOGGER.warning("ACTION REQUIRED:")
        LOGGER.warning("  Please run prewarm_metadata.py to fix missing items:")
        LOGGER.warning("    python scripts/prewarm_metadata.py")
        LOGGER.warning("")
        LOGGER.warning("This script only validates and reports - it does not modify or delete any data.")
        sys.exit(1)


if __name__ == "__main__":
    main()

