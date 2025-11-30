#!/usr/bin/env python
"""Pre-compute table metadata (descriptions + samples) before launching the UI.

Prerequisites
-------------
Set the core environment variables before running this script so the service can
connect to both your database and LLM.

macOS / Linux (bash or zsh):
    export SQLAI_DB_URL="oracle+oracledb://USER:PASSWORD@HOST:1521/?service_name=AGENT_DEMO"
    export SQLAI_DB_SCHEMA="AGENT_DEMO"
    export SQLAI_LLM_PROVIDER="ollama"
    export SQLAI_LLM_MODEL="llama3"
    export SQLAI_LLM_BASE_URL="http://localhost:11434"
    export SQLAI_EMBED_PROVIDER="huggingface"
    export SQLAI_EMBED_MODEL="google/embeddinggemma-300m"
    export SQLAI_EMBED_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    export SQLAI_VECTOR_PROVIDER="chroma"

Windows (PowerShell):
    $env:SQLAI_DB_URL = "oracle+oracledb://USER:PASSWORD@HOST:1521/?service_name=AGENT_DEMO"
    $env:SQLAI_DB_SCHEMA = "AGENT_DEMO"
    $env:SQLAI_LLM_PROVIDER = "ollama"
    $env:SQLAI_LLM_MODEL = "llama3"
    $env:SQLAI_LLM_BASE_URL = "http://localhost:11434"
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
from sqlai.config import load_app_config
from sqlai.services.analytics_service import AnalyticsService
from sqlai.utils.logging import _disable_telemetry

# Disable telemetry if configured (prevents PostHog SSL errors at root cause)
_disable_telemetry()

# Suppress SSL warnings as fallback (in case telemetry still attempts connection)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*certificate.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*posthog.*", category=UserWarning)

import logging


def main() -> None:
    """
    Pre-compute table metadata, graph cards, and embeddings.
    
    This script ALWAYS performs full introspection and validation, regardless of cache state.
    It works correctly in all scenarios:
    - Fresh start (no cache): Creates all caches from scratch
    - Warm start (cache exists): Validates and fixes any missing items
    - Failed start (partial cache): Completes missing items
    - Stopped start (interrupted cache): Validates and fixes inconsistencies
    
    The script will:
    1. Connect to database and introspect all tables
    2. Check metadata cache and generate missing descriptions/samples
    3. Check graph cards cache and generate missing cards
    4. Check embeddings and generate missing vectors
    5. Ensure all tables have complete metadata, graph cards, and embeddings
    """
    logging.basicConfig(level=logging.INFO)
    # Suppress urllib3 and backoff SSL warnings (harmless telemetry connection failures)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("backoff").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    
    logging.info("=" * 80)
    logging.info("PREWARM METADATA - Full introspection and validation")
    logging.info("=" * 80)
    logging.info("This will:")
    logging.info("  1. Connect to database and introspect all tables")
    logging.info("  2. Generate/update metadata (descriptions + samples) for all tables")
    logging.info("  3. Generate/update graph cards (table, column, relationship) for all tables")
    logging.info("  4. Generate/update embeddings for all graph cards")
    logging.info("")
    logging.info("This works in all states: fresh, warm, failed, or stopped cache.")
    logging.info("=" * 80)
    logging.info("")
    
    # FORCE full prewarm: always introspect DB, validate, and fix missing items
    # This ensures all tables have metadata, graph cards, and embeddings
    # regardless of cache state (fresh, warm, failed, stopped)
    service = AnalyticsService(skip_prewarm_if_cached=False)
    
    cache_path = service.app_config.cache_dir / "table_metadata.db"
    graph_cache_path = service.app_config.cache_dir / "graph_cards.db"
    logging.info("")
    logging.info("=" * 80)
    logging.info("PREWARM COMPLETE")
    logging.info("=" * 80)
    logging.info("Metadata cache: %s", cache_path)
    logging.info("Graph cards cache: %s", graph_cache_path)
    logging.info("")
    logging.info("All tables now have:")
    logging.info("  ✓ Metadata (descriptions + samples)")
    logging.info("  ✓ Graph cards (table, column, relationship)")
    logging.info("  ✓ Embeddings (vector store)")
    logging.info("")
    logging.info("You can now run validate_cache.py to verify completeness.")
    logging.info("=" * 80)
    
    # Dispose DB connections to avoid hanging processes
    service.engine.dispose(close=True)


if __name__ == "__main__":
    main()

