#!/usr/bin/env python
"""Recover cache data from backup files.

This script can restore table_metadata and graph_cards from backup files
if they exist in the .cache directory.
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

import logging
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from sqlai.config import load_app_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Recover cache from backup files."""
    app_config = load_app_config()
    cache_dir = app_config.cache_dir
    
    LOGGER.info("=" * 80)
    LOGGER.info("CACHE RECOVERY")
    LOGGER.info("=" * 80)
    LOGGER.info("Cache directory: %s", cache_dir)
    LOGGER.info("")
    
    # Check for backup files
    metadata_backup = cache_dir / "table_metadata copy.db"
    graph_backup = cache_dir / "graph_cards copy.db"
    
    recovered = False
    
    # Recover metadata cache
    if metadata_backup.exists():
        metadata_current = cache_dir / "table_metadata.db"
        LOGGER.info("Found metadata backup: %s", metadata_backup)
        
        # Check if backup has data
        import sqlite3
        conn = sqlite3.connect(str(metadata_backup))
        cursor = conn.execute("SELECT COUNT(*) FROM table_metadata")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            LOGGER.info("  Backup contains %s table(s)", count)
            response = input("  Restore from backup? This will overwrite current cache. (y/N): ")
            if response.lower() == 'y':
                if metadata_current.exists():
                    # Create a backup of current before overwriting
                    current_backup = cache_dir / f"table_metadata.db.backup.{int(__import__('time').time())}"
                    shutil.copy2(metadata_current, current_backup)
                    LOGGER.info("  Created backup of current file: %s", current_backup.name)
                
                shutil.copy2(metadata_backup, metadata_current)
                LOGGER.info("  ✓ Restored metadata cache from backup")
                
                # Set cache version to prevent re-migration
                import sqlite3
                conn = sqlite3.connect(str(metadata_current))
                conn.execute(
                    "INSERT OR REPLACE INTO cache_version (key, value) VALUES (?, ?)",
                    ("hash_algorithm_version", "2")
                )
                conn.commit()
                conn.close()
                LOGGER.info("  ✓ Set cache version to prevent re-migration")
                recovered = True
            else:
                LOGGER.info("  Skipped metadata recovery")
        else:
            LOGGER.warning("  Backup file exists but is empty")
    else:
        LOGGER.info("No metadata backup found: %s", metadata_backup)
    
    # Recover graph cache
    if graph_backup.exists():
        graph_current = cache_dir / "graph_cards.db"
        LOGGER.info("")
        LOGGER.info("Found graph cards backup: %s", graph_backup)
        
        import sqlite3
        conn = sqlite3.connect(str(graph_backup))
        cursor = conn.execute("SELECT COUNT(*) FROM graph_cards")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            LOGGER.info("  Backup contains %s graph card(s)", count)
            response = input("  Restore from backup? This will overwrite current cache. (y/N): ")
            if response.lower() == 'y':
                if graph_current.exists():
                    # Create a backup of current before overwriting
                    current_backup = cache_dir / f"graph_cards.db.backup.{int(__import__('time').time())}"
                    shutil.copy2(graph_current, current_backup)
                    LOGGER.info("  Created backup of current file: %s", current_backup.name)
                
                shutil.copy2(graph_backup, graph_current)
                LOGGER.info("  ✓ Restored graph cards cache from backup")
                
                # Set cache version to prevent re-migration
                import sqlite3
                conn = sqlite3.connect(str(graph_current))
                conn.execute(
                    "INSERT OR REPLACE INTO cache_version (key, value) VALUES (?, ?)",
                    ("hash_algorithm_version", "2")
                )
                conn.commit()
                conn.close()
                LOGGER.info("  ✓ Set cache version to prevent re-migration")
                recovered = True
            else:
                LOGGER.info("  Skipped graph cards recovery")
        else:
            LOGGER.warning("  Backup file exists but is empty")
    else:
        LOGGER.info("No graph cards backup found: %s", graph_backup)
    
    LOGGER.info("")
    if recovered:
        LOGGER.info("✓ Recovery complete!")
        LOGGER.info("")
        LOGGER.info("The cache version has been set to prevent re-migration.")
        LOGGER.info("If you need to regenerate hashes, run: python scripts/prewarm_metadata.py")
    else:
        LOGGER.info("No recovery performed.")


if __name__ == "__main__":
    main()

