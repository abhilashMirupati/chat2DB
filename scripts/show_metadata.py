#!/usr/bin/env python
"""Print cached table metadata (descriptions + sample values)."""

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

import json
from datetime import datetime

from sqlai.config import load_app_config
from sqlai.services.metadata_cache import MetadataCache


def main() -> None:
    app_config = load_app_config()
    cache_path = app_config.cache_dir / "table_metadata.db"
    cache = MetadataCache(cache_path)

    rows = list(cache.iter_metadata())
    if not rows:
        print(f"No cached table metadata found in {cache_path}")
        return

    for row in rows:
        updated_at = datetime.fromtimestamp(row["updated_at"]).isoformat()
        header = f"{row['schema_name']}.{row['table_name']}  (hash={row['schema_hash']}, updated={updated_at})"
        print(header)
        print("-" * len(header))
        description = row.get("description") or "(no description cached)"
        print(description)
        samples = row.get("samples") or {}
        if samples:
            print("Sample values:")
            for column, values in samples.items():
                print(f"  - {column}: {json.dumps(values)}")
        else:
            print("Sample values: (none cached)")
        print()


if __name__ == "__main__":
    main()

