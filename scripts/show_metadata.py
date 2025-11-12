#!/usr/bin/env python
"""Print cached table metadata (descriptions + sample values)."""

from __future__ import annotations

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

