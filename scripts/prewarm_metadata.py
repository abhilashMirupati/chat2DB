#!/usr/bin/env python
"""Pre-compute table metadata (descriptions + samples) before launching the UI."""

from __future__ import annotations

import logging

from sqlai.services.analytics_service import AnalyticsService


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    service = AnalyticsService()
    cache_path = service.app_config.cache_dir / "table_metadata.db"
    logging.info("Metadata cache populated at %s", cache_path)
    # Dispose DB connections to avoid hanging processes
    service.engine.dispose(close=True)


if __name__ == "__main__":
    main()

