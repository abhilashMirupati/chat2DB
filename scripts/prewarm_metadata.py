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

