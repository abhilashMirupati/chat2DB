"""
Logging utilities for the SQLAI application.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

from sqlai.config import load_app_config


def _disable_telemetry() -> None:
    """Disable telemetry to prevent PostHog SSL errors. Called based on AppConfig."""
    # Always disable telemetry (regardless of config) to prevent SSL errors
    # Set all telemetry-related environment variables
    os.environ.setdefault("LANGCHAIN_TELEMETRY", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGCHAIN_API_KEY", "")
    os.environ.setdefault("LLAMA_INDEX_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("POSTHOG_DISABLE", "true")
    os.environ.setdefault("POSTHOG_DISABLED", "true")
    os.environ.setdefault("OPENAI_TELEMETRY_OPTOUT", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_TELEMETRY", "false")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    
    # Hard monkeypatch PostHog so it CANNOT send anything
    try:
        import posthog
        posthog.disabled = True
    except Exception:
        pass


def configure_logging(level: int = logging.INFO) -> None:
    # Disable telemetry before configuring logging (if disabled in config)
    _disable_telemetry()
    env_level = os.getenv("SQLAI_LOG_LEVEL")
    if env_level:
        resolved = getattr(logging, env_level.upper(), None)
        if isinstance(resolved, int):
            level = resolved
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True,
    )
    # Suppress urllib3 and backoff SSL warnings (harmless telemetry connection failures)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("backoff").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
    warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*certificate.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*posthog.*", category=UserWarning)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "sqlai")

