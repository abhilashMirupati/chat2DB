"""
Logging utilities for the SQLAI application.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional


def configure_logging(level: int = logging.INFO) -> None:
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
    logging.getLogger("backoff").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "sqlai")

