"""
Logging utilities for the SQLAI application.
"""

from __future__ import annotations

import logging
import os
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


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "sqlai")

