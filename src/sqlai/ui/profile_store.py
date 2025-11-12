"""Persistence helpers for connection profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sqlai.config import load_app_config


def _store_path() -> Path:
    config = load_app_config()
    return config.cache_dir / "connections.json"


def load_profiles() -> Dict[str, Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    path = _store_path()
    path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def upsert_profile(name: str, profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    profiles = load_profiles()
    profiles[name] = profile
    save_profiles(profiles)
    return profiles


def delete_profile(name: str) -> Dict[str, Dict[str, Any]]:
    profiles = load_profiles()
    if name in profiles:
        del profiles[name]
        save_profiles(profiles)
    return profiles

