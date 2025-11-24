"""Streamlit UI for the SQLAI agent."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

# Load config early to check telemetry setting
load_dotenv()
from sqlai.config import DatabaseConfig, LLMConfig, EmbeddingConfig, load_app_config
from sqlai.utils.logging import _disable_telemetry

# Disable telemetry if configured (prevents PostHog SSL errors)
_disable_telemetry()

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlai.database.connectors import create_db_engine, test_connection, list_schemas
from sqlai.database.schema_introspector import _format_fk_detail
from sqlai.services.analytics_service import AnalyticsService
from sqlai.services.visualization import build_chart
from sqlai.ui.profile_store import delete_profile, load_profiles, upsert_profile
from sqlai.utils.logging import configure_logging, get_logger

DB_TYPE_TEMPLATES: Dict[str, str] = {
    "Oracle": "oracle+oracledb://user:password@host:1521/?service_name=ORCLPDB1",
    "PostgreSQL": "postgresql+psycopg2://user:password@host:5432/database",
    "MySQL": "mysql+pymysql://user:password@host:3306/database",
    "SQL Server": "mssql+pyodbc://user:password@host:1433/database?driver=ODBC+Driver+18+for+SQL+Server",
    "SQLite": "sqlite:///path/to/database.db",
}

configure_logging()

st.set_page_config(page_title="SQLAI: Natural Language Analytics", page_icon="💡", layout="wide")

LOGGER = get_logger(__name__)
APP_CONFIG = load_app_config()
LAST_SESSION_CONFIG_PATH = APP_CONFIG.cache_dir / "ui_last_session.json"
PERSISTED_SESSION_KEYS = [
    "active_profile_value",
    "profile_name",
    "db_type",
    "db_url",
    "db_schema",
    "oracle_thick_mode",
    "oracle_lib_dir",
    "oracle_config_dir",
    "sample_limit",
    "include_system_tables",
    "schema_select_option",
    "llm_provider",
    "llm_model",
    "llm_base_url",
    "llm_api_key",
    "temperature",
    "max_tokens",
    "embedding_provider",
    "embedding_model",
    "embedding_base_url",
    "embedding_api_key",
]

st.markdown(
    """
    <style>
    section.main > div.block-container {
        padding-bottom: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _resolve_brand_logo() -> Optional[str]:
    if APP_CONFIG.brand_logo_path and APP_CONFIG.brand_logo_path.exists():
        return str(APP_CONFIG.brand_logo_path)
    assets_dir = APP_CONFIG.project_root / "assets"
    if assets_dir.exists():
        for name in ("logo.png", "logo.jpg", "logo.jpeg", "logo.svg", "logo.webp"):
            candidate = assets_dir / name
            if candidate.exists():
                return str(candidate)
    return None


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    exp_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(exp_rerun_fn):
        exp_rerun_fn()
        return
    raise RuntimeError("Current Streamlit version does not support rerun APIs.")


def _load_last_session_config_if_needed() -> None:
    if st.session_state.get("_last_session_loaded"):
        return
    st.session_state["_last_session_loaded"] = True
    if not LAST_SESSION_CONFIG_PATH.exists():
        return
    try:
        with open(LAST_SESSION_CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to load last session config: %s", exc)
        return
    for key in PERSISTED_SESSION_KEYS:
        if key in payload and payload[key] is not None:
            st.session_state[key] = payload[key]


def _persist_last_session_config() -> None:
    try:
        LAST_SESSION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: st.session_state.get(key) for key in PERSISTED_SESSION_KEYS}
        with open(LAST_SESSION_CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to persist last session config: %s", exc)


def _clear_persisted_session_config() -> None:
    try:
        if LAST_SESSION_CONFIG_PATH.exists():
            LAST_SESSION_CONFIG_PATH.unlink()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to remove persisted session config: %s", exc)

def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "analytics_service": None,
        "saved_profiles": load_profiles(),
        "active_profile_value": "(New configuration)",
        "profile_name": "",
        "db_type": "Oracle",
        "db_url": "",
        "db_schema": "",
        "oracle_thick_mode": False,
        "oracle_lib_dir": "",
        "oracle_config_dir": "",
        "sample_limit": 100,
        "include_system_tables": False,
        "llm_provider": "ollama",
        "llm_model": "llama3",
        "llm_base_url": "",
        "llm_api_key": "",
        "temperature": 0.2,
        "max_tokens": 1024,
        "connection_status": None,
        "llm_status": None,
        "service_initialised": False,
        "available_schemas": [],
        "schema_select_option": "(current schema)",
        "embedding_provider": "huggingface",
        "embedding_model": "google/embeddinggemma-300m",
        "embedding_base_url": "",
        "embedding_api_key": "",
        "embedding_status": None,
        "current_question": "",
        "saved_queries": [],
        "saved_query_result": None,
        "_last_session_loaded": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    _load_last_session_config_if_needed()


def init_service() -> AnalyticsService | None:
    return st.session_state.get("analytics_service")


def set_service(service: AnalyticsService) -> None:
    st.session_state["analytics_service"] = service


def clear_service() -> None:
    st.session_state["analytics_service"] = None


def _update_inputs_from_profile(profile_name: str) -> None:
    profile = st.session_state["saved_profiles"].get(profile_name)
    if not profile:
        return
    st.session_state.update(
        {
            "profile_name": profile_name,
            "db_type": profile.get("db_type", "Oracle"),
            "db_url": profile.get("db_url", ""),
            "db_schema": profile.get("schema", ""),
            "oracle_thick_mode": profile.get("thick_mode", False),
            "oracle_lib_dir": profile.get("oracle_lib_dir", ""),
            "oracle_config_dir": profile.get("oracle_config_dir", ""),
            "sample_limit": profile.get("sample_limit", 100),
            "llm_provider": profile.get("provider", "ollama"),
            "llm_model": profile.get("model", "llama3"),
            "llm_base_url": profile.get("base_url", ""),
            "llm_api_key": profile.get("api_key", ""),
            "temperature": profile.get("temperature", 0.2),
            "max_tokens": profile.get("max_tokens", 1024),
            "connection_status": None,
            "llm_status": None,
            "service_initialised": False,
            "include_system_tables": profile.get("include_system_tables", False),
            "available_schemas": [],
            "schema_select_option": "(current schema)",
            "embedding_provider": (
                profile.get("embedding_provider") if profile.get("embedding_provider") not in {"", "none", None} else "huggingface"
            ),
            "embedding_model": profile.get("embedding_model", "google/embeddinggemma-300m"),
            "embedding_base_url": profile.get("embedding_base_url", ""),
            "embedding_api_key": profile.get("embedding_api_key", ""),
            "embedding_status": None,
        }
    )


def _render_status(status: Tuple[str, str] | None, placeholder) -> None:
    placeholder.empty()
    if not status:
        return
    level, message = status
    if level == "success":
        placeholder.success(message)
    elif level == "warning":
        placeholder.warning(message)
    else:
        placeholder.error(message)


def _test_connection(db_config: DatabaseConfig) -> Tuple[str, str]:
    engine = create_db_engine(db_config)
    try:
        error = test_connection(engine)
        if error:
            return "error", f"Connection failed: {error}"
        return "success", "Connection successful."
    finally:
        engine.dispose()

def _fetch_schema_options(db_config: DatabaseConfig) -> List[str]:
    engine = create_db_engine(db_config)
    try:
        return list_schemas(engine)
    finally:
        engine.dispose()

def _validate_db_inputs(db_url: str) -> Optional[str]:
    if not db_url or not db_url.strip():
        return "Database URL is required."
    return None


def _validate_llm_inputs(
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> Optional[str]:
    if not model or not model.strip():
        return "Model name is required."
    provider_key = (provider or "").lower()
    if provider_key in {"openai", "azure_openai", "anthropic", "huggingface"} and not api_key:
        return f"API key is required for provider '{provider}'."
    return None


def _validate_embedding_inputs(provider: str, model: str | None, api_key: str | None) -> Optional[str]:
    if not provider or not provider.strip():
        return "Embedding provider is required."
    provider_key = provider.lower()
    if not model or not model.strip():
        return "Embedding model is required for the selected provider."
    if provider_key == "huggingface" and not api_key:
        return "Embedding API key is required for Hugging Face."
    return None


def sidebar_configuration() -> None:
    # Show prewarm instructions at the top - expanded by default for visibility
    with st.sidebar.expander("📋 Setup Instructions (Run First!)", expanded=True):
        st.markdown("""
        **Required workflow before using UI:**
        
        1. **Pre-load cache**: 
           ```bash
           python scripts/prewarm_metadata.py
           ```
           This populates metadata, graph cards, and embeddings.
        
        2. **Validate cache**:
           ```bash
           python scripts/validate_cache.py
           ```
           Verify all data is loaded correctly.
        
        3. **Then use UI**: Launch with `python run_app.py`
        
        ⚡ **Benefits**: Fast startup, no expensive LLM calls, optimal performance.
        """)
    logo_path = _resolve_brand_logo()
    if logo_path:
        st.sidebar.image(logo_path, width=220)
    elif APP_CONFIG.brand_name:
        st.sidebar.markdown(f"### {APP_CONFIG.brand_name}")
    st.sidebar.header("Configuration")
    profiles = st.session_state["saved_profiles"]
    profile_options = ["(New configuration)"] + sorted(profiles.keys())
    selected_profile = st.sidebar.selectbox(
        "Saved connections",
        options=profile_options,
        index=profile_options.index(st.session_state.get("active_profile_value", "(New configuration)")),
    )

    if selected_profile != st.session_state.get("active_profile_value"):
        st.session_state["active_profile_value"] = selected_profile
        if selected_profile == "(New configuration)":
            st.session_state.update(
                {
                    "profile_name": "",
                    "db_type": "Oracle",
                    "db_url": "",
                    "db_schema": "",
                    "oracle_thick_mode": False,
                    "oracle_lib_dir": "",
                    "oracle_config_dir": "",
                    "include_system_tables": False,
                    "llm_base_url": "",
                    "llm_api_key": "",
                    "connection_status": None,
                    "llm_status": None,
                    "service_initialised": False,
                    "available_schemas": [],
                    "schema_select_option": "(current schema)",
                    "embedding_provider": "huggingface",
                    "embedding_model": "google/embeddinggemma-300m",
                    "embedding_base_url": "",
                    "embedding_api_key": "",
                    "embedding_status": None,
                }
            )
        else:
            _update_inputs_from_profile(selected_profile)

    st.sidebar.subheader("Profile")
    profile_name = st.sidebar.text_input("Profile name", value=st.session_state["profile_name"])
    st.session_state["profile_name"] = profile_name

    with st.sidebar.expander("Database", expanded=True):
        db_status_placeholder = st.empty()
        _render_status(st.session_state.get("connection_status"), db_status_placeholder)

        db_types = list(DB_TYPE_TEMPLATES.keys())
        selected_db_type = st.selectbox(
            "Database type",
            options=db_types,
            index=db_types.index(st.session_state["db_type"]),
        )
        if selected_db_type != st.session_state["db_type"]:
            st.session_state["db_type"] = selected_db_type
            if selected_db_type != "Oracle":
                st.session_state["oracle_thick_mode"] = False
                st.session_state["oracle_lib_dir"] = ""
                st.session_state["oracle_config_dir"] = ""
            st.session_state["connection_status"] = None
            _render_status(None, db_status_placeholder)

        db_url = st.text_input(
            "SQLAlchemy connection URL",
            value=st.session_state["db_url"],
            placeholder=DB_TYPE_TEMPLATES.get(st.session_state["db_type"], ""),
        )

        available_schemas: List[str] = st.session_state.get("available_schemas", [])
        manual_schema = st.session_state.get("db_schema", "")
        schema_select_option = st.session_state.get("schema_select_option", "(current schema)")
        schema_options = ["(current schema)"]
        if available_schemas:
            schema_options.extend(available_schemas)
        schema_options.append("(manual entry)")
        if schema_select_option not in schema_options:
            schema_select_option = "(current schema)"
        schema_choice = st.selectbox(
            "Schema",
            options=schema_options,
            index=schema_options.index(schema_select_option),
            help="Choose a schema discovered during the connection test, select manual entry, or use the session's default.",
        )
        st.session_state["schema_select_option"] = schema_choice

        if schema_choice == "(current schema)":
            schema = ""
        elif schema_choice == "(manual entry)":
            schema = st.text_input(
                "Schema (manual entry)",
                value=manual_schema,
                help="Leave blank to rely on the current schema.",
            )
        else:
            schema = schema_choice

        st.session_state["db_schema"] = schema

        thick_mode = bool(st.session_state["oracle_thick_mode"])
        oracle_lib_dir = st.session_state["oracle_lib_dir"]
        oracle_config_dir = st.session_state["oracle_config_dir"]
        if st.session_state["db_type"] == "Oracle":
            thick_mode = st.checkbox(
                "Enable Oracle thick mode",
                value=bool(st.session_state["oracle_thick_mode"]),
                help="Requires Oracle Instant Client; specify directories below if needed.",
            )
            oracle_lib_dir = st.text_input(
                "Oracle client library directory",
                value=st.session_state["oracle_lib_dir"],
                help="Path containing OCI libraries (leave blank if on PATH).",
            )
            oracle_config_dir = st.text_input(
                "Oracle network config directory",
                value=st.session_state["oracle_config_dir"],
                help="Folder with tnsnames.ora; set when thick mode uses TNS aliases.",
            )
        else:
            thick_mode = False
            oracle_lib_dir = ""
            oracle_config_dir = ""

        sample_limit = st.slider(
            "Sample rows per table",
            min_value=10,
            max_value=500,
            value=st.session_state["sample_limit"],
            step=10,
        )

        include_system_tables = st.checkbox(
            "Include system tables",
            value=bool(st.session_state.get("include_system_tables", False)),
            help="Show database system tables in schema summaries (may be noisy).",
        )

        st.session_state.update(
            {
                "db_type": st.session_state["db_type"],
                "db_url": db_url,
                "db_schema": schema,
                "oracle_thick_mode": thick_mode,
                "oracle_lib_dir": oracle_lib_dir,
                "oracle_config_dir": oracle_config_dir,
                "sample_limit": sample_limit,
                "include_system_tables": include_system_tables,
                "available_schemas": available_schemas,
                "schema_select_option": schema_choice,
            }
        )

        test_db_clicked = st.button("Test database connection")
        save_clicked = st.button("Save configuration")
        delete_clicked = st.button("Delete configuration")

    with st.sidebar.expander("LLM", expanded=True):
        llm_status_placeholder = st.empty()
        _render_status(st.session_state.get("llm_status"), llm_status_placeholder)

        providers = ["ollama", "openai", "azure_openai", "anthropic", "huggingface"]
        previous_provider = st.session_state["llm_provider"]
        # Get current provider from session state, default to first option if not set
        current_provider_index = 0
        if st.session_state["llm_provider"] in providers:
            current_provider_index = providers.index(st.session_state["llm_provider"])
        provider = st.selectbox(
            "Provider",
            options=providers,
            index=current_provider_index,
            key="llm_provider_selectbox",  # Add key to ensure proper state tracking
        )
        # Update session state immediately when provider changes
        if provider != st.session_state.get("llm_provider"):
            st.session_state["llm_provider"] = provider
            st.session_state["llm_status"] = None  # Clear status when provider changes
        
        model = st.text_input("Model name", value=st.session_state["llm_model"], key="llm_model_input")
        base_url = st.session_state["llm_base_url"]
        base_url_help = ""
        base_url_placeholder = ""
        show_base_url = False
        if provider == "ollama":
            base_url_placeholder = "http://localhost:11434"
            base_url_help = "URL of your local Ollama server (default http://localhost:11434)."
            show_base_url = True
        elif provider == "azure_openai":
            base_url_placeholder = "https://<resource>.openai.azure.com"
            base_url_help = "Base endpoint for your Azure OpenAI resource."
            show_base_url = True
        elif provider == "huggingface":
            base_url_placeholder = "Optional custom inference endpoint URL"
            base_url_help = (
                "Leave blank to use the Hugging Face Inference API. "
                "Provide a custom endpoint only if you deployed your own inference server."
            )
            show_base_url = True

        if show_base_url:
            base_url = st.text_input(
                "Base URL",
                value=st.session_state["llm_base_url"],
                placeholder=base_url_placeholder,
                help=base_url_help,
            )
        else:
            base_url = ""
        api_key = st.text_input("API key", value=st.session_state["llm_api_key"], type="password")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["temperature"]),
            step=0.05,
        )
        max_tokens = st.slider(
            "Max output tokens",
            min_value=256,
            max_value=4096,
            value=int(st.session_state["max_tokens"]),
            step=256,
        )

        st.session_state.update(
            {
                "llm_provider": provider,
                "llm_model": model,
                "llm_base_url": base_url,
                "llm_api_key": api_key,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if provider != previous_provider:
            st.session_state["llm_status"] = None
            _render_status(None, llm_status_placeholder)

        test_llm_clicked = st.button("Test LLM connection")

    with st.sidebar.expander("Embeddings", expanded=False):
        embedding_status_placeholder = st.empty()
        _render_status(st.session_state.get("embedding_status"), embedding_status_placeholder)
        embedding_options = ["huggingface", "ollama"]
        previous_embedding_provider = st.session_state["embedding_provider"]
        embedding_provider = st.selectbox(
            "Embedding provider",
            options=embedding_options,
            index=embedding_options.index(previous_embedding_provider),
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=st.session_state["embedding_model"],
            help="Leave blank to disable embeddings when provider is 'none'.",
        ).strip()
        embedding_base_url = st.text_input(
            "Embedding base URL",
            value=st.session_state["embedding_base_url"],
            help="Required for self-hosted providers such as Ollama.",
            disabled=embedding_provider != "ollama",
        ).strip()
        embedding_api_key = st.text_input(
            "Embedding API key",
            value=st.session_state["embedding_api_key"],
            type="password",
            disabled=embedding_provider != "huggingface",
        ).strip()
        st.session_state.update(
            {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_base_url": embedding_base_url,
                "embedding_api_key": embedding_api_key,
            }
        )
        if embedding_provider != previous_embedding_provider:
            st.session_state["embedding_status"] = None
            _render_status(None, embedding_status_placeholder)

        test_embedding_clicked = st.button("Test embedding connection")

    st.sidebar.subheader("Agent actions")
    apply_clicked = st.sidebar.button("Initialise Agent", type="primary")
    clear_clicked = st.sidebar.button("Reset Agent")

    if test_db_clicked:
        db_error = _validate_db_inputs(db_url)
        if db_error:
            st.session_state["connection_status"] = ("error", db_error)
        else:
            try:
                db_config = DatabaseConfig(
                    url=db_url,
                    schema=schema or None,
                    sample_row_limit=sample_limit,
                    thick_mode=thick_mode,
                    oracle_lib_dir=oracle_lib_dir or None,
                    oracle_config_dir=oracle_config_dir or None,
                    include_system_tables=include_system_tables,
                )
                st.session_state["connection_status"] = _test_connection(db_config)
                if st.session_state["connection_status"][0] == "success":
                    try:
                        schemas = _fetch_schema_options(db_config)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Failed to fetch schema list: %s", exc)
                        schemas = []
                    st.session_state["available_schemas"] = schemas
                    if schemas:
                        st.session_state["schema_select_option"] = schemas[0]
                        st.session_state["db_schema"] = schemas[0]
                    else:
                        st.session_state["schema_select_option"] = schema_choice
            except Exception as exc:  # noqa: BLE001
                st.session_state["connection_status"] = ("error", str(exc))
        _render_status(st.session_state.get("connection_status"), db_status_placeholder)

    if save_clicked:
        db_error = _validate_db_inputs(db_url)
        llm_error = _validate_llm_inputs(provider, model, base_url, api_key)
        embedding_error = _validate_embedding_inputs(embedding_provider, embedding_model, embedding_api_key)
        if db_error:
            st.session_state["connection_status"] = ("error", db_error)
            _render_status(st.session_state.get("connection_status"), db_status_placeholder)
        elif llm_error:
            st.session_state["llm_status"] = ("error", llm_error)
            _render_status(st.session_state.get("llm_status"), llm_status_placeholder)
        elif embedding_error:
            st.session_state["embedding_status"] = ("error", embedding_error)
            _render_status(st.session_state.get("embedding_status"), embedding_status_placeholder)
        elif not profile_name:
            st.session_state["connection_status"] = ("error", "Profile name is required to save the configuration.")
        else:
            profile_payload = {
                "db_type": st.session_state["db_type"],
                "db_url": db_url,
                "schema": schema,
                "thick_mode": thick_mode,
                "oracle_lib_dir": oracle_lib_dir,
                "oracle_config_dir": oracle_config_dir,
                "sample_limit": sample_limit,
                "include_system_tables": include_system_tables,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_base_url": embedding_base_url,
                "embedding_api_key": embedding_api_key,
            }
            st.session_state["saved_profiles"] = upsert_profile(profile_name, profile_payload)
            st.session_state["connection_status"] = ("success", f"Saved configuration '{profile_name}'.")
            st.session_state["active_profile_value"] = profile_name
            st.session_state["embedding_status"] = None
        _render_status(st.session_state.get("connection_status"), db_status_placeholder)

    if delete_clicked and st.session_state.get("active_profile_value") not in (None, "(New configuration)"):
        name = st.session_state["active_profile_value"]
        st.session_state["saved_profiles"] = delete_profile(name)
        st.session_state["connection_status"] = ("warning", f"Deleted configuration '{name}'.")
        st.session_state["active_profile_value"] = "(New configuration)"
        st.session_state["profile_name"] = ""
        _render_status(st.session_state.get("connection_status"), db_status_placeholder)

    if apply_clicked:
        db_error = _validate_db_inputs(db_url)
        llm_error = _validate_llm_inputs(provider, model, base_url, api_key)
        embedding_error = _validate_embedding_inputs(embedding_provider, embedding_model, embedding_api_key)
        
        # Check if schema is selected (required for most databases, especially Oracle)
        schema_validation_error = None
        if not db_error:  # Only check schema if DB URL is valid
            schema_choice = st.session_state.get("schema_select_option", "(current schema)")
            manual_schema = st.session_state.get("db_schema", "")
            
            # Determine the actual schema value
            if schema_choice == "(current schema)":
                actual_schema = None  # Will use default schema
            elif schema_choice == "(manual entry)":
                actual_schema = manual_schema.strip() if manual_schema else None
            else:
                actual_schema = schema_choice
            
            # For Oracle databases, schema is typically required
            # Check if DB URL suggests Oracle (oracle:// or contains :1521)
            is_oracle = "oracle" in db_url.lower() or ":1521" in db_url or "oracle" in (st.session_state.get("db_type", "") or "").lower()
            
            if not actual_schema:
                if is_oracle:
                    # Oracle: Make it an error (blocking)
                    schema_validation_error = (
                        "⚠️ **Schema selection required for Oracle database.**\n\n"
                        "Please select a schema from the dropdown or enter one manually in the 'Database' section. "
                        "Initializing without a schema can take a very long time as it tries to introspect all schemas."
                    )
                else:
                    # Other databases: Show warning but allow (some DBs don't require schema)
                    # We'll show this as a warning in the status, not blocking
                    st.session_state["connection_status"] = (
                        "warning",
                        "⚠️ No schema selected. Initialization may take longer. "
                        "Consider selecting a schema from the dropdown for faster startup."
                    )
                    _render_status(st.session_state.get("connection_status"), db_status_placeholder)
        
        if db_error:
            st.session_state["connection_status"] = ("error", db_error)
            _render_status(st.session_state.get("connection_status"), db_status_placeholder)
        elif schema_validation_error:
            st.session_state["connection_status"] = ("error", schema_validation_error)
            _render_status(st.session_state.get("connection_status"), db_status_placeholder)
        elif llm_error:
            st.session_state["llm_status"] = ("error", llm_error)
            _render_status(st.session_state.get("llm_status"), llm_status_placeholder)
        elif embedding_error:
            st.session_state["embedding_status"] = ("error", embedding_error)
            _render_status(st.session_state.get("embedding_status"), embedding_status_placeholder)
        else:
            # Check cache status BEFORE initialization (so user knows cache status even if init fails)
            try:
                from sqlai.services.cache_health import metadata_table_names
                from sqlai.services.graph_cache import GraphCache
                from sqlai.services.metadata_cache import MetadataCache
                
                db_config = DatabaseConfig(
                    url=db_url,
                    schema=schema or None,
                    sample_row_limit=sample_limit,
                    thick_mode=thick_mode,
                    oracle_lib_dir=oracle_lib_dir or None,
                    oracle_config_dir=oracle_config_dir or None,
                    include_system_tables=include_system_tables,
                )
                
                # Quick cache check using the schema the user selected
                schema_to_check = schema or "(default)"
                app_config = load_app_config()
                metadata_cache = MetadataCache(app_config.cache_dir / "table_metadata.db")
                graph_cache = GraphCache(app_config.cache_dir / "graph_cards.db")
                metadata_tables = metadata_table_names(metadata_cache, schema_to_check)
                graph_tables = set(graph_cache.list_tables(schema_to_check))
                
                LOGGER.info(
                    "Pre-initialization cache check for schema '%s': %s metadata tables, %s graph tables",
                    schema_to_check,
                    len(metadata_tables),
                    len(graph_tables),
                )
                
                if len(metadata_tables) > 0 or len(graph_tables) > 0:
                    st.session_state["cache_precheck"] = (
                        "info",
                        f"✓ Cache detected for schema '{schema_to_check}': {len(metadata_tables)} metadata tables, {len(graph_tables)} graph tables. Initializing agent..."
                    )
                else:
                    st.session_state["cache_precheck"] = (
                        "warning",
                        f"⚠ No cache found for schema '{schema_to_check}'. Run 'python scripts/prewarm_metadata.py' first for faster startup."
                    )
            except Exception as cache_exc:  # noqa: BLE001
                LOGGER.debug("Could not check cache before initialization: %s", cache_exc)
                st.session_state["cache_precheck"] = None
            
            try:
                # Use session state values to ensure we get the latest (selectbox updates happen after render)
                # The session state is updated at line 466-474, so use those values
                final_provider = st.session_state.get("llm_provider", provider)
                final_model = st.session_state.get("llm_model", model)
                final_base_url = st.session_state.get("llm_base_url", base_url)
                final_api_key = st.session_state.get("llm_api_key", api_key)
                
                # Log what we're using for debugging
                LOGGER.info(
                    "Creating LLMConfig: provider='%s' (selectbox: '%s', session_state: '%s'), model='%s', base_url='%s'",
                    final_provider,
                    provider,
                    st.session_state.get("llm_provider", "NOT_SET"),
                    final_model,
                    final_base_url or "(none)",
                )
                llm_config = LLMConfig(
                    provider=final_provider,
                    model=final_model,
                    base_url=final_base_url or None,
                    api_key=final_api_key or None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                embedding_config = EmbeddingConfig(
                    provider=embedding_provider,
                    model=embedding_model,
                    base_url=embedding_base_url or None,
                    api_key=embedding_api_key or None,
                )
                # Explicitly enable cache usage - assumes prewarm_metadata.py was run
                # This prevents expensive LLM calls for regenerating descriptions
                service = AnalyticsService(
                    db_config=db_config,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                    skip_prewarm_if_cached=True,  # Use cache by default
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                
                # Show cache precheck message if available
                cache_precheck = st.session_state.get("cache_precheck")
                if cache_precheck:
                    cache_type, cache_msg = cache_precheck
                    if cache_type == "info":
                        st.info(cache_msg)
                    elif cache_type == "warning":
                        st.warning(cache_msg)
                
                # Enhance error message with specific guidance
                enhanced_error = error_msg
                error_lower = error_msg.lower()
                if "llm" in error_lower or "provider" in error_lower:
                    if "embedding" not in error_lower:
                        enhanced_error = (
                            f"{error_msg}\n\n"
                            "**Required:** Configure LLM provider and model in the sidebar under 'LLM' section. "
                            "Test the LLM connection before initializing the agent."
                        )
                elif "embedding" in error_lower:
                    enhanced_error = (
                        f"{error_msg}\n\n"
                        "**Required:** Configure embedding provider and model in the sidebar under 'Embeddings' section. "
                        "Test the embedding connection before initializing the agent."
                    )
                elif "connection" in error_lower or "failed" in error_lower:
                    enhanced_error = (
                        f"{error_msg}\n\n"
                        "**Troubleshooting:**\n"
                        "1. Verify LLM provider is running (e.g., Ollama server for 'ollama' provider)\n"
                        "2. Check API keys are correct (for Hugging Face, OpenAI, etc.)\n"
                        "3. Test connections individually using 'Test LLM connection' and 'Test embedding connection' buttons\n"
                        "4. Ensure all required fields are filled in the sidebar"
                    )
                
                st.session_state["connection_status"] = ("error", enhanced_error)
                _render_status(st.session_state.get("connection_status"), db_status_placeholder)
                LOGGER.error("Agent initialization failed: %s", exc, exc_info=True)
            else:
                set_service(service)
                # Get cache status from service using the ACTUAL schema the user configured
                # This is the accurate check - uses the schema from db_config
                try:
                    # Log what schema we're checking
                    configured_schema = db_config.schema or "(default)"
                    LOGGER.info("Checking cache status for configured schema: '%s'", configured_schema)
                    
                    cache_status = service.get_cache_status()
                    schema_name = cache_status.get("schema", configured_schema)
                    
                    LOGGER.info(
                        "Cache status result: schema='%s', metadata_tables=%s, graph_tables=%s, has_cache=%s, is_complete=%s",
                        schema_name,
                        cache_status["metadata_tables"],
                        cache_status["graph_tables"],
                        cache_status["has_cache"],
                        cache_status["is_complete"],
                    )
                    
                    if cache_status["has_cache"]:
                        cache_msg = (
                            f"Using cached metadata ({cache_status['metadata_tables']} tables) "
                            f"and graph cards ({cache_status['graph_tables']} tables) for schema '{schema_name}'."
                        )
                        if not cache_status["is_complete"]:
                            cache_msg += " ⚠️ Cache incomplete - run 'python scripts/prewarm_metadata.py' for full prewarm."
                        st.session_state["connection_status"] = ("success", cache_msg)
                    else:
                        st.session_state["connection_status"] = (
                            "warning",
                            f"No cache detected for schema '{schema_name}'. "
                            "Run 'python scripts/prewarm_metadata.py' first for faster startup.",
                        )
                except Exception as exc:  # noqa: BLE001
                    # If cache check fails, log the error and show success
                    LOGGER.error("Could not check cache status: %s", exc, exc_info=True)
                st.session_state["connection_status"] = ("success", "Agent initialised successfully.")
                
                _render_status(st.session_state.get("connection_status"), db_status_placeholder)
                st.session_state["service_initialised"] = True
                st.session_state["embedding_status"] = None
                # Clear cache precheck message after successful initialization
                st.session_state.pop("cache_precheck", None)
                st.session_state["conversation_history"] = []
                if not st.session_state.get("available_schemas"):
                    try:
                        st.session_state["available_schemas"] = _fetch_schema_options(db_config)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Failed to refresh schema list: %s", exc)

    if clear_clicked:
        clear_service()
        st.session_state["connection_status"] = ("warning", "Cleared cached agent. Configure to begin again.")
        _render_status(st.session_state.get("connection_status"), db_status_placeholder)
        st.session_state["llm_status"] = None
        _render_status(st.session_state.get("llm_status"), llm_status_placeholder)
        st.session_state["service_initialised"] = False
        st.session_state["available_schemas"] = []
        st.session_state["schema_select_option"] = "(current schema)"
        st.session_state["embedding_status"] = None
        st.session_state["conversation_history"] = []
        _clear_persisted_session_config()

    if test_llm_clicked:
        llm_error = _validate_llm_inputs(provider, model, base_url, api_key)
        if llm_error:
            status = ("error", llm_error)
        else:
            status = _test_llm_connection(provider, model, base_url, api_key)
        st.session_state["llm_status"] = status
        _render_status(status, llm_status_placeholder)

    if test_embedding_clicked:
        embedding_error = _validate_embedding_inputs(embedding_provider, embedding_model, embedding_api_key)
        if embedding_error:
            status = ("error", embedding_error)
        else:
            status = _test_embedding_connection(
                embedding_provider,
                embedding_model or "",
                embedding_base_url or "",
                embedding_api_key or "",
            )
        st.session_state["embedding_status"] = status
        _render_status(status, embedding_status_placeholder)

    _persist_last_session_config()


def _test_llm_connection(provider: str, model: str, base_url: str | None, api_key: str | None) -> Tuple[str, str]:
    provider_key = (provider or "").lower()
    try:
        if provider_key == "ollama":
            try:
                from ollama import Client
            except ImportError:
                return "error", "Ollama client library missing. Install with `pip install ollama`."
            client = Client(host=base_url or "http://localhost:11434")
            response = client.chat(model=model, messages=[{"role": "user", "content": "ping"}], stream=False)
            content = (response or {}).get("message", {}).get("content", "")
            snippet = content.strip()
            if len(snippet) > 80:
                snippet = snippet[:80] + "…"
            return "success", f"Ollama responded: {snippet or '<<empty>>'}"
        if provider_key == "huggingface":
            if not api_key:
                return "error", "Hugging Face API key is required."
            try:
                from openai import OpenAI
            except ImportError:
                return "error", "openai package missing. Install with `pip install openai`."
            # Always hit the HF OpenAI-compatible router for chat models
            base = (base_url or "https://router.huggingface.co/v1").rstrip("/")
            client = OpenAI(base_url=base, api_key=api_key or "")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=16,
                    # Prefer a widely available provider to avoid 401s on private providers
                    extra_body={"provider": "nscale"},
                )
                content = ""
                if response.choices:
                    content = response.choices[0].message.content or ""
                snippet = content.strip()
            except Exception as exc:  # noqa: BLE001
                return "error", f"Hugging Face router error: {exc}"
            snippet = snippet.strip()
            if len(snippet) > 80:
                snippet = snippet[:80] + "…"
            message = f"Hugging Face responded: {snippet or '<<empty>>'} (using router endpoint)"
            return "success", message
        return "warning", f"LLM connectivity test not implemented for provider '{provider}'."
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"


def _test_embedding_connection(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Tuple[str, str]:
    provider_key = (provider or "").lower()
    try:
        LOGGER.debug(
            "Testing embedding connection",
            extra={
                "provider": provider_key,
                "model": model,
                "base_url": base_url,
                "has_api_key": bool(api_key),
            },
        )
        if not provider_key:
            return "error", "Embedding provider is required."
        if not model.strip():
            return "error", "Embedding model is required."
        if provider_key == "huggingface":
            if not api_key:
                return "error", "Embedding API key is required for Hugging Face."
            try:
                from huggingface_hub import InferenceClient
                import os
            except ImportError:
                return "error", "huggingface_hub package missing. Install with `pip install huggingface_hub`."
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            os.environ["HF_TOKEN"] = api_key
            parameters: Dict[str, str] = {}
            if base_url:
                parameters["base_url"] = base_url.rstrip("/")
            client = InferenceClient(provider="hf-inference", api_key=api_key, **parameters)
            try:
                payload = {
                    "source_sentence": "ping",
                    "sentences": ["pong", "embedding test"],
                }
                LOGGER.debug(
                    "Sending Hugging Face sentence_similarity request",
                    extra={"model": model},
                )
                try:
                    result = client.sentence_similarity("ping", ["pong", "embedding test"], model=model)
                except TypeError:
                    result = client.sentence_similarity(payload, model=model)
                return "success", f"Hugging Face embedding OK (scores: {', '.join(f'{score:.3f}' for score in result)})"
            except Exception as inner_exc:  # noqa: BLE001
                LOGGER.exception(
                    "Hugging Face embedding test failed",
                    extra={"model": model},
                )
                return "error", f"{type(inner_exc).__name__}: {inner_exc}"
        if provider_key == "ollama":
            url = (base_url or "http://localhost:11434").rstrip("/")
            try:
                import requests
            except ImportError:
                return "error", "requests package missing. Install with `pip install requests`."
            response = requests.post(
                f"{url}/api/embeddings",
                json={"model": model, "prompt": "ping"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("embedding", [])
            size = len(vector)
            return "success", f"Ollama embedding OK (vector length {size})."
        return "warning", f"Embedding connectivity test not implemented for provider '{provider}'."
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "Embedding connection test failed",
            extra={
                "provider": provider_key,
                "model": model,
                "base_url": base_url,
            },
        )
        return "error", f"{type(exc).__name__}: {exc}"


def _render_schema_from_cache(
    graph_cache: "GraphCache",
    metadata_cache: "MetadataCache",
    schema: str,
) -> None:
    """Render schema directly from cache without requiring full service initialization."""
    import json
    
    with st.expander("Database Overview (from cache)", expanded=False):
        try:
            cached_tables = set(graph_cache.list_tables(schema))
            if not cached_tables:
                st.info("No tables found in cache for this schema. Run 'python scripts/prewarm_metadata.py' to populate cache.")
                return
            
            st.caption(
                "ℹ️ Schema information loaded from cache. "
                "If database schema changed, re-run 'python scripts/prewarm_metadata.py' to refresh."
            )
            
            for table_name in sorted(cached_tables):
                # Get all cards for this table
                cards = graph_cache.get_cards_for_table(schema, table_name)
                table_card = next((c for c in cards if c.card_type == "table"), None)
                column_cards = [c for c in cards if c.card_type == "column"]
                relationship_cards = [c for c in cards if c.card_type == "relationship"]
                
                if not table_card:
                    continue
                
                st.markdown(f"#### {table_name}")
                
                # Get metadata
                metadata_entry = metadata_cache.fetch(schema, table_name)
                if metadata_entry and metadata_entry.get("description"):
                    st.markdown(f"*{metadata_entry['description']}*")
                
                # Show columns
                if column_cards:
                    column_lines = []
                    for col_card in sorted(column_cards, key=lambda c: c.identifier):
                        try:
                            # GraphCardRecord has metadata as dict, not metadata_json
                            metadata = col_card.metadata if hasattr(col_card, 'metadata') and col_card.metadata else {}
                            col_type = metadata.get("type", "unknown")
                            nullable = "NULL" if metadata.get("nullable") else "NOT NULL"
                            column_lines.append(f"- **{col_card.identifier}**: {col_type} ({nullable})")
                        except Exception as col_exc:  # noqa: BLE001
                            # If metadata parsing fails, just show the column name
                            column_lines.append(f"- **{col_card.identifier}**: (metadata unavailable)")
                    if column_lines:
                        st.markdown("\n".join(column_lines))
                
                # Show relationships
                if relationship_cards:
                    try:
                        fk_text = ", ".join([c.identifier for c in relationship_cards])
                        st.markdown(f"*Foreign keys:* {fk_text}")
                    except Exception:  # noqa: BLE001
                        pass  # Skip FK display if there's an error
                
                st.markdown("---")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Error loading schema from cache")
            st.error(f"Could not load schema from cache: {exc}")
            st.info("This might happen if the cache is incomplete. Run 'python scripts/prewarm_metadata.py' to populate the cache.")


def render_schema(service: AnalyticsService) -> None:
    with st.expander("Database Overview", expanded=False):
        if not service.has_schema():
            st.info(
                "No tables were detected for the current configuration. "
                "Verify the schema name or enable 'Include system tables' in the sidebar."
            )
        else:
            # Note: Schema information is loaded from cache if available (for performance).
            # If database schema changed, re-run 'python scripts/prewarm_metadata.py' to refresh cache.
            st.caption(
                "ℹ️ Schema information loaded from cache for fast display. "
                "If database schema changed, re-run 'python scripts/prewarm_metadata.py' to refresh."
            )
            for summary in service.schema_summaries:
                st.markdown(f"#### {summary.name}")
                if getattr(summary, "description", None):
                    st.caption(summary.description)
                column_lines = []
                for column in summary.columns:
                    value_hint = (
                        f" | values≈{', '.join(column.sample_values)}"
                        if getattr(column, "sample_values", None)
                        else ""
                    )
                    column_lines.append(
                        f"- `{column.name}` ({column.type})"
                        f"{' nullable' if column.nullable else ''}"
                        f"{value_hint}"
                    )
                st.markdown("\n".join(column_lines))
                if summary.foreign_keys:
                    fk_text = ", ".join(_format_fk_detail(fk) for fk in summary.foreign_keys)
                    st.markdown(f"*Foreign keys:* {fk_text}")
                if summary.comment:
                    st.markdown(f"*Comment:* {summary.comment}")
                st.markdown("---")


def render_saved_queries(service: AnalyticsService) -> None:
    saved = service.list_saved_queries()
    with st.expander("Saved queries", expanded=False):
        if not saved:
            st.info("No saved queries cached yet. Run a question to create one.")
            return
        for entry in saved:
            run_at = datetime.fromtimestamp(entry["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"**{entry['question']}**")
            st.caption(f"Last saved: {run_at}")
            cols = st.columns([1, 1, 3])
            if cols[0].button("Run", key=f"run_saved_{entry['id']}"):
                # Rerun with fresh summary - skip similarity detection
                st.session_state["saved_query_result"] = service.rerun_saved_query_with_fresh_summary(entry["id"])
                st.session_state["current_question"] = entry["question"]
                st.session_state["saved_query_message"] = f"Reran query with fresh results (saved on {run_at})."
                st.session_state["skip_similarity_check"] = True  # Skip similarity check on next run
                _trigger_rerun()
            if cols[1].button("Load", key=f"load_saved_{entry['id']}"):
                st.session_state["current_question"] = entry["question"]
                _trigger_rerun()
            with cols[2]:
                st.code(entry["sql"], language="sql")
            st.markdown("---")


def render_executions(result: Dict[str, Any]) -> None:
    executions = result.get("executions", [])
    if not executions:
        st.info("No execution results to display.")
        return
    for idx, execution in enumerate(executions):
        st.subheader(f"Query {idx + 1}")
        st.code(execution["sql"], language="sql")
        dataframe = pd.DataFrame(execution["data"])
        row_count = execution.get("row_count")
        if row_count is not None:
            st.caption(f"{row_count} row(s) returned.")
        st.dataframe(dataframe, width="stretch")
        stats = execution.get("stats") or {}
        if stats.get("columns"):
            with st.expander("Summary stats", expanded=False):
                st.json(stats)
        chart_spec = result.get("chart")
        if chart_spec:
            try:
                figure = build_chart(chart_spec, dataframe)
            except ValueError as err:
                st.warning(f"Unable to render chart: {err}")
            else:
                if figure:
                    st.plotly_chart(figure, width="stretch")


def _check_cache_before_start() -> None:
    """
    Show informational message about cache setup before user configures database.
    
    Note: We can't check cache for a specific schema here because the user hasn't
    configured the database yet. The actual cache check happens after agent initialization
    using the schema the user selected (see get_cache_status() call after service initialization).
    """
    st.info(
        "💡 **Tip:** For faster startup, pre-load your cache before using the UI:\n\n"
        "1. **Pre-load cache**: `python scripts/prewarm_metadata.py`\n"
        "2. **Validate cache**: `python scripts/validate_cache.py`\n\n"
        "After you configure the database and initialize the agent, the UI will check cache "
        "for your selected schema and show the status."
    )


def main() -> None:
    init_session_state()
    st.title("💡 SQLAI: Ask your database anything")
    
    # Check cache status before showing sidebar
    _check_cache_before_start()
    
    sidebar_configuration()

    # Check configuration status BEFORE trying to initialize service
    # This allows us to show schema from cache even if LLM/embedding aren't configured
    db_config = None
    llm_config = None
    embedding_config = None
    missing_requirements = []
    
    # Try to get DB config from session state
    db_url = st.session_state.get("db_url", "")
    # Get schema from session state - this is the actual selected schema
    selected_schema = st.session_state.get("db_schema", "")
    
    # Log the schema being used for debugging
    LOGGER.info(
        "Main function: db_url='%s', selected_schema from session_state='%s'",
        db_url[:50] + "..." if len(db_url) > 50 else db_url,
        selected_schema or "(empty/None)",
    )
    
    # Check if user has actually configured database (not just empty)
    has_configured_db = bool(db_url and db_url.strip())
    
    if has_configured_db:
        try:
            db_config = DatabaseConfig(
                url=db_url,
                schema=selected_schema.strip() if selected_schema else None,  # Use selected_schema, strip whitespace
                sample_row_limit=st.session_state.get("sample_limit", 100),
                thick_mode=st.session_state.get("oracle_thick_mode", False),
                oracle_lib_dir=st.session_state.get("oracle_lib_dir") or None,
                oracle_config_dir=st.session_state.get("oracle_config_dir") or None,
                include_system_tables=st.session_state.get("include_system_tables", False),
            )
            # Log what schema was actually set in db_config
            LOGGER.info(
                "DatabaseConfig created: db_config.schema='%s'",
                db_config.schema or "(None/empty)",
            )
        except Exception as db_config_exc:  # noqa: BLE001
            LOGGER.warning("Failed to create DatabaseConfig: %s", db_config_exc)
            db_config = None
    
    # Check LLM config
    llm_provider = st.session_state.get("llm_provider", "")
    llm_model = st.session_state.get("llm_model", "")
    llm_base_url = st.session_state.get("llm_base_url", "")
    llm_api_key = st.session_state.get("llm_api_key", "")
    
    # Check if these are just default values (user hasn't configured anything)
    is_default_llm = (
        llm_provider == "ollama" and
        llm_model == "llama3" and
        not llm_base_url and
        not llm_api_key
    )
    
    if not llm_provider or not llm_model:
        # Empty values - don't show error (user hasn't started configuring)
        pass
    elif is_default_llm:
        # Default values - don't show error (user hasn't configured anything)
        pass
    else:
        # User has configured something - validate it
        llm_validation_error = _validate_llm_inputs(
            llm_provider,
            llm_model,
            llm_base_url or None,
            llm_api_key or None,
        )
        if llm_validation_error:
            missing_requirements.append(f"**LLM**: {llm_validation_error}")
        else:
            try:
                llm_config = LLMConfig(
                    provider=llm_provider,
                    model=llm_model,
                    base_url=llm_base_url or None,
                    api_key=llm_api_key or None,
                    temperature=st.session_state.get("temperature", 0.1),
                    max_output_tokens=st.session_state.get("max_tokens", 2048),
                )
            except Exception as exc:  # noqa: BLE001
                missing_requirements.append(f"**LLM**: Configuration error: {str(exc)}. Check the sidebar under 'LLM' section.")
    
    # Check Embedding config
    embedding_provider = st.session_state.get("embedding_provider", "")
    embedding_model = st.session_state.get("embedding_model", "")
    embedding_api_key = st.session_state.get("embedding_api_key", "")
    
    # Check if these are just default values (user hasn't configured anything)
    is_default_embedding = (
        embedding_provider == "huggingface" and
        embedding_model == "google/embeddinggemma-300m" and
        not embedding_api_key
    )
    
    if not embedding_provider or not embedding_model:
        # Empty values - don't show error (user hasn't started configuring)
        pass
    elif is_default_embedding:
        # Default values - don't show error (user hasn't configured anything)
        pass
    else:
        # User has configured something - validate it
        embedding_validation_error = _validate_embedding_inputs(
            embedding_provider, 
            embedding_model, 
            embedding_api_key
        )
        if embedding_validation_error:
            missing_requirements.append(f"**Embedding**: {embedding_validation_error}")
        else:
            try:
                embedding_config = EmbeddingConfig(
                    provider=embedding_provider,
                    model=embedding_model,
                    base_url=st.session_state.get("embedding_base_url") or None,
                    api_key=embedding_api_key or None,
                )
            except Exception as exc:  # noqa: BLE001
                missing_requirements.append(f"**Embedding**: Configuration error: {str(exc)}. Check the sidebar under 'Embeddings' section.")
    
    # Only show configuration errors if user has actually started configuring something
    # OR if they've tried to initialize the agent
    service_initialized_attempted = st.session_state.get("service_initialised", False)
    
    if missing_requirements and (has_configured_db or not is_default_llm or not is_default_embedding or service_initialized_attempted):
        st.error("**⚠️ Configuration Required**\n\n" + "\n\n".join(f"• {req}" for req in missing_requirements))
        st.info("💡 **Setup Steps:**\n1. Configure **Database** connection in sidebar\n2. Configure **LLM** provider and model in sidebar\n3. Configure **Embedding** provider and model in sidebar\n4. Click **'Initialise Agent'** button\n5. Then you can ask questions!")
    
    # Try to initialize service
    service = init_service()
    service_ready = False
    
    if not service:
        if st.session_state.get("service_initialised"):
            st.error("❌ Agent initialisation failed. Check the sidebar configuration and logs for details.")
        # Don't return early - continue to show schema from cache if available
    else:
        # Service initialized - check if it's fully ready
        try:
            if service.llm_config and service.llm_config.provider and service.llm_config.model:
                if service.embedding_config and service.embedding_config.provider and service.embedding_config.model:
                    service_ready = True
        except Exception:  # noqa: BLE001
            pass
    
    # Try to render schema from cache even if service isn't fully initialized
    if db_config:
        try:
            from sqlai.services.cache_health import metadata_table_names
            from sqlai.services.graph_cache import GraphCache
            from sqlai.services.metadata_cache import MetadataCache
            from sqlai.database.schema_introspector import TableSummary, ColumnMetadata, ForeignKeyDetail
            import json
            
            # Use the schema from db_config (which comes from session state) to ensure consistency
            schema_to_check = (db_config.schema or "(default)") if db_config else "(default)"
            LOGGER.info("Rendering schema from cache for schema: '%s'", schema_to_check)
            app_config = load_app_config()
            metadata_cache = MetadataCache(app_config.cache_dir / "table_metadata.db")
            graph_cache = GraphCache(app_config.cache_dir / "graph_cards.db")
            metadata_tables = metadata_table_names(metadata_cache, schema_to_check)
            graph_tables = set(graph_cache.list_tables(schema_to_check))
            
            if metadata_tables or graph_tables:
                # We have cache - try to render schema
                if service and service.has_schema():
                    render_schema(service)
                else:
                    # Render schema directly from cache
                    _render_schema_from_cache(graph_cache, metadata_cache, schema_to_check)
        except Exception as cache_exc:  # noqa: BLE001
            LOGGER.debug("Could not render schema from cache: %s", cache_exc)
            if service and service.has_schema():
                render_schema(service)
    
    st.divider()
    
    # Only show saved queries and conversation if service is fully initialized
    if service and service_ready:
        render_saved_queries(service)
        st.divider()

        history = service.get_conversation()
        if history:
            with st.expander("Recent conversation", expanded=False):
                for item in history:
                    st.markdown(f"**You:** {item['question']}")
                answer_text = item.get("answer") or "<no answer>"
                if isinstance(answer_text, str):
                    st.markdown(f"**SQLAI:** {answer_text}")
                else:
                    st.json(answer_text)
                if item.get("execution_error"):
                    st.warning(f"Execution error: {item['execution_error']}")
                st.markdown("---")

    question_placeholder = "Which test sets had the highest failure rate last week?"
    st.text_area(
        "Ask a question",
        placeholder=question_placeholder,
        key="current_question",
        disabled=not service_ready,  # Disable if requirements not met
    )
    
    # Optional desired columns input - directly below question input
    st.caption(
        "💡 **Optional:** Specify desired columns (comma-separated) for more exact output. "
        "You can use synonyms or natural names - the LLM will map them to actual column names. "
        "Example: 'test set name, test case name, status' or 'name, id, created date'"
    )
    desired_columns_input = st.text_input(
        "Desired columns (optional)",
        value=st.session_state.get("desired_columns", ""),
        placeholder="e.g., test set name, test case name, status, severity",
        key="desired_columns_input",
        disabled=not service_ready,
        help="Leave blank to let the agent decide which columns to include. You can use natural language or synonyms - the LLM will map them correctly."
    )
    st.session_state["desired_columns"] = desired_columns_input.strip()
    
    ask_clicked = st.button("Run analysis", type="primary", disabled=not service_ready)  # Disable button if requirements not met

    # Execute saved query automatically if user already confirmed reuse
    reuse_entry_id = st.session_state.pop("reuse_query_entry_id", None)
    if reuse_entry_id:
        if service_ready and service:
            try:
                saved_result = service.execute_saved_query(reuse_entry_id)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to execute saved query: {exc}")
                LOGGER.exception("Failed to execute saved query", exc_info=True)
                st.session_state["similar_question_confirmation"] = None
            else:
                source_question = st.session_state.pop("reuse_query_source", "previous question")
                st.session_state["saved_query_result"] = saved_result
                st.session_state["saved_query_message"] = f"Reused query from: '{source_question}'"
                st.session_state["similar_question_confirmation"] = None
                st.success("Query executed successfully from saved history.")
        else:
            st.warning("Cannot execute saved query until the agent is fully initialised.")
            st.session_state["similar_question_confirmation"] = None

    if ask_clicked or st.session_state.get("force_new_query"):
        # Double-check service is ready (defensive check)
        if not service_ready:
            st.error("Cannot run analysis: Agent is not properly configured. Please complete the setup steps shown above.")
            return
        
        question = (st.session_state.get("current_question") or "").strip()
        desired_columns = st.session_state.get("desired_columns", "").strip()
        st.session_state["saved_query_result"] = None
        force_new = st.session_state.pop("force_new_query", False)
        
        if not question:
            st.warning("Please enter a question before running the analysis.")
        else:
            # Check for similar questions first (unless forcing new query or explicitly skipping)
            skip_similarity = force_new or st.session_state.pop("skip_similarity_check", False)
            spinner_text = "Generating new query..." if skip_similarity else "Checking for similar questions..."
            with st.spinner(spinner_text):
                try:
                    result = service.ask(
                        question, 
                        skip_similarity_check=skip_similarity,
                        desired_columns=desired_columns if desired_columns else None,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                else:
                    # Check if similar question found and confirmation required
                    if result.get("similar_question_found") and result.get("confirmation_required") and not force_new:
                        # Store confirmation info for rendering outside the spinner (don't show duplicate messages here)
                        st.session_state["similar_question_confirmation"] = {
                            "question": question,
                            "matched_question": result["matched_question"],
                            "matched_sql": result["matched_sql"],
                            "entry_id": result["entry_id"],
                            "similarity_score": result["similarity_score"],
                            "intent_confidence": result["intent_confidence"],
                        }
                        # Don't show duplicate messages here - they'll be shown in the confirmation UI below
                        st.success("Similar question detected. Please choose an option below.")
                    else:
                        # Normal result display
                        if not force_new and not result.get("similar_question_found"):
                            st.info("No similar saved question found. Generating a fresh analysis.")
                    if result.get("execution_error"):
                        st.warning(result["execution_error"])
                    else:
                        if result.get("plan", {}).get("sql_generation_note"):
                            st.warning(result["plan"]["sql_generation_note"])
                        st.success("Analysis complete.")
                        
                        # Handle answer display (could be dict or string)
                        answer_text = result.get("answer", {})
                        if isinstance(answer_text, dict):
                            answer_text = answer_text.get("text", "")
                        st.markdown(answer_text)
                    
                    # Display the full prompt and final SQL
                    if result.get("formatted_prompt") or result.get("final_sql"):
                        with st.expander("📋 Full Prompt & Final SQL (After Intent Critic/Repair)", expanded=False):
                            if result.get("final_sql"):
                                st.subheader("Final SQL (After All Intent Repairs)")
                                st.caption("This is the SQL that will be executed, after all intent critic/repair iterations.")
                                st.code(result["final_sql"], language="sql")
                                st.divider()
                            
                            if result.get("formatted_prompt"):
                                st.subheader("Full Planner Prompt")
                                st.caption("This is the complete prompt including all graph context (tables, columns, relationships) that was sent to the planner LLM to generate the SQL query. This prompt is logged at INFO level in the application logs.")
                                st.code(result["formatted_prompt"], language="text")
                    
                    if result.get("plan"):
                        with st.expander("LLM Plan", expanded=False):
                            st.json(result["plan"])
                    if result.get("followups"):
                        st.markdown("**Suggested follow-ups:**")
                        for item in result["followups"]:
                            st.markdown(f"- {item}")
                    if not result.get("execution_error"):
                        render_executions(result)
                    st.session_state["conversation_history"] = service.get_conversation()
                    st.session_state["saved_queries"] = service.list_saved_queries()
                    st.session_state["saved_query_message"] = None

    # Display saved query result (from "Yes, reuse previous query" or saved queries panel)
    # Show confirmation UI if similar question detected
    confirmation_state = st.session_state.get("similar_question_confirmation")
    if confirmation_state:
        st.divider()
        st.info(
            f"A similar question was found:\n\n"
            f"- Previous question: **{confirmation_state['matched_question']}**\n"
            f"- Similarity: {confirmation_state['similarity_score']:.2f}\n"
            f"- Intent confidence: {confirmation_state['intent_confidence']:.2f}\n\n"
            "Would you like to reuse the previous SQL or generate a new one?"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, reuse previous query", type="primary", key="reuse_query_confirm"):
                st.session_state["reuse_query_entry_id"] = confirmation_state["entry_id"]
                st.session_state["reuse_query_source"] = confirmation_state["matched_question"]
                st.rerun()
        with col2:
            if st.button("🔄 No, generate new query", key="generate_new_confirm"):
                st.session_state["similar_question_confirmation"] = None
                st.session_state["force_new_query"] = True
                st.rerun()

    saved_result = st.session_state.get("saved_query_result")
    if saved_result:
        st.divider()
        message = st.session_state.get("saved_query_message")
        if message:
            st.success(message)  # Changed to success for better visibility
        # Display answer text
        answer_text = saved_result.get("answer", "")
        if answer_text:
            # Handle both dict and string formats
            if isinstance(answer_text, dict):
                answer_text = answer_text.get("text", "") or str(answer_text)
            if answer_text:
                st.markdown(f"**Answer:** {answer_text}")
        # Display SQL that was executed
        if saved_result.get("sql"):
            with st.expander("📋 Executed SQL", expanded=True):
                st.code(saved_result["sql"], language="sql")
        # Display plan if available
        if saved_result.get("plan"):
            with st.expander("Cached plan", expanded=False):
                st.json(saved_result["plan"])
        # Render execution results (data, charts, etc.)
        render_executions(saved_result)
        # Clear the saved result after displaying (so it doesn't persist across new questions)
        # But keep the message for context
        # st.session_state["saved_query_result"] = None  # Don't clear - let user see it


if __name__ == "__main__":
    main()

