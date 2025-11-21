"""Streamlit UI for the SQLAI agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sqlai.config import DatabaseConfig, LLMConfig, EmbeddingConfig, load_app_config
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

load_dotenv()
configure_logging()

st.set_page_config(page_title="SQLAI: Natural Language Analytics", page_icon="💡", layout="wide")

LOGGER = get_logger(__name__)
APP_CONFIG = load_app_config()

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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


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
        provider = st.selectbox(
            "Provider",
            options=providers,
            index=providers.index(st.session_state["llm_provider"]),
        )
        model = st.text_input("Model name", value=st.session_state["llm_model"])
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
        if db_error:
            st.session_state["connection_status"] = ("error", db_error)
            _render_status(st.session_state.get("connection_status"), db_status_placeholder)
        elif llm_error:
            st.session_state["llm_status"] = ("error", llm_error)
            _render_status(st.session_state.get("llm_status"), llm_status_placeholder)
        elif embedding_error:
            st.session_state["embedding_status"] = ("error", embedding_error)
            _render_status(st.session_state.get("embedding_status"), embedding_status_placeholder)
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
                llm_config = LLMConfig(
                    provider=provider,
                    model=model,
                    base_url=base_url or None,
                    api_key=api_key or None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                embedding_config = EmbeddingConfig(
                    provider=embedding_provider,
                    model=embedding_model,
                    base_url=embedding_base_url or None,
                    api_key=embedding_api_key or None,
                )
                service = AnalyticsService(
                    db_config=db_config,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state["connection_status"] = ("error", str(exc))
                _render_status(st.session_state.get("connection_status"), db_status_placeholder)
            else:
                set_service(service)
                st.session_state["connection_status"] = ("success", "Agent initialised successfully.")
                _render_status(st.session_state.get("connection_status"), db_status_placeholder)
                st.session_state["service_initialised"] = True
                st.session_state["embedding_status"] = None
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


def render_schema(service: AnalyticsService) -> None:
    with st.expander("Database Overview", expanded=False):
        if not service.has_schema():
            st.info(
                "No tables were detected for the current configuration. "
                "Verify the schema name or enable 'Include system tables' in the sidebar."
            )
        else:
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
                st.session_state["saved_query_result"] = service.execute_saved_query(entry["id"])
                st.session_state["current_question"] = entry["question"]
                st.session_state["saved_query_message"] = f"Replayed cached SQL saved on {run_at}."
                _trigger_rerun()
            if cols[1].button("Load", key=f"load_saved_{entry['id']}"):
                st.session_state["current_question"] = entry["question"]
                _trigger_rerun()
            with cols[2]:
                st.code(entry["sql"], language="sql")
            st.markdown("---")


def render_executions(result: Dict[str, Any]) -> None:
    for idx, execution in enumerate(result["executions"]):
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


def main() -> None:
    init_session_state()
    st.title("💡 SQLAI: Ask your database anything")
    sidebar_configuration()

    service = init_service()
    if not service:
        if st.session_state.get("service_initialised"):
            st.error("Agent initialisation failed. Check the sidebar configuration and logs for details.")
        else:
            st.info("Configure the database and LLM in the sidebar to begin.")
        return

    render_schema(service)
    st.divider()
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
    )
    ask_clicked = st.button("Run analysis", type="primary")

    if ask_clicked or st.session_state.get("force_new_query"):
        question = (st.session_state.get("current_question") or "").strip()
        st.session_state["saved_query_result"] = None
        force_new = st.session_state.pop("force_new_query", False)
        
        if not question:
            st.warning("Please enter a question before running the analysis.")
        else:
            # Check for similar questions first (unless forcing new query)
            spinner_text = "Generating new query..." if force_new else "Checking for similar questions..."
            with st.spinner(spinner_text):
                try:
                    result = service.ask(question, skip_similarity_check=force_new)
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                else:
                    # Check if similar question found and confirmation required
                    if result.get("similar_question_found") and result.get("confirmation_required") and not force_new:
                        st.session_state["similar_question_confirmation"] = {
                            "question": question,
                            "matched_question": result["matched_question"],
                            "matched_sql": result["matched_sql"],
                            "entry_id": result["entry_id"],
                            "similarity_score": result["similarity_score"],
                            "intent_confidence": result["intent_confidence"],
                        }
                        st.info(result["answer"]["text"])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Yes, reuse previous query", type="primary", key="reuse_query"):
                                # Execute the saved query
                                try:
                                    saved_result = service.execute_saved_query(result["entry_id"])
                                    st.session_state["saved_query_result"] = saved_result
                                    st.session_state["saved_query_message"] = f"Reused query from: '{result['matched_question']}'"
                                    st.session_state["similar_question_confirmation"] = None
                                    st.success("Query executed successfully!")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Failed to execute saved query: {exc}")
                        
                        with col2:
                            if st.button("🔄 No, generate new query", key="generate_new"):
                                # Re-run with skip_similarity_check
                                st.session_state["similar_question_confirmation"] = None
                                st.session_state["force_new_query"] = True
                                st.rerun()
                    else:
                        # Normal result display
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

    saved_result = st.session_state.get("saved_query_result")
    if saved_result:
        st.divider()
        message = st.session_state.get("saved_query_message")
        if message:
            st.info(message)
        st.markdown(saved_result["answer"])
        if saved_result.get("plan"):
            with st.expander("Cached plan", expanded=False):
                st.json(saved_result["plan"])
        render_executions(saved_result)


if __name__ == "__main__":
    main()

