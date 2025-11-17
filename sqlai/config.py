"""
Configuration models and loaders for SQLAI.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """
    Settings required to connect to any SQL database supported by SQLAlchemy.
    """

    model_config = SettingsConfigDict(extra="ignore")

    url: str = Field(..., description="SQLAlchemy connection string.")
    schema: Optional[str] = Field(
        default=None,
        description="Default schema for introspection when the database supports it.",
    )
    sample_row_limit: int = Field(
        default=100,
        description="Maximum number of rows to sample for data previews.",
        ge=10,
        le=10_000,
    )
    thick_mode: bool = Field(
        default=False,
        description="Enable Oracle thick mode (requires Instant Client).",
    )
    oracle_lib_dir: Optional[str] = Field(
        default=None,
        description="Path to Oracle Instant Client libraries (required when thick_mode is true).",
    )
    oracle_config_dir: Optional[str] = Field(
        default=None,
        description="Path to Oracle network configuration (tnsnames.ora) when thick_mode is true.",
    )
    include_system_tables: bool = Field(
        default=False,
        description="Include database system tables in schema summaries.",
    )

    model_config = SettingsConfigDict(
        env_prefix="SQLAI_DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMConfig(BaseSettings):
    """
    Settings for the large language model provider.
    """

    model_config = SettingsConfigDict(extra="ignore")

    provider: Literal["openai", "anthropic", "ollama", "huggingface", "azure_openai"] = Field(
        default="ollama",
        description="Name of the LLM provider to use.",
    )
    model: str = Field(default="llama3", description="Model identifier for the chosen provider.")
    base_url: Optional[str] = Field(
        default=None,
        description="Optional base URL for self-hosted providers or custom endpoints.",
    )
    api_key: Optional[str] = Field(default=None, description="API key for hosted providers.")
    temperature: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Creativity setting for the model."
    )
    max_output_tokens: int = Field(default=1_024, description="Maximum tokens for responses.")

    model_config = SettingsConfigDict(
        env_prefix="SQLAI_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @validator("api_key", always=True)
    def validate_api_key(cls, value: Optional[str], values: dict[str, str]) -> Optional[str]:
        provider = values.get("provider")
        if provider in {"openai", "anthropic", "azure_openai", "huggingface"} and not value:
            raise ValueError(f"Provider '{provider}' requires an API key.")
        return value


class EmbeddingConfig(BaseSettings):
    """
    Settings for the embedding provider used in graph retrieval.
    """

    model_config = SettingsConfigDict(extra="ignore")

    provider: Literal["huggingface", "ollama"] = Field(
        default="huggingface",
        description="Embedding provider to use for semantic retrieval.",
    )
    model: Optional[str] = Field(
        default="google/embeddinggemma-300m",
        description="Embedding model identifier.",
    )
    base_url: Optional[str] = Field(
        default=None, description="Optional base URL for self-hosted embedding providers."
    )
    api_key: Optional[str] = Field(default=None, description="API key for hosted providers.")

    model_config = SettingsConfigDict(
        env_prefix="SQLAI_EMBED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @validator("api_key", always=True)
    def validate_embed_api_key(cls, value: Optional[str], values: dict[str, str]) -> Optional[str]:
        provider = values.get("provider")
        if provider in {"huggingface"} and not value:
            raise ValueError(f"Provider '{provider}' requires an API key.")
        return value


class VectorStoreConfig(BaseSettings):
    """
    Settings for persisted vector store used to cache graph card embeddings.
    """

    model_config = SettingsConfigDict(extra="ignore")

    provider: Literal["chroma"] = Field(
        default="chroma",
        description="Vector store provider for persisted embeddings.",
    )
    path: Optional[Path] = Field(
        default=None,
        description="Directory for vector store persistence (defaults to cache_dir/vector_store).",
    )
    collection: str = Field(
        default="graph_cards",
        description="Collection name used within the vector store backend.",
    )

    model_config = SettingsConfigDict(
        env_prefix="SQLAI_VECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppConfig(BaseSettings):
    """
    Top-level settings for application behaviour.
    """

    model_config = SettingsConfigDict(extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    cache_dir: Path = Field(default_factory=lambda: Path(".cache"))
    telemetry_enabled: bool = Field(default=False)
    streamlit_port: int = Field(default=8501)
    brand_name: Optional[str] = Field(default=None, description="Display name for UI branding.")
    brand_logo_path: Optional[Path] = Field(
        default=None,
        description="Path to a logo image displayed in the UI sidebar.",
    )

    model_config = SettingsConfigDict(
        env_prefix="SQLAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @validator("brand_logo_path", pre=True)
    def _validate_logo(cls, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            base = Path(__file__).resolve().parents[2]
            path = base / path
        return path if path.exists() else None


@lru_cache()
def load_database_config() -> DatabaseConfig:
    return DatabaseConfig()


@lru_cache()
def load_llm_config() -> LLMConfig:
    return LLMConfig()


@lru_cache()
def load_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig()


@lru_cache()
def load_vector_store_config() -> VectorStoreConfig:
    return VectorStoreConfig()


@lru_cache()
def load_app_config() -> AppConfig:
    config = AppConfig()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config

