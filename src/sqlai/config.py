"""
Configuration models and loaders for SQLAI.

ALL CONFIGURATION IS CENTRALIZED IN THIS FILE - ONE LOCATION FOR ALL CONFIGS.

This file contains all configuration classes:
- DatabaseConfig: Database connection settings (SQLAI_DB_ prefix)
- LLMConfig: LLM provider settings (SQLAI_LLM_ prefix)
- EmbeddingConfig: Embedding provider settings (SQLAI_EMBED_ prefix)
- VectorStoreConfig: Vector store settings (SQLAI_VECTOR_ prefix)
- LimitsConfig: All limits and thresholds (SQLAI_LIMITS_ prefix)
- AppConfig: Application behavior settings (SQLAI_ prefix)

All configs are loaded from .env file via environment variables.
Each config class has its own prefix to avoid conflicts.
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


class LimitsConfig(BaseSettings):
    """
    Centralized configuration for all limits and thresholds in the application.
    All limits can be configured via environment variables with SQLAI_LIMITS_ prefix.
    Example: SQLAI_LIMITS_MAX_TABLES=10 in .env file
    
    ============================================================================
    TABLE FILTER FEATURE - QUICK GUIDE
    ============================================================================
    
    WHAT IS TABLE FILTER?
    - Feature in UI that lets you select specific tables to focus the search
    - Creates "filter groups" to save and reuse table selections
    - Speeds up vector search by limiting scope to selected tables
    
    WHEN TO USE FILTER:
    ✓ Large databases (500+ tables) - significantly faster searches
    ✓ Domain-specific queries - you know the answer is in specific tables
    ✓ Performance optimization - reduces search time and token usage
    ✓ Precision over recall - want only highly relevant tables
    
    WHEN NOT TO USE FILTER:
    ✗ Exploring unknown schemas - let the system search all tables
    ✗ Small databases (< 50 tables) - filter overhead not worth it
    ✗ Unsure which tables are relevant - better to search broadly first
    
    HOW FILTER AFFECTS BEHAVIOR:
    1. Selected tables are MANDATORY - always included (even if score < threshold)
    2. Vector search is RESTRICTED - only searches selected tables (faster)
    3. Similarity threshold INCREASES - 0.70 (with filter) vs 0.5 (no filter)
    4. FK expansion is UNRESTRICTED - can add related tables outside filter
    5. Question mentions are DETECTED - tables mentioned in query are added
    
    ============================================================================
    
    All Limits in LimitsConfig:
    - max_tables: Maximum tables from vector search (before FK expansion)
    - max_columns: Maximum columns in context (None = no limit)
    - fk_expansion_max_depth: Maximum FK expansion depth (hops)
    - fk_expansion_max_tables: Maximum additional tables via FK expansion
    - similarity_threshold_no_filter: Similarity threshold when no filter (better recall)
    - similarity_threshold_with_filter: Similarity threshold when filter active (better precision)
    - similarity_threshold_similar_questions: Threshold for detecting similar questions
    - max_repair_iterations: Maximum repair/retry iterations
    - max_top_values: Maximum top values in dataframe profiling
    - recent_queries_limit: Maximum recent queries from cache
    - conversation_limit: Maximum conversation messages
    - sample_query_limit: Number of sample rows for previews
    - vector_store_limit: Maximum records in vector store operations
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Table Selection Limits
    # 
    # NOTE: When filter is active, filter tables are MANDATORY and never cut off by max_tables limit.
    # Example: If max_tables=6 but filter has 10 tables, all 10 filter tables are included.
    #
    max_tables: int = Field(
        default=6,
        description=(
            "Maximum number of tables to select from vector search (before FK expansion). "
            "When filter is active: Filter tables are MANDATORY (always included, even if exceeds this limit). "
            "This limit applies to optional tables (vector results, question mentions, FK expanded)."
        ),
        ge=1,
        le=50,
    )
    max_columns: Optional[int] = Field(
        default=None,
        description="Maximum number of columns to include in context. None = no limit (include all columns).",
        ge=10,
    )

    # FK Expansion Limits
    # 
    # FK EXPANSION BEHAVIOR WITH FILTER:
    # - When filter is active: FK expansion starts from selected filter tables
    # - Can add tables OUTSIDE filter: This is intentional for join accuracy
    # - Example: Filter=["orders"], FK finds "customers" table → "customers" is added even if not in filter
    # - This ensures complete join paths even when filter is restrictive
    #
    fk_expansion_max_depth: int = Field(
        default=3,
        description=(
            "Maximum depth for FK expansion (multi-hop traversal). "
            "1=direct only, 2=1-hop, 3=2-hop, etc. Higher values include more connected tables. "
            "Works with or without filter: When filter is active, expansion starts from filter tables but can add tables outside filter for join accuracy."
        ),
        ge=1,
        le=5,
    )
    fk_expansion_max_tables: int = Field(
        default=20,
        description=(
            "Maximum number of additional tables to add via FK expansion. "
            "Prevents context explosion in highly connected schemas. "
            "When filter is active: Counts tables added via FK relationships (may include tables outside filter)."
        ),
        ge=5,
        le=50,
    )

    # Similarity Thresholds
    # 
    # TABLE FILTER FEATURE EXPLANATION:
    # The table filter allows users to select specific tables (via UI filter groups) to focus the search.
    # 
    # WHEN TO USE FILTER:
    # - Large databases (500+ tables): Speeds up vector search by limiting scope
    # - Domain-specific queries: When you know the answer is in specific tables (e.g., "sales", "inventory")
    # - Performance optimization: Reduces search time and token usage
    # - Precision over recall: When you want only highly relevant tables, not all possible matches
    #
    # HOW FILTER WORKS:
    # - Selected tables are MANDATORY: Always included in context (even if similarity score is low)
    # - Vector search is RESTRICTED: Only searches within selected tables (faster)
    # - FK expansion is UNRESTRICTED: Can still add related tables outside filter (for join accuracy)
    # - Question mentions are DETECTED: Tables mentioned in question are added even if not in filter
    #
    # THRESHOLD DIFFERENCES:
    # - No filter (0.5): Lower threshold = better recall (includes more tables, even less relevant ones)
    # - With filter (0.70): Higher threshold = better precision (only highly relevant from filtered set)
    #
    similarity_threshold_no_filter: float = Field(
        default=0.5,
        description=(
            "Minimum similarity score (0.0-1.0) for table selection when NO filter is active. "
            "Lower threshold (0.5) = better recall (more tables included, even if less relevant). "
            "Use this mode when: searching across entire database, exploring unknown schemas, or when you're unsure which tables are relevant. "
            "Vector search scans ALL tables in the database."
        ),
        ge=0.0,
        le=1.0,
    )
    similarity_threshold_with_filter: float = Field(
        default=0.70,
        description=(
            "Minimum similarity score (0.0-1.0) for table selection when filter IS active. "
            "Higher threshold (0.70) = better precision (only highly relevant tables). "
            "Use this mode when: you know which tables contain the answer, working with large databases (500+ tables), "
            "or want faster, more focused searches. Selected filter tables are MANDATORY (always included). "
            "Vector search is restricted to selected tables only (faster), but FK expansion can still add related tables outside filter."
        ),
        ge=0.0,
        le=1.0,
    )
    similarity_threshold_similar_questions: float = Field(
        default=0.75,
        description=(
            "Minimum similarity score (0.0-1.0) for detecting similar questions in conversation cache. "
            "Used to find previously asked questions that match the current question. "
            "Higher threshold (0.75) = only very similar questions are matched (prevents false positives)."
        ),
        ge=0.0,
        le=1.0,
    )

    # Repair/Retry Limits
    max_repair_iterations: int = Field(
        default=2,
        description="Maximum number of repair iterations for both intent critic/repair (pre-execution) and post-execution critic/repair. "
        "Each iteration includes one critic check and one repair attempt.",
        ge=1,
        le=5,
    )

    # Data Profiling Limits
    max_top_values: int = Field(
        default=5,
        description="Maximum number of top values to show in dataframe profiling.",
        ge=1,
        le=20,
    )

    # Conversation/Cache Limits
    recent_queries_limit: int = Field(
        default=20,
        description="Maximum number of recent queries to retrieve from conversation cache.",
        ge=5,
        le=100,
    )
    conversation_limit: int = Field(
        default=10,
        description="Maximum number of conversation messages to retrieve.",
        ge=5,
        le=50,
    )

    # Sampling Limits
    sample_query_limit: int = Field(
        default=3,
        description="Number of sample rows to fetch for table previews.",
        ge=1,
        le=10,
    )

    # Vector Store Limits
    vector_store_limit: int = Field(
        default=100_000,
        description="Maximum number of records to process in vector store operations.",
        ge=1_000,
        le=1_000_000,
    )

    model_config = SettingsConfigDict(
        env_prefix="SQLAI_LIMITS_",
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
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable telemetry/analytics (default: False to prevent PostHog SSL errors). "
        "Set SQLAI_TELEMETRY_ENABLED=true in .env to enable.",
    )
    streamlit_port: int = Field(default=8501)
    brand_name: Optional[str] = Field(default=None, description="Display name for UI branding.")
    brand_logo_path: Optional[Path] = Field(
        default=None,
        description="Path to a logo image displayed in the UI sidebar.",
    )
    skip_prewarm_if_cached: bool = Field(
        default=True,
        description="If True, skip expensive prewarm steps (LLM calls, sampling) when cache exists. "
        "Set to False to force refresh or if cache is stale.",
    )
    # DEPRECATED: These fields are kept for backward compatibility only.
    # They are REDUNDANT - use LimitsConfig instead (SQLAI_LIMITS_ prefix).
    # These will be removed in a future version.
    fk_expansion_max_depth: int = Field(
        default=3,
        description="[DEPRECATED/REDUNDANT: Use SQLAI_LIMITS_FK_EXPANSION_MAX_DEPTH instead] Maximum depth for FK expansion.",
        ge=1,
        le=5,
    )
    fk_expansion_max_tables: int = Field(
        default=20,
        description="[DEPRECATED/REDUNDANT: Use SQLAI_LIMITS_FK_EXPANSION_MAX_TABLES instead] Maximum additional tables via FK expansion.",
        ge=5,
        le=50,
    )
    max_repair_iterations: int = Field(
        default=2,
        description="[DEPRECATED/REDUNDANT: Use SQLAI_LIMITS_MAX_REPAIR_ITERATIONS instead] Maximum repair iterations.",
        ge=1,
        le=5,
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
def load_limits_config() -> LimitsConfig:
    """Load limits configuration from environment variables."""
    return LimitsConfig()


@lru_cache()
def load_app_config() -> AppConfig:
    config = AppConfig()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config

