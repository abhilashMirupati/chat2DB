"""
SQL dialect transpilation using SQLGlot.
Automatically converts SQL from one dialect to another.
"""

from __future__ import annotations

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

# SQLGlot dialect name mapping from SQLAlchemy dialect names
DIALECT_MAPPING = {
    "oracle": "oracle",
    "postgresql": "postgres",
    "postgres": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",  # SQLGlot uses mysql for mariadb
    "mssql": "tsql",
    "sqlserver": "tsql",
    "sqlite": "sqlite",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "redshift": "redshift",
    "databricks": "spark",
    "spark": "spark",
    "trino": "trino",
    "presto": "presto",
    "hive": "hive",
    "clickhouse": "clickhouse",
    "duckdb": "duckdb",
}


def transpile_sql(sql: str, target_dialect: str, source_dialect: Optional[str] = None) -> tuple[str, bool]:
    """
    Transpile SQL to target dialect using SQLGlot.
    
    Args:
        sql: SQL query to transpile
        target_dialect: Target SQLAlchemy dialect name (e.g., "oracle", "postgresql")
        source_dialect: Optional source dialect name. If None, SQLGlot will try to auto-detect.
    
    Returns:
        Tuple of (transpiled_sql, was_transpiled)
        - transpiled_sql: The transpiled SQL (or original if transpilation failed/not needed)
        - was_transpiled: True if transpilation was successful, False otherwise
    
    Raises:
        ImportError: If SQLGlot is not installed
    """
    import sqlglot
    
    # Map SQLAlchemy dialect name to SQLGlot dialect name
    target_glot = DIALECT_MAPPING.get(target_dialect.lower())
    if not target_glot:
        LOGGER.debug("Unknown target dialect '%s', skipping transpilation", target_dialect)
        return sql, False
    
    source_glot = None
    if source_dialect:
        source_glot = DIALECT_MAPPING.get(source_dialect.lower())
        if not source_glot:
            LOGGER.debug("Unknown source dialect '%s', will auto-detect", source_dialect)
            source_glot = None
    
    try:
        # Parse SQL (auto-detect source dialect if not provided)
        parsed = sqlglot.parse_one(sql, read=source_glot)
        
        # Transpile to target dialect
        transpiled = parsed.sql(dialect=target_glot, pretty=False)
        
        if transpiled != sql:
            LOGGER.info(
                "Transpiled SQL from %s to %s dialect",
                source_glot or "auto-detected",
                target_glot
            )
            LOGGER.debug("Original SQL: %s", sql)
            LOGGER.debug("Transpiled SQL: %s", transpiled)
            return transpiled, True
        
        # SQL unchanged (already in target dialect or no changes needed)
        return sql, False
        
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Failed to transpile SQL to %s dialect: %s. Using original SQL.",
            target_glot,
            exc
        )
        LOGGER.debug("SQL that failed to transpile: %s", sql)
        return sql, False

