"""
Database connector utilities built on SQLAlchemy.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
import importlib
from typing import Any, Dict, List, Optional

from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from sqlai.config import DatabaseConfig

LOGGER = logging.getLogger(__name__)


def create_db_engine(config: DatabaseConfig) -> Engine:
    """
    Instantiate a SQLAlchemy engine from the provided configuration.

    The connection string may point to Oracle, PostgreSQL, MySQL, SQL Server,
    or any other backend supported by SQLAlchemy.
    """

    _ensure_oracle_client(config)
    kwargs: Dict[str, Any] = {"pool_pre_ping": True, "future": True}
    engine = create_engine(config.url, **kwargs)
    LOGGER.info("Created SQLAlchemy engine for url=%s", mask_url(config.url))
    return engine


def mask_url(url: str) -> str:
    if "@" in url:
        prefix, suffix = url.split("@", maxsplit=1)
        return "***@" + suffix
    return url


@contextmanager
def get_connection(engine: Engine):
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def build_sql_database(engine: Engine, config: DatabaseConfig) -> SQLDatabase:
    """
    Create a LangChain SQLDatabase wrapper that agents can query with SQL.
    """

    return SQLDatabase(engine, schema=config.schema, sample_rows_in_table_info=config.sample_row_limit)


def test_connection(engine: Engine) -> Optional[str]:
    """
    Execute a lightweight query to validate connectivity.
    Returns an error message if the connection fails.
    """

    try:
        probe_sql = _probe_statement_for_dialect(engine.dialect.name or "")
        with engine.connect() as connection:
            connection.execute(text(probe_sql))
        return None
    except SQLAlchemyError as err:
        LOGGER.exception("Database connectivity test failed.")
        return str(err)


def _probe_statement_for_dialect(dialect_name: str) -> str:
    name = dialect_name.lower()
    if "oracle" in name:
        return "SELECT 1 FROM DUAL"
    if "mssql" in name or "sqlserver" in name:
        return "SELECT 1"
    if "sqlite" in name:
        return "SELECT 1"
    if "snowflake" in name:
        return "SELECT 1"
    return "SELECT 1"


def _ensure_oracle_client(config: DatabaseConfig) -> None:
    if not config.thick_mode:
        return
    if "oracle" not in config.url.lower():
        LOGGER.warning("Oracle thick mode requested but connection URL is not Oracle: %s", config.url)
        return
    try:
        oracledb = importlib.import_module("oracledb")
    except ImportError as exc:  # noqa: PERF203
        raise RuntimeError(
            "Oracle thick mode requires the python-oracledb package. Install with `pip install oracledb`."
        ) from exc

    init_kwargs: Dict[str, Any] = {}
    if config.oracle_lib_dir:
        init_kwargs["lib_dir"] = config.oracle_lib_dir
    if config.oracle_config_dir:
        init_kwargs["config_dir"] = config.oracle_config_dir

    try:
        if init_kwargs:
            oracledb.init_oracle_client(**init_kwargs)
        else:
            oracledb.init_oracle_client()
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "DPY-3013" in message:  # already initialized
            LOGGER.debug("Oracle client already initialised; continuing.")
            return
        raise


def list_schemas(engine: Engine) -> List[str]:
    inspector = inspect(engine)
    try:
        schema_names = inspector.get_schema_names()
    except NotImplementedError:
        schema_names = []
    normalized = {name.upper() for name in schema_names if name}
    return sorted(normalized)

