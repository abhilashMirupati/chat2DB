"""
Schema introspection helpers for summarising databases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class ColumnMetadata:
    name: str
    type: str
    nullable: bool
    default: Optional[str] = None
    comment: Optional[str] = None
    sample_values: Optional[List[str]] = None


@dataclass(frozen=True)
class ForeignKeyDetail:
    constrained_columns: List[str]
    referred_schema: Optional[str]
    referred_table: str
    referred_columns: List[str]


@dataclass(frozen=True)
class TableSummary:
    name: str
    columns: List[ColumnMetadata]
    foreign_keys: List[ForeignKeyDetail]
    row_estimate: int | None = None
    comment: Optional[str] = None
    description: Optional[str] = None


def introspect_database(
    engine: Engine,
    schema: str | None = None,
    include_system_tables: bool = False,
) -> List[TableSummary]:
    """
    Collect table metadata to ground the LLM.
    """

    inspector = inspect(engine)
    schema_name = schema or inspector.default_schema_name
    table_names = inspector.get_table_names(schema=schema_name)
    summaries: List[TableSummary] = []
    dialect = getattr(engine.dialect, "name", "").lower()
    for table in table_names:
        if not include_system_tables and _is_system_table(table, dialect):
            continue
        column_info: List[ColumnMetadata] = []
        for column in inspector.get_columns(table, schema=schema_name):
            column_info.append(
                ColumnMetadata(
                    name=column["name"],
                    type=str(column.get("type")),
                    nullable=bool(column.get("nullable", True)),
                    default=column.get("default"),
                    comment=column.get("comment"),
                )
            )
        foreign_keys: List[ForeignKeyDetail] = []
        for fk in inspector.get_foreign_keys(table, schema=schema_name):
            foreign_keys.append(
                ForeignKeyDetail(
                    constrained_columns=fk.get("constrained_columns", []),
                    referred_schema=fk.get("referred_schema"),
                    referred_table=fk.get("referred_table", ""),
                    referred_columns=fk.get("referred_columns", []),
                )
            )
        table_comment = inspector.get_table_comment(table, schema=schema_name) or {}
        summaries.append(
            TableSummary(
                name=table,
                columns=column_info,
                foreign_keys=foreign_keys,
                row_estimate=_estimate_row_count(inspector, table, schema_name),
                comment=table_comment.get("text"),
            )
        )
    return summaries


def introspection_dataframe(summaries: Iterable[TableSummary]) -> pd.DataFrame:
    """
    Represent metadata as a Pandas DataFrame for display or logging.
    """

    rows = [
        {
            "table": summary.name,
            "columns": ", ".join(column.name for column in summary.columns),
            "foreign_keys": ", ".join(_format_fk_detail(fk) for fk in summary.foreign_keys)
            if summary.foreign_keys
            else "",
            "row_estimate": summary.row_estimate,
            "comment": summary.comment or "",
            "description": summary.description or "",
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows)


def summarise_schema(summaries: Iterable[TableSummary]) -> str:
    """
    Convert metadata to a concise text block for prompt context.
    """

    return "\n".join(
        f"Table {summary.name} (approx {summary.row_estimate or 'unknown'} rows): "
        f"columns={', '.join(column.name for column in summary.columns)}; "
        f"foreign_keys={', '.join(_format_fk_detail(fk) for fk in summary.foreign_keys) if summary.foreign_keys else 'none'}"
        for summary in summaries
    )


def _estimate_row_count(inspector, table: str, schema: str | None) -> int | None:
    try:
        info = inspector.get_table_options(table, schema=schema)
    except NotImplementedError:
        info = {}
    return info.get("row_count")


def _is_system_table(table_name: str, dialect: str) -> bool:
    name = table_name.upper()
    if dialect == "oracle":
        oracle_prefixes = (
            "LOGMNR",
            "OBJ$",
            "AQ$",
            "DR$",
            "WRI$",
            "MLOG$",
            "RUPD$",
            "CDEF$",
            "DEST_",
            "IDL_",
            "SOURCE_$",
            "X$",
            "V_$",
            "SYS_IOT",
        )
        return "$" in name or name.startswith(oracle_prefixes)
    if dialect in {"postgresql", "postgres"}:
        return name.startswith(("PG_", "SQL_"))
    if dialect in {"mysql", "mariadb"}:
        return name.startswith(("SYS_", "MYSQL", "PERFORMANCE_SCHEMA", "INFORMATION_SCHEMA"))
    if dialect == "sqlite":
        return name.startswith("SQLITE_")
    if dialect in {"mssql", "sqlserver"}:
        return name.startswith(("SYS", "MS"))
    return False


def _format_fk_detail(detail: ForeignKeyDetail) -> str:
    left = ",".join(detail.constrained_columns)
    right_schema = f"{detail.referred_schema}." if detail.referred_schema else ""
    right = ",".join(detail.referred_columns)
    return f"[{left}] -> {right_schema}{detail.referred_table}.[{right}]"

