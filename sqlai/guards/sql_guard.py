"""
Basic guard and repair utilities for generated SQL.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

_FROM_JOIN_PATTERN = re.compile(r"\b(FROM|JOIN)\s+([a-zA-Z0-9_\.\"]+)", re.IGNORECASE)


def guard_sql(
    sql: str,
    metadata: Dict[str, object],
    schema: Optional[str],
    dialect: Optional[str],
    row_cap: Optional[int],
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Validate SQL statement against known tables and attempt simple repairs.

    Returns (possibly modified sql, error message (if any), guard note).
    """

    tables_meta = metadata.get("tables") if isinstance(metadata, dict) else None
    if not isinstance(tables_meta, list):
        return sql, None, None
    known_tables = {
        (table.get("name") or "").lower(): table for table in tables_meta if table.get("name")
    }
    qualified_names = {
        f"{(table.get('schema') or schema or '').lower()}.{table.get('name').lower()}": table
        for table in tables_meta
        if table.get("name")
    }

    replacements: Dict[str, str] = {}
    missing: List[str] = []
    for match in _FROM_JOIN_PATTERN.finditer(sql):
        raw = match.group(2).rstrip(",")
        identifier = raw.split()[0]
        identifier_clean = identifier.strip('"')
        identifier_lower = identifier_clean.lower()
        base = identifier_lower.split(".")[-1]
        if identifier_lower in qualified_names or base in known_tables:
            if schema and "." not in identifier_clean and base in known_tables:
                replacements[identifier] = f"{schema}.{identifier_clean}"
            continue
        missing.append(identifier_clean)

    if missing:
        missing_str = ", ".join(sorted(set(missing)))
        return sql, f"Unknown tables referenced: {missing_str}", None

    repaired_sql = sql
    note_parts: List[str] = []
    for original, replacement in replacements.items():
        repaired_sql = re.sub(rf"\b{re.escape(original)}\b", replacement, repaired_sql, count=1)
        if original != replacement:
            note_parts.append(f"Prefixed table {original} -> {replacement}")

    limited_sql = _ensure_row_cap(repaired_sql, row_cap=row_cap, dialect=dialect)
    if limited_sql != repaired_sql:
        note_parts.append("Applied row cap to limit result size.")

    note_text = "; ".join(note_parts) if note_parts else None
    return limited_sql, None, note_text


def _ensure_row_cap(sql: str, row_cap: Optional[int], dialect: Optional[str]) -> str:
    if row_cap is None or row_cap <= 0:
        return sql
    dialect = (dialect or "").lower()
    lower_sql = sql.lower()
    if not lower_sql.strip().startswith("select"):
        return sql
    if " limit " in lower_sql or " fetch first " in lower_sql or " rownum " in lower_sql:
        return sql
    row_cap = int(row_cap)
    if dialect.startswith("oracle"):
        return f"{sql.rstrip(';')} FETCH FIRST {row_cap} ROWS ONLY"
    if dialect in {"postgresql", "postgres", "mysql", "mariadb", "sqlite"}:
        return f"{sql.rstrip(';')} LIMIT {row_cap}"
    if dialect in {"mssql", "sqlserver"} and " top " not in lower_sql[:20]:
        return f"SELECT TOP ({row_cap})" + sql[6:]
    return sql

