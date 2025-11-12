"""
Guard and repair utilities for generated SQL.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sqlparse

from sqlai.graph.context import GraphContext

TABLE_PATTERN = re.compile(r"\bFROM\s+([a-zA-Z0-9_\.]+)|\bJOIN\s+([a-zA-Z0-9_\.]+)", re.IGNORECASE)
SELECT_STAR_PATTERN = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)
SENSITIVE_PATTERN = re.compile(r"\b({columns})\b", re.IGNORECASE)
ALIAS_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_\.]+)(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?",
    re.IGNORECASE,
)
COLUMN_EQ_LITERAL_PATTERN = re.compile(
    r"(?P<lhs>\b[a-zA-Z0-9_\.]+)\s*=\s*'(?P<value>[^']*)'",
    re.IGNORECASE,
)
LIKE_LITERAL_PATTERN = re.compile(
    r"(?P<lhs>\b[a-zA-Z0-9_\.]+)\s+LIKE\s+'(?P<value>[^']*)'",
    re.IGNORECASE,
)


def extract_tables(sql: str) -> List[str]:
    matches = TABLE_PATTERN.findall(sql)
    tables: List[str] = []
    for left, right in matches:
        if left:
            tables.append(left.strip())
        elif right:
            tables.append(right.strip())
    return tables


def contains_select_star(sql: str) -> bool:
    return bool(SELECT_STAR_PATTERN.search(sql))


def contains_sensitive_columns(sql: str, sensitive: Sequence[str]) -> bool:
    if not sensitive:
        return False
    pattern = SENSITIVE_PATTERN.pattern.format(
        columns="|".join(re.escape(col) for col in sensitive)
    )
    return bool(re.compile(pattern, re.IGNORECASE).search(sql))


def _build_alias_map(sql: str) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for match in ALIAS_PATTERN.finditer(sql):
        table_token = (match.group(1) or "").strip()
        alias_token = (match.group(2) or "").strip()
        if not table_token:
            continue
        bare_table = table_token.split(".")[-1].lower()
        alias_map[bare_table] = bare_table
        alias_map[table_token.lower()] = bare_table
        if alias_token:
            alias_map[alias_token.lower()] = bare_table
    return alias_map


def _build_sample_index(graph_context: GraphContext) -> Dict[str, Dict[str, List[str]]]:
    table_samples: Dict[str, Dict[str, List[str]]] = {}
    for table_card in graph_context.tables:
        column_samples: Dict[str, List[str]] = {}
        for column in table_card.columns:
            values = getattr(column, "sample_values", None)
            if values:
                column_samples[column.name.lower()] = list(values)
        if column_samples:
            table_samples[table_card.name.lower()] = column_samples
    return table_samples


def _resolve_table_for_column(
    qualifier: Optional[str],
    column_name: str,
    alias_map: Dict[str, str],
    table_samples: Dict[str, Dict[str, List[str]]],
) -> Optional[str]:
    if qualifier:
        table = alias_map.get(qualifier.lower())
        if table:
            return table
    candidates = [
        table for table, columns in table_samples.items() if column_name.lower() in columns
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _align_literals_with_samples(
    sql: str,
    graph_context: GraphContext,
) -> Tuple[str, List[str]]:
    alias_map = _build_alias_map(sql)
    table_samples = _build_sample_index(graph_context)
    if not table_samples:
        return sql, []

    errors: List[str] = []

    def _replace(match: re.Match) -> str:
        lhs = match.group("lhs")
        value = match.group("value")
        parts = lhs.split(".")
        column_name = parts[-1]
        qualifier = parts[-2] if len(parts) >= 2 else None
        table = _resolve_table_for_column(qualifier, column_name, alias_map, table_samples)
        if not table:
            return match.group(0)
        samples = table_samples.get(table, {}).get(column_name.lower())
        if not samples:
            return match.group(0)
        lookup = {sample.lower(): sample for sample in samples}
        normalized = value.lower()
        if normalized in lookup:
            canonical = lookup[normalized]
            if canonical != value:
                return f"{lhs} = '{canonical}'"
            return match.group(0)
        close = get_close_matches(normalized, lookup.keys(), n=1, cutoff=0.7)
        if close:
            suggestion = lookup[close[0]]
            return f"{lhs} = '{suggestion}'"
        return match.group(0)

    patched_sql = COLUMN_EQ_LITERAL_PATTERN.sub(_replace, sql)
    # Normalize LIKE patterns for case-insensitive matching when users type lowercase literals.
    patched_sql = _normalize_like_patterns(patched_sql)
    return patched_sql, errors


def _normalize_like_patterns(sql: str) -> str:
    def _needs_lower_wrap(lhs: str) -> bool:
        stripped = lhs.strip()
        return not stripped.lower().startswith("lower(")

    def _repl(match: re.Match) -> str:
        lhs = match.group("lhs")
        value = match.group("value")
        if not value:
            return match.group(0)
        literal_has_upper = any(char.isupper() for char in value)
        if not literal_has_upper and _needs_lower_wrap(lhs):
            return f"LOWER({lhs}) LIKE '{value.lower()}'"
        return match.group(0)

    return LIKE_LITERAL_PATTERN.sub(_repl, sql)


def enforce_row_cap(sql: str, row_cap: int, dialect: str) -> Tuple[bool, str]:
    if row_cap <= 0:
        return True, sql
    stripped = sql.rstrip()
    # Remove any stray ROWNUM predicates that may have been introduced by previous repairs
    stripped = re.sub(r"\bAND\s+ROWNUM\s*(?:<=|<|=)\s*\d+\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\bROWNUM\s*(?:<=|<|=)\s*\d+\s*", "", stripped, flags=re.IGNORECASE)
    normalized = stripped.lower()
    if dialect.startswith("oracle"):
        if "fetch first" in normalized:
            return True, stripped.rstrip(";")
        return True, f"{stripped.rstrip(';')} FETCH FIRST {row_cap} ROWS ONLY"
    if "fetch first" in normalized:
        base = stripped.lower().split("fetch first")[0].rstrip(";")
        return True, f"{base} LIMIT {row_cap}"
    if dialect in {"postgresql", "postgres", "mysql", "mariadb", "sqlite"}:
        if "limit" in normalized:
            return True, stripped
        return True, f"{stripped.rstrip(';')} LIMIT {row_cap}"
    if dialect in {"mssql", "sqlserver"}:
        if "top" in normalized.split():
            return True, stripped
        parsed = sqlparse.parse(stripped)
        if not parsed:
            return False, stripped
        statement = parsed[0]
        tokens = list(statement.tokens)
        for index, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT":
                tokens.insert(index + 1, sqlparse.sql.Token(sqlparse.tokens.Keyword, f" TOP ({row_cap})"))
                break
        return True, "".join(token.value for token in tokens)
    return True, stripped


def validate_sql(
    sql: str,
    graph_context: GraphContext,
    *,
    row_cap: int,
    dialect: str,
    sensitive_columns: Sequence[str],
) -> Tuple[bool, List[str], str]:
    errors: List[str] = []
    patched_sql = sql
    tables = extract_tables(sql)
    known_tables = {card.name.lower() for card in graph_context.tables}
    table_card_lookup = {card.name.lower(): card for card in graph_context.tables}

    def _is_known(table: str) -> bool:
        normalized = table.lower()
        bare = normalized.split(".")[-1]
        return (
            bare in known_tables
            or normalized in known_tables
            or bare in METADATA_TABLES
            or normalized in METADATA_TABLES
            or normalized in METADATA_TABLES_QUALIFIED
        )

    missing = [table for table in tables if not _is_known(table)]
    if missing:
        errors.append(f"Unknown tables referenced: {', '.join(missing)}")

    if contains_select_star(sql):
        if len(tables) == 1:
            target_table = tables[0].split(".")[-1].lower()
            card = table_card_lookup.get(target_table)
            if card:
                for column in card.columns:
                    pattern = re.compile(rf"\b{re.escape(column.name)}\b", re.IGNORECASE)
                    if pattern.search(sql):
                        errors.append("Avoid SELECT *; list columns explicitly.")
                        break
            else:
                errors.append("Avoid SELECT *; list columns explicitly.")
        else:
            errors.append("Avoid SELECT *; list columns explicitly.")

    if contains_sensitive_columns(sql, sensitive_columns):
        errors.append("Query selects sensitive columns which are not allowed.")

    _ok, patched_sql = enforce_row_cap(patched_sql, row_cap, dialect)
    if not _ok:
        errors.append("Unable to guarantee row cap enforcement.")

    patched_sql, literal_errors = _align_literals_with_samples(patched_sql, graph_context)
    errors.extend(literal_errors)

    return (not errors, errors, patched_sql)


METADATA_TABLES = {
    "user_tables",
    "all_tables",
    "dba_tables",
    "user_tab_columns",
    "all_tab_columns",
    "dba_tab_columns",
    "user_views",
    "all_views",
    "dual",
}


METADATA_TABLES_QUALIFIED = {
    "sys.user_tables",
    "sys.all_tables",
    "sys.dba_tables",
    "sys.user_tab_columns",
    "sys.all_tab_columns",
    "sys.dba_tab_columns",
}


def repair_sql(
    sql: str,
    schema: Optional[str],
    row_cap: int,
    dialect: str,
) -> str:
    patched_sql = sql
    if schema:
        for table in extract_tables(sql):
            normalized = table.lower()
            bare = normalized.split(".")[-1]
            if "." in table or bare in METADATA_TABLES or normalized in METADATA_TABLES_QUALIFIED:
                continue
            pattern = re.compile(rf"\b{re.escape(table)}\b", re.IGNORECASE)
            patched_sql = pattern.sub(lambda match: f"{schema}.{match.group(0)}", patched_sql)
    _ok, patched_sql = enforce_row_cap(patched_sql, row_cap, dialect)
    if _ok:
        return patched_sql
    return sql

