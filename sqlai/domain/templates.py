"""
Domain-specific SQL fallback templates for common analytics questions.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def maybe_generate_domain_sql(
    question: str,
    metadata: Optional[Dict[str, object]],
    schema: Optional[str],
) -> Optional[List[str]]:
    if not metadata or not isinstance(metadata, dict):
        return None
    question_lower = question.lower()
    if "failure" in question_lower and "reason" in question_lower:
        sql = _failure_reason_fallback(metadata, schema)
        if sql:
            return [sql]
    return None


def _failure_reason_fallback(metadata: Dict[str, object], schema: Optional[str]) -> Optional[str]:
    tables = metadata.get("tables") or []
    if not isinstance(tables, list):
        return None

    failure_table = None
    failure_column = None
    for table in tables:
        table_name = (table.get("name") or "").lower()
        for column in table.get("columns") or []:
            col_name = (column.get("name") or "").lower()
            if "failure" in col_name and "reason" in col_name:
                failure_table = table
                failure_column = column
                break
        if failure_table:
            break
    if not failure_table or not failure_column:
        return None

    qualified_failure_table = _qualify_table(schema or failure_table.get("schema"), failure_table["name"])
    failure_col_name = failure_column["name"]

    # Attempt to find a test set table for richer context.
    test_set_table = None
    for table in tables:
        if "test_set" in (table.get("name") or "").lower():
            test_set_table = table
            break
    if not test_set_table:
        # Fall back to global failure reason counts.
        return (
            f"SELECT {failure_col_name} AS failure_reason, COUNT(*) AS failure_count "
            f"FROM {qualified_failure_table} "
            f"WHERE {failure_col_name} IS NOT NULL "
            f"GROUP BY {failure_col_name} "
            f"ORDER BY failure_count DESC"
        )

    test_case_table = None
    for table in tables:
        if "test_case" in (table.get("name") or "").lower():
            test_case_table = table
            break
    if not test_case_table:
        return (
            f"SELECT {failure_col_name} AS failure_reason, COUNT(*) AS failure_count "
            f"FROM {qualified_failure_table} "
            f"WHERE {failure_col_name} IS NOT NULL "
            f"GROUP BY {failure_col_name} "
            f"ORDER BY failure_count DESC"
        )

    failure_case_col = _find_column(failure_table, ["test_case", "id"])
    case_pk = _find_column(test_case_table, ["id"]) or _find_column(test_case_table, ["test_case", "id"])
    case_set_col = _find_column(test_case_table, ["test_set"])
    set_pk = _find_column(test_set_table, ["id"])
    set_name_col = _find_column(test_set_table, ["name"])

    if not all([failure_case_col, case_pk, case_set_col, set_pk, set_name_col]):
        return (
            f"SELECT {failure_col_name} AS failure_reason, COUNT(*) AS failure_count "
            f"FROM {qualified_failure_table} "
            f"WHERE {failure_col_name} IS NOT NULL "
            f"GROUP BY {failure_col_name} "
            f"ORDER BY failure_count DESC"
        )

    qualified_case_table = _qualify_table(schema or test_case_table.get("schema"), test_case_table["name"])
    qualified_set_table = _qualify_table(schema or test_set_table.get("schema"), test_set_table["name"])

    failure_alias = "f"
    case_alias = "tc"
    set_alias = "ts"

    return (
        f"SELECT {set_alias}.{set_name_col['name']} AS test_set_name, "
        f"{failure_alias}.{failure_col_name} AS failure_reason, "
        f"COUNT(*) AS failure_count "
        f"FROM {qualified_failure_table} {failure_alias} "
        f"JOIN {qualified_case_table} {case_alias} "
        f"ON {failure_alias}.{failure_case_col['name']} = {case_alias}.{case_pk['name']} "
        f"JOIN {qualified_set_table} {set_alias} "
        f"ON {case_alias}.{case_set_col['name']} = {set_alias}.{set_pk['name']} "
        f"WHERE {failure_alias}.{failure_col_name} IS NOT NULL "
        f"GROUP BY {set_alias}.{set_name_col['name']}, {failure_alias}.{failure_col_name} "
        f"ORDER BY failure_count DESC"
    )


def _find_column(table_meta: Dict[str, object], keywords: List[str]) -> Optional[Dict[str, object]]:
    for column in table_meta.get("columns") or []:
        name = (column.get("name") or "").lower()
        if all(keyword in name for keyword in keywords):
            return column
    return None


def _qualify_table(schema: Optional[str], table_name: str) -> str:
    if schema:
        return f"{schema}.{table_name}"
    return table_name

