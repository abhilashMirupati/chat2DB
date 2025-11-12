"""
Domain-specific SQL fallback templates.
"""

from __future__ import annotations

import re
from typing import List, Optional

from sqlai.graph.context import GraphContext, TableCard


def maybe_generate_domain_sql(
    question: str,
    graph_context: GraphContext,
    schema: Optional[str],
    dialect: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Provide domain-specific fallback SQL when the planner returns nothing.
    Currently supports failure reason aggregation and daily pass counts.
    """

    lower_question = question.lower()
    tables = {card.name.lower(): card for card in graph_context.tables}

    schema_prefix = f"{schema}." if schema else ""

    has_executions = "executions" in tables
    has_test_cases = "test_cases" in tables
    has_test_sets = "test_sets" in tables

    # Failure reason aggregation
    if has_executions and has_test_cases and has_test_sets and "failure" in lower_question:
        return [
            (
                "SELECT ts.name AS test_set_name, "
                "e.failure_reason, "
                "COUNT(*) AS failure_count "
                f"FROM {schema_prefix}executions e "
                f"JOIN {schema_prefix}test_cases tc ON tc.id = e.test_case_id "
                f"JOIN {schema_prefix}test_sets ts ON ts.id = tc.test_set_id "
                "WHERE e.status = 'FAILED' "
                "GROUP BY ts.name, e.failure_reason "
                "ORDER BY failure_count DESC "
                "FETCH FIRST 5 ROWS ONLY"
            )
        ]

    # Daily pass / fail totals
    if has_executions and "today" in lower_question and ("pass" in lower_question or "success" in lower_question):
        date_condition = _build_today_condition(dialect or "")
        return [
            (
                "SELECT "
                "SUM(CASE WHEN status = 'PASSED' THEN 1 ELSE 0 END) AS passed_count, "
                "SUM(CASE WHEN status <> 'PASSED' THEN 1 ELSE 0 END) AS failed_count "
                f"FROM {schema_prefix}executions "
                f"WHERE {date_condition}"
            )
        ]

    # Generic pass count
    if has_executions and re.search(r"\bhow many\b.*\bpass", lower_question):
        return [
            (
                "SELECT status, COUNT(*) AS occurrence "
                f"FROM {schema_prefix}executions "
                "GROUP BY status "
                "ORDER BY occurrence DESC"
            )
        ]

    return None


def _build_today_condition(dialect: str) -> str:
    dialect = dialect.lower()
    if dialect.startswith("oracle"):
        return "TRUNC(run_at) = TRUNC(SYSDATE)"
    if dialect in {"postgresql", "postgres"}:
        return "run_at::date = CURRENT_DATE"
    if dialect in {"mysql", "mariadb"}:
        return "DATE(run_at) = CURRENT_DATE"
    if dialect in {"sqlite"}:
        return "DATE(run_at) = DATE('now')"
    if dialect in {"mssql", "sqlserver"}:
        return "CAST(run_at AS DATE) = CAST(GETDATE() AS DATE)"
    return "run_at >= CURRENT_DATE"

