"""
LangGraph workflow orchestrating question -> SQL -> analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:
    raise RuntimeError(
        "LangGraph is required to run the SQLAI agent. Install it with `pip install langgraph`."
    ) from exc
from sqlalchemy.engine import Engine

from sqlai.llm.prompt_templates import (
    agent_prompt,
    CRITIC_PROMPT,
    REPAIR_PROMPT,
    INTENT_CRITIC_PROMPT,
)
from sqlai.agents.guard import repair_sql, validate_sql
from sqlai.graph.context import GraphContext
from sqlai.services.domain_fallbacks import maybe_generate_domain_sql

LOGGER = logging.getLogger(__name__)

MAX_INTENT_REPAIRS = 5


class QueryState(TypedDict, total=False):
    question: str
    prompt_inputs: Dict[str, Any]
    schema_markdown: str
    dialect_hint: str
    schema_name: str
    dialect: str
    row_cap: Optional[int]
    intent_critic: Dict[str, Any]
    intent_attempts: int
    plan: Dict[str, Any]
    executions: List["ExecutionResult"]
    answer: Dict[str, Any]
    execution_error: Optional[str]
    graph_context: Optional[GraphContext]
    critic: Dict[str, Any]
    repair_attempts: int


def _route_after_intent(state: QueryState) -> str:
    critic = state.get("intent_critic") or {}
    verdict = (critic.get("verdict") or "").lower()
    attempts = state.get("intent_attempts", 0)
    if verdict == "reject":
        if attempts >= MAX_INTENT_REPAIRS:
            reasons = critic.get("reasons") or []
            sql = state.get("plan", {}).get("sql")
            message = (
                "Unable to produce SQL that satisfies the question after "
                f"{attempts} intent repair attempt(s).\n"
                f"Reasons: {reasons}\n\nLast SQL:\n{sql}"
            )
            state["execution_error"] = message
            return "fail"
        return "repair"
    return "execute"


@dataclass
class ExecutionResult:
    sql: str
    dataframe: pd.DataFrame
    preview_markdown: str
    row_count: int
    stats: Dict[str, Any]


def _profile_dataframe(df: pd.DataFrame, max_top_values: int = 5) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"row_count": int(len(df)) if df is not None else 0, "columns": []}
    if df is None or df.empty:
        return summary
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        column_summary: Dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "unique": unique,
        }
        if unique <= max_top_values or series.dtype == "object":
            top_values = (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(max_top_values)
            )
            if not top_values.empty:
                column_summary["top_values"] = [
                    (value, int(count)) for value, count in top_values.items()
                ]
        summary["columns"].append(column_summary)
    return summary


def _format_execution_stats(executions: Sequence[ExecutionResult], row_cap: Optional[int]) -> str:
    if not executions:
        return "No executions to summarise."
    lines: List[str] = []
    for idx, execution in enumerate(executions, start=1):
        capped_note = ""
        if row_cap and execution.row_count >= row_cap:
            capped_note = f" (Reached row cap {row_cap}; results may be truncated.)"
        lines.append(f"Result {idx}: {execution.row_count} row(s){capped_note}")
        informative_columns = [
            column_summary
            for column_summary in execution.stats.get("columns", [])
            if column_summary.get("top_values")
        ][:3]
        for column_summary in informative_columns:
            top_values = ", ".join(
                f"{value} ({count})"
                for value, count in column_summary.get("top_values", [])[:3]
            )
            lines.append(f"  - {column_summary['name']}: top values {top_values}")
    return "\n".join(lines)


def format_schema_markdown(schema: List[Dict[str, Any]]) -> str:
    """
    Convert table summaries into markdown for the prompt.
    """

    lines = []
    for summary in schema:
        lines.append(f"### {summary['table']}")
        lines.append(f"- columns: {summary['columns']}")
        if summary.get("foreign_keys"):
            lines.append(f"- foreign_keys: {summary['foreign_keys']}")
        if summary.get("row_estimate"):
            lines.append(f"- row_estimate: {summary['row_estimate']}")
    return "\n".join(lines)


def create_query_workflow(llm: Any, engine: Engine) -> Any:
    """
    Compile a LangGraph for the SQL query workflow.
    """

    graph = StateGraph(QueryState)
    graph.add_node("plan", _plan_sql(llm))
    graph.add_node("intent_critic", _intent_critic(llm))
    graph.add_node("intent_repair", _intent_repair(llm))
    graph.add_node("execute", _execute_sql(engine))
    graph.add_node("critic", _critic_sql(llm))
    graph.add_node("repair", _repair_sql(llm))
    graph.add_node("summarise", _summarise(llm))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "intent_critic")
    graph.add_conditional_edges(
        "intent_critic",
        _route_after_intent,
        {
            "repair": "intent_repair",
            "execute": "execute",
            "fail": "summarise",
        },
    )
    graph.add_edge("intent_repair", "intent_critic")
    graph.add_conditional_edges(
        "execute",
        _needs_critique,
        {True: "critic", False: "summarise"},
    )
    graph.add_conditional_edges(
        "critic",
        _needs_repair,
        {True: "repair", False: "summarise"},
    )
    graph.add_edge("repair", "execute")
    graph.add_edge("summarise", END)

    return graph.compile()


def _plan_sql(llm: Any):
    parser = JsonOutputParser()
    prompt = agent_prompt()
    chain = prompt | llm | parser

    def node(state: QueryState) -> QueryState:
        LOGGER.debug("Planning SQL for question=%s", state.get("question"))
        prompt_inputs = dict(state.get("prompt_inputs") or {})
        prompt_inputs.setdefault("user_question", state.get("question", ""))
        plan = chain.invoke(prompt_inputs)
        if isinstance(plan, dict):
            sql_statements = plan.get("sql")
            if not sql_statements and plan.get("rationale"):
                plan["sql_generation_note"] = "LLM provided a rationale but no SQL."
            else:
                plan = _post_process_plan(plan, state)
            LOGGER.debug("LLM plan: %s", plan)
        return {
            "plan": plan,
            "prompt_inputs": prompt_inputs,
            "critic": {},
            "repair_attempts": 0,
            "intent_critic": {},
            "intent_attempts": 0,
            "execution_error": None,
        }

    return node


def _intent_critic(llm: Any):
    parser = JsonOutputParser()
    chain = INTENT_CRITIC_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if isinstance(sql, list):
            sql_text = "\n".join(sql)
        else:
            sql_text = sql or ""
        plan_summary = plan.get("plan") or {}
        result = chain.invoke(
            {
                "question": state.get("question", ""),
                "plan": plan_summary,
                "sql": sql_text,
            }
        )
        LOGGER.debug("Intent critic verdict: %s", result)
        state["intent_critic"] = result
        # Clear any previous execution error when we are re-evaluating intent
        state.pop("execution_error", None)
        return state

    return node


def _intent_repair(llm: Any):
    parser = JsonOutputParser()
    chain = REPAIR_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if isinstance(sql, list):
            sql_text = sql[0]
        else:
            sql_text = sql or ""
        critic = state.get("intent_critic") or {}
        attempts = state.get("intent_attempts", 0)
        error_text = "; ".join(critic.get("reasons", []))
        result = chain.invoke(
            {
                "dialect": state.get("dialect", ""),
                "sql": sql_text,
                "error": error_text,
                "repair_hints": critic.get("repair_hints", []),
            }
        )
        LOGGER.debug("Intent repair attempt %s: %s", attempts + 1, result)
        patched_sql = result.get("patched_sql")
        if patched_sql:
            plan["sql"] = patched_sql
            plan.setdefault("notes", []).append(
                f"Intent repair iteration {attempts + 1} applied."
            )
            state["plan"] = plan
        state["intent_attempts"] = attempts + 1
        state["intent_critic"] = {}
        state.pop("execution_error", None)
        return state

    return node


def _execute_sql(engine: Engine):
    def node(state: QueryState) -> QueryState:
        plan = state["plan"]
        sql_statements = plan.get("sql")
        executions: List[ExecutionResult] = []
        if isinstance(sql_statements, str):
            sql_statements = [sql_statements]
        if not sql_statements:
            fallback_sql = _metadata_fallback(state)
            if fallback_sql:
                plan = dict(plan or {})
                plan.setdefault("notes", []).append("Applied metadata fallback query.")
                plan["sql"] = fallback_sql if len(fallback_sql) > 1 else fallback_sql[0]
                sql_statements = fallback_sql
            else:
                return {
                    "executions": [],
                    "execution_error": "No SQL generated by the plan.",
                    "plan": plan,
                    "executions_available": False,
                }

        execution_error: Optional[str] = None
        prompt_inputs = state.get("prompt_inputs") or {}
        graph: Optional[GraphContext] = state.get("graph_context")  # type: ignore[assignment]
        sensitive_columns: Sequence[str] = []
        if isinstance(graph, GraphContext):
            sensitive_columns = graph.sensitive_columns

        for idx, sql in enumerate(sql_statements):
            if isinstance(graph, GraphContext):
                is_valid, errors, patched_sql = validate_sql(
                    sql,
                    graph_context=graph,
                    row_cap=state.get("row_cap") or 0,
                    dialect=state.get("dialect") or "",
                    sensitive_columns=sensitive_columns,
                )
                sql = patched_sql
                sql_statements[idx] = sql
                if not is_valid:
                    LOGGER.debug("Guard errors for SQL '%s': %s", sql, errors)
                    repaired = repair_sql(
                        sql,
                        schema=state.get("schema_name"),
                        row_cap=state.get("row_cap") or 0,
                        dialect=state.get("dialect") or "",
                    )
                    if repaired != sql:
                        sql = repaired
                        sql_statements[idx] = sql
                        is_valid, errors, patched_sql = validate_sql(
                            sql,
                            graph_context=graph,
                            row_cap=state.get("row_cap") or 0,
                            dialect=state.get("dialect") or "",
                            sensitive_columns=sensitive_columns,
                        )
                        sql = patched_sql
                        sql_statements[idx] = sql
                if not is_valid:
                    execution_error = "; ".join(errors) or "SQL validation failed."
                    LOGGER.warning("Guard rejected SQL '%s': %s", sql, execution_error)
                    executions.append(
                        ExecutionResult(
                            sql=sql,
                            dataframe=pd.DataFrame(),
                            preview_markdown=f"SQL guard rejected this plan: {execution_error}",
                            row_count=0,
                            stats={"row_count": 0, "columns": []},
                        )
                    )
                    break

            try:
                dataframe = pd.read_sql_query(sql, engine)
            except Exception as exc:  # noqa: BLE001
                execution_error = f"Database execution failed: {exc}"
                LOGGER.warning("Error executing SQL '%s': %s", sql, exc)
                executions.append(
                    ExecutionResult(
                        sql=sql,
                        dataframe=pd.DataFrame(),
                        preview_markdown=f"SQL execution failed: {exc}",
                        row_count=0,
                        stats={"row_count": 0, "columns": []},
                    )
                )
                break
            row_count = len(dataframe)
            stats = _profile_dataframe(dataframe)
            preview = dataframe.head(20).to_markdown(index=False) if not dataframe.empty else "No rows."
            executions.append(
                ExecutionResult(
                    sql=sql,
                    dataframe=dataframe,
                    preview_markdown=preview,
                    row_count=row_count,
                    stats=stats,
                )
            )
        if isinstance(plan.get("sql"), str):
            plan["sql"] = sql_statements[0] if sql_statements else plan.get("sql")
        else:
            plan["sql"] = sql_statements
        return {
            "executions": executions,
            "plan": plan,
            "execution_error": execution_error,
            "executions_available": True,
        }

    return node


def _summarise(llm: Any):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior analytics engineer. Summarise SQL query results for business stakeholders. "
                "Answer precisely and, if charts are requested, ensure the chart specification is consistent "
                "with the data. You must always produce a summary, even if some fields appear missing or look "
                "like template placeholders (e.g. {question}). When information is absent, make a best-effort "
                "inference and explicitly note the gap rather than refusing.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Plan rationale: {rationale}\n\n"
                "SQL:\n{sql}\n\n"
                "Table preview:\n{preview}\n\n"
                "If a chart specification was provided, repeat it and explain it briefly.",
            ),
        ]
    )

    def node(state: QueryState) -> QueryState:
        if state.get("execution_error"):
            plan = state.get("plan", {})
            critic = state.get("critic") or {}
            attempts = state.get("repair_attempts", 0)
            message = state.get("execution_error")
            if critic:
                message = (
                    f"Unable to produce a working SQL after {attempts} repair attempt(s).\n"
                    f"Critic verdict: {critic.get('verdict')}\n"
                    f"Reasons: {critic.get('reasons')}\n\n"
                    f"SQL tried:\n{plan.get('sql')}"
                )
            return {
                "answer": {
                    "text": message,
                    "chart": None,
                    "followups": plan.get("followups") if plan else [],
                }
            }
        executions = state.get("executions") or []
        preview_sections = []
        for idx, execution in enumerate(executions, start=1):
            preview_sections.append(f"Query {idx}:\n{execution.preview_markdown}")
        preview_markdown = "\n\n".join(preview_sections)
        stats_text = _format_execution_stats(executions, state.get("row_cap"))
        combined_preview = f"{preview_markdown}\n\n[DATA SUMMARY]\n{stats_text}"
        sql_text = "\n\n".join(execution.sql for execution in executions)
        plan = state["plan"]
        response = prompt | llm
        summariser_payload = {
            "question": state["question"],
            "rationale": plan.get("rationale_summary") or plan.get("rationale", ""),
            "sql": sql_text,
            "preview": combined_preview,
        }
        LOGGER.debug(
            "Summariser payload: question=%s, rationale(len)=%s, sql(len)=%s, preview(len)=%s",
            summariser_payload["question"],
            len(summariser_payload["rationale"] or ""),
            len(summariser_payload["sql"] or ""),
            len(summariser_payload["preview"] or ""),
        )
        summary = response.invoke(summariser_payload)
        LOGGER.debug("Summariser raw response: %s", getattr(summary, "content", summary))
        summary_text = summary.content if hasattr(summary, "content") else str(summary)
        fallback_needed = (
            "I cannot fulfill this request because you have not provided the actual values" in summary_text
            or "{question}" in summary_text
        )
        if fallback_needed:
            row_total = sum(execution.row_count for execution in executions)
            table_count = len(executions)
            fallback_preview = "No rows returned." if row_total == 0 else f"{row_total} row(s) returned."
            stats_text = _format_execution_stats(executions, state.get("row_cap"))
            summary_text = (
                f"Generated {table_count} result set(s). {fallback_preview} "
                "Showing raw data below; chart generation skipped because the summariser reported missing context.\n\n"
                f"[Auto Summary]\n{stats_text}"
            )
        return {
            "answer": {
                "text": summary_text,
                "chart": plan.get("chart") if not fallback_needed else None,
                "followups": plan.get("followups"),
            }
        }

    return node


def _post_process_plan(plan: Dict[str, Any], state: QueryState) -> Dict[str, Any]:
    sql_statements = plan.get("sql")
    if not sql_statements:
        return plan
    if isinstance(sql_statements, str):
        sql_list = [sql_statements]
    else:
        sql_list = list(sql_statements)
    schema_name = state.get("schema_name") or ""
    schema_upper = schema_name.upper()
    adjusted = []
    changed = False
    applied_all_tables_note = False
    used_columns_override = False
    for sql in sql_list:
        sql_text = sql
        lower_sql = sql.lower()
        if schema_upper and "user_tables" in lower_sql and "all_tables" not in lower_sql:
            sql_text = (
                "SELECT table_name FROM all_tables "
                f"WHERE owner = '{schema_upper}' ORDER BY table_name"
            )
            changed = True
            applied_all_tables_note = True
        adjusted.append(sql_text)

    question_lower = state.get("question", "").lower()
    row_cap = state.get("row_cap") or 0
    dialect = (state.get("dialect") or "").lower()
    if "column" in question_lower:
        if all("all_tab_columns" not in sql.lower() for sql in adjusted):
            if schema_upper:
                base_columns_sql = (
                    "SELECT table_name, column_name, data_type FROM all_tab_columns "
                    f"WHERE owner = '{schema_upper}' ORDER BY table_name, column_id"
                )
            else:
                base_columns_sql = (
                    "SELECT table_name, column_name, data_type FROM user_tab_columns "
                    "ORDER BY table_name, column_id"
                )
            if row_cap and dialect.startswith("oracle"):
                columns_sql = f"{base_columns_sql} FETCH FIRST {row_cap} ROWS ONLY"
            elif row_cap and dialect in {"postgresql", "postgres", "mysql", "mariadb", "sqlite"}:
                columns_sql = f"{base_columns_sql} LIMIT {row_cap}"
            elif row_cap and dialect in {"mssql", "sqlserver"}:
                columns_sql = base_columns_sql.replace(
                    "SELECT table_name, column_name, data_type",
                    f"SELECT TOP ({row_cap}) table_name, column_name, data_type",
                    1,
                )
            else:
                columns_sql = base_columns_sql
            adjusted = [columns_sql]
            changed = True
            used_columns_override = True

    if isinstance(sql_statements, str):
        plan["sql"] = adjusted[0]
    else:
        plan["sql"] = adjusted

    notes: List[str] = plan.get("notes") or []
    plan["notes"] = notes
    if applied_all_tables_note:
        note = "Adjusted metadata query to use ALL_TABLES for schema-specific metadata."
        if note not in notes:
            notes.append(note)
    if used_columns_override:
        note = "Replaced plan with column metadata query using ALL_TAB_COLUMNS."
        if note not in notes:
            notes.append(note)

    return plan


def _metadata_fallback(state: QueryState) -> Optional[List[str]]:
    question = state.get("question", "")
    if not question:
        return None
    graph: Optional[GraphContext] = state.get("graph_context")  # type: ignore[assignment]
    schema = state.get("schema_name")
    if isinstance(graph, GraphContext):
        domain_sql = maybe_generate_domain_sql(
            question,
            graph,
            schema,
            state.get("dialect"),
        )
        if domain_sql:
            return domain_sql

    lower_question = question.lower()
    if "table" not in lower_question:
        return None

    wants_list = any(word in lower_question for word in ["list", "names", "show", "display"]) or "table_name" in lower_question
    wants_count = any(word in lower_question for word in ["count", "how many", "number"])
    if not wants_list and not wants_count:
        return None

    dialect = (state.get("dialect") or "").lower()
    schema_upper = (schema or "").upper()

    sql = _metadata_sql_for_dialect(dialect, schema_upper, "list" if wants_list else "count")
    if not sql:
        return None
    return [sql]


def _metadata_sql_for_dialect(dialect: str, schema: str, mode: str) -> Optional[str]:
    schema_clause = ""
    if schema:
        if dialect == "oracle":
            schema_clause = f" WHERE owner = '{schema}'"
        elif dialect in {"postgresql", "postgres"}:
            schema_clause = f" WHERE table_schema = '{schema.lower()}'"
        elif dialect in {"mysql", "mariadb"}:
            schema_clause = f" WHERE table_schema = '{schema}'"
        elif dialect in {"mssql", "sqlserver"}:
            schema_clause = f" WHERE table_schema = '{schema}'"
        elif dialect == "sqlite":
            # SQLite does not support multiple schemas; ignore
            schema_clause = ""

    if dialect == "oracle":
        if mode == "count":
            return f"SELECT COUNT(*) FROM {'all_tables' if schema else 'user_tables'}{schema_clause}"
        return (
            f"SELECT table_name FROM {'all_tables' if schema else 'user_tables'}"
            f"{schema_clause} ORDER BY table_name"
        )
    if dialect in {"postgresql", "postgres"}:
        base = "FROM information_schema.tables"
        default_filter = " WHERE table_schema NOT IN ('pg_catalog','information_schema')"
        filter_clause = schema_clause if schema_clause else default_filter
        if mode == "count":
            return f"SELECT COUNT(*) {base}{filter_clause}"
        return f"SELECT table_name {base}{filter_clause} ORDER BY table_name"
    if dialect in {"mysql", "mariadb"}:
        base = "FROM information_schema.tables"
        if mode == "count":
            return f"SELECT COUNT(*) {base}{schema_clause or ''}"
        return f"SELECT table_name {base}{schema_clause or ''} ORDER BY table_name"
    if dialect in {"mssql", "sqlserver"}:
        base = "FROM information_schema.tables"
        if mode == "count":
            return f"SELECT COUNT(*) {base}{schema_clause or ''}"
        return f"SELECT table_name {base}{schema_clause or ''} ORDER BY table_name"
    if dialect == "sqlite":
        if mode == "count":
            return "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
        return "SELECT name AS table_name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    # Generic fallback
    base = "FROM information_schema.tables"
    if mode == "count":
        return f"SELECT COUNT(*) {base}"
    return f"SELECT table_name {base} ORDER BY table_name"


def _needs_critique(state: QueryState) -> bool:
    return bool(state.get("execution_error"))


def _needs_repair(state: QueryState) -> bool:
    critic = state.get("critic") or {}
    attempts = state.get("repair_attempts", 0)
    if attempts >= 3:
        return False
    return critic.get("verdict") == "reject"


def _critic_sql(llm: Any):
    parser = JsonOutputParser()
    chain = CRITIC_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if isinstance(sql, list):
            sql_text = "\n".join(sql)
        else:
            sql_text = sql or ""
        plan_summary = {
            "steps": plan.get("plan", {}).get("steps", []),
            "notes": plan.get("plan", {}).get("notes", []),
        }
        result = chain.invoke(
            {
                "sql": sql_text,
                "plan": plan_summary,
                "error": state.get("execution_error", ""),
            }
        )
        LOGGER.debug("SQL critic verdict: %s", result)
        state["critic"] = result
        return state

    return node


def _repair_sql(llm: Any):
    parser = JsonOutputParser()
    chain = REPAIR_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if isinstance(sql, list):
            sql_text = sql[0]
        else:
            sql_text = sql or ""
        critic = state.get("critic") or {}
        attempts = state.get("repair_attempts", 0)
        result = chain.invoke(
            {
                "dialect": state.get("dialect", ""),
                "sql": sql_text,
                "error": state.get("execution_error", ""),
                "repair_hints": critic.get("repair_hints", []),
                "plan": plan,
            }
        )
        LOGGER.debug("SQL repair attempt %s: %s", attempts + 1, result)
        patched_sql = result.get("patched_sql")
        if patched_sql:
            plan["sql"] = patched_sql
            plan.setdefault("notes", []).append(
                f"Applied repair iteration {attempts + 1}."
            )
        state["plan"] = plan
        state["repair_attempts"] = attempts + 1
        state.pop("execution_error", None)
        state["executions"] = []
        return state

    return node

