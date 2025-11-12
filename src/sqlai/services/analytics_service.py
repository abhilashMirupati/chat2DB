"""
High level service for answering natural language questions about databases.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage

from sqlai.agents.query_agent import (
    create_query_workflow,
    format_schema_markdown,
    _profile_dataframe,
)
from sqlai.config import (
    DatabaseConfig,
    EmbeddingConfig,
    LLMConfig,
    load_app_config,
    load_database_config,
    load_embedding_config,
    load_llm_config,
)
from sqlai.database.connectors import build_sql_database, create_db_engine, test_connection
from sqlai.database.schema_introspector import (
    ForeignKeyDetail,
    TableSummary,
    introspect_database,
    _format_fk_detail,
)
from sqlai.graph.context import GraphContext, TableCard, build_graph_context
from sqlai.semantic.retriever import RetrievalResult, SemanticRetriever
from sqlai.llm.providers import LLMProviderError, load_chat_model
from sqlai.services.metadata_cache import MetadataCache
from sqlai.services.conversation_cache import ConversationCache
from sqlai.utils.logging import get_logger

LOGGER = get_logger(__name__)


class AnalyticsService:
    """
    Entry point used by the UI and CLI to interact with the agent.
    """

    def __init__(
        self,
        db_config: DatabaseConfig | None = None,
        llm_config: LLMConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self.db_config = db_config or load_database_config()
        self.llm_config = llm_config or load_llm_config()
        self.embedding_config = embedding_config or load_embedding_config()
        self.semantic_retriever = SemanticRetriever(self.embedding_config)
        self.app_config = load_app_config()
        self.metadata_cache = MetadataCache(self.app_config.cache_dir / "table_metadata.db")
        self.conversation_cache = ConversationCache(self.app_config.cache_dir / "conversation_history.db")
        self.table_hashes: Dict[str, str] = {}

        self.engine = create_db_engine(self.db_config)
        connectivity_error = test_connection(self.engine)
        if connectivity_error:
            raise ConnectionError(f"Database connection failed: {connectivity_error}")

        self.dialect = (self.engine.dialect.name or "").lower()
        self.dialect_hint = self._build_dialect_hint()

        try:
            self.llm = load_chat_model(self.llm_config)
        except LLMProviderError as exc:
            raise RuntimeError(f"Failed to load LLM provider: {exc}") from exc

        self._llm_lock = threading.Lock()

        self.schema_summaries = introspect_database(
            self.engine,
            schema=self.db_config.schema,
            include_system_tables=self.db_config.include_system_tables,
        )
        LOGGER.info("Loaded %s tables for schema introspection.", len(self.schema_summaries))

        self._hydrate_table_metadata()

        self.graph_context = build_graph_context(
            summaries=self.schema_summaries,
            schema=self.db_config.schema,
        )
        self.row_cap = self.db_config.sample_row_limit

        self.workflow = create_query_workflow(self.llm, self.engine)
        self.sql_database = build_sql_database(self.engine, self.db_config)
        self.conversation: List[Dict[str, Any]] = []

    def schema_markdown(self) -> str:
        if not self.schema_summaries:
            return ""
        summaries = [
            {
                "table": summary.name,
                "columns": ", ".join(column.name for column in summary.columns),
                "foreign_keys": ", ".join(_format_fk_detail(fk) for fk in summary.foreign_keys),
                "row_estimate": summary.row_estimate or "",
            }
            for summary in self.schema_summaries
        ]
        return format_schema_markdown(summaries)

    def has_schema(self) -> bool:
        return bool(self.schema_summaries)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow end-to-end.
        """

        if not self.schema_summaries:
            raise ValueError(
                "No tables were discovered for the configured schema. "
                "Try specifying a schema or enabling system tables in the sidebar."
            )

        LOGGER.info("Received question: %s", question)
        retrieval = self.semantic_retriever.select_cards(
            graph=self.graph_context,
            question=question,
        )
        LOGGER.debug("Retrieval details: %s", retrieval.details)
        value_anchor_text = self._build_value_anchors(retrieval.tables)

        prompt_inputs = self.graph_context.prepare_prompt_inputs(
            question=question,
            dialect_guide=self.dialect_hint,
            token_budget=4096,
            row_cap=self.row_cap,
            tables=retrieval.tables,
            columns=retrieval.columns,
            relationships=retrieval.relationships,
        )
        prompt_inputs["value_anchors"] = value_anchor_text
        prompt_inputs["retrieval_details"] = json.dumps(retrieval.details, indent=2)
        prompt_inputs["analysis_hints"] = self._question_hints(question)
        metadata_snapshot = self.graph_context.serialize_metadata()
        prompt_inputs["user_question"] = question

        result_state = self.workflow.invoke(
            {
                "question": question,
                "prompt_inputs": prompt_inputs,
                "schema_markdown": self.schema_markdown(),
                "dialect_hint": self.dialect_hint,
                "schema_name": self.db_config.schema or "",
                "dialect": self.dialect,
                "row_cap": self.row_cap,
                "graph_metadata": metadata_snapshot,
                "graph_context": self.graph_context,
            }
        )
        LOGGER.debug("Workflow result state: %s", result_state)
        response = {
            "answer": result_state["answer"]["text"],
            "chart": result_state["answer"].get("chart"),
            "followups": result_state["answer"].get("followups") or [],
            "plan": result_state.get("plan"),
            "execution_error": result_state.get("execution_error"),
            "executions": [
                {
                    "sql": execution.sql,
                    "data": execution.dataframe.to_dict(orient="records"),
                    "columns": list(execution.dataframe.columns),
                    "preview": execution.preview_markdown,
                    "row_count": execution.row_count,
                    "stats": execution.stats,
                }
                for execution in result_state.get("executions", [])
            ],
        }
        self._record_conversation(question, response)
        self._persist_saved_query(question, response)
        return response

    def dataframe_for_sql(self, sql: str) -> pd.DataFrame:
        """
        Helper to execute raw SQL and return a DataFrame.
        """

        LOGGER.debug("Executing ad-hoc SQL: %s", sql)
        return pd.read_sql_query(sql, self.engine)

    def get_conversation(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.conversation[-limit:]

    def list_saved_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        schema_key = self._schema_key() or "(default)"
        return self.conversation_cache.list_interactions(schema_key, limit=limit)

    def execute_saved_query(self, entry_id: int) -> Dict[str, Any]:
        record = self.conversation_cache.get_interaction(entry_id)
        if not record:
            raise ValueError(f"No saved query found for id {entry_id}")
        dataframe = pd.read_sql_query(record["sql"], self.engine)
        preview = dataframe.head(20).to_markdown(index=False) if not dataframe.empty else "No rows."
        plan = record.get("plan") or {}
        chart = record.get("chart")
        summary = record.get("summary") or "Replayed cached SQL."
        row_count = len(dataframe)
        stats = _profile_dataframe(dataframe)
        result = {
            "question": record["question"],
            "answer": summary,
            "chart": chart,
            "followups": [],
            "plan": plan,
            "execution_error": None,
            "executions": [
                {
                    "sql": record["sql"],
                    "data": dataframe.to_dict(orient="records"),
                    "columns": list(dataframe.columns),
                    "preview": preview,
                    "row_count": row_count,
                    "stats": stats,
                }
            ],
        }
        self._record_conversation(record["question"], result, cached=True)
        return result

    def _question_hints(self, question: str) -> str:
        if not question:
            return "None"
        q = question.lower()
        hints: List[str] = []
        if any(keyword in q for keyword in ["how many", "count", "total", "number of", "volume"]):
            hints.append("Use COUNT(*) or SUM(...) with GROUP BY to report totals instead of raw detail rows.")
        if any(keyword in q for keyword in ["top", "highest", "most", "max", "largest", "biggest"]):
            hints.append("Order results descending by the relevant metric and limit to the top results.")
        if "failure" in q and ("reason" in q or "root cause" in q):
            hints.append("Group results by failure reason/status and include counts per reason.")
        if "pass" in q and "fail" in q and ("how many" in q or "count" in q):
            hints.append("Aggregate counts per status (PASS vs FAIL).")
        return " ".join(hints) if hints else "None"

    def _record_conversation(self, question: str, result: Dict[str, Any], cached: bool = False) -> None:
        entry = {
            "question": question,
            "answer": result.get("answer"),
            "plan": result.get("plan"),
            "execution_error": result.get("execution_error"),
            "cached": cached,
        }
        self.conversation.append(entry)

    def _persist_saved_query(self, question: str, response: Dict[str, Any]) -> None:
        if response.get("execution_error"):
            return
        executions = response.get("executions") or []
        if not executions:
            return
        sql_text = executions[0]["sql"]
        plan = response.get("plan") or {}
        plan_sql = plan.get("sql")
        if isinstance(plan_sql, list):
            sql_text = plan_sql[0] or sql_text
        elif isinstance(plan_sql, str) and plan_sql.strip():
            sql_text = plan_sql
        summary = response.get("answer")
        if isinstance(summary, dict):
            summary_text = summary.get("text") or ""
        else:
            summary_text = summary or ""
        schema_key = self._schema_key() or "(default)"
        try:
            self.conversation_cache.save_interaction(
                schema=schema_key,
                question=question,
                sql_text=sql_text,
                plan=plan,
                summary=summary_text,
                chart=response.get("chart"),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to persist saved query: %s", exc)

    def _hydrate_table_metadata(self) -> None:
        schema_key = self._schema_key()
        summaries = self.schema_summaries
        total_tables = len(summaries)
        if total_tables == 0:
            LOGGER.info("No tables detected during metadata hydration.")
            return

        LOGGER.info(
            "Hydrating metadata for %s tables in schema '%s'.",
            total_tables,
            schema_key or "(default)",
        )

        self.table_hashes = {summary.name: self._table_schema_hash(summary) for summary in summaries}

        def process_summary(summary: TableSummary) -> TableSummary:
            table_name = summary.name
            schema_hash = self.table_hashes[table_name]
            LOGGER.debug("Starting metadata hydration for table '%s'", table_name)
            try:
                cached = self.metadata_cache.fetch(schema_key, table_name)
                cached_hash = cached.get("schema_hash") if cached else None
                cache_hit = bool(cached and cached_hash == schema_hash)
                LOGGER.debug(
                    "Cache lookup for '%s': hit=%s cached_hash=%s current_hash=%s",
                    table_name,
                    cache_hit,
                    cached_hash,
                    schema_hash,
                )
                if cache_hit and cached:
                    samples: Dict[str, List[str]] = cached.get("samples") or {}
                    description: Optional[str] = cached.get("description")
                else:
                    samples = {}
                    description = None

                if not description:
                    LOGGER.debug("Generating description for table '%s'", table_name)
                    description = self._generate_table_description(summary)
                else:
                    LOGGER.debug("Reusing cached description for table '%s'", table_name)

                if not samples:
                    LOGGER.debug(
                        "Collecting column samples for table '%s' (%s columns)",
                        table_name,
                        len(summary.columns),
                    )
                    samples = self._collect_column_samples(summary)
                    LOGGER.debug(
                        "Collected samples for table '%s': %s",
                        table_name,
                        ", ".join(samples.keys()) or "<none>",
                    )
                else:
                    LOGGER.debug("Reusing cached samples for table '%s'", table_name)

                self.metadata_cache.upsert(schema_key, table_name, schema_hash, description, samples)
                LOGGER.debug("Persisted metadata for table '%s' into cache.", table_name)

                columns = [
                    replace(column, sample_values=samples.get(column.name))
                    for column in summary.columns
                ]
                hydrated_summary = replace(summary, columns=columns, description=description)
                LOGGER.debug("Finished metadata hydration for table '%s'", table_name)
                return hydrated_summary
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to hydrate metadata for table '%s': %s", table_name, exc)
                return summary

        if total_tables > 50:
            max_workers = min(5, total_tables)
            LOGGER.info(
                "Using ThreadPoolExecutor with %s worker(s) for metadata hydration.",
                max_workers,
            )
            updated_map: Dict[str, TableSummary] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(process_summary, summary): (idx, summary.name)
                    for idx, summary in enumerate(summaries)
                }
                for future in as_completed(future_map):
                    idx, table_name = future_map[future]
                    try:
                        updated_map[table_name] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception(
                            "Unhandled exception hydrating table '%s': %s",
                            table_name,
                            exc,
                        )
                        updated_map[table_name] = summaries[idx]
            updated_summaries = [
                updated_map.get(summary.name, summary) for summary in summaries
            ]
        else:
            updated_summaries = [process_summary(summary) for summary in summaries]

        self.schema_summaries = updated_summaries
        LOGGER.info("Metadata hydration complete for %s table(s).", total_tables)

    def _schema_key(self) -> str:
        if self.db_config.schema:
            return self.db_config.schema
        default_schema = getattr(self.engine.dialect, "default_schema_name", None)
        if isinstance(default_schema, str):
            return default_schema
        if callable(default_schema):
            try:
                return default_schema(self.engine)
            except TypeError:
                return default_schema()
        return ""

    def _table_schema_hash(self, summary) -> str:
        payload_parts = [
            f"{column.name}:{column.type}:{column.nullable}"
            for column in summary.columns
        ]
        if summary.foreign_keys:
            payload_parts.extend(
                [f"fk:{fk}" for fk in summary.foreign_keys]
            )
        payload = "|".join(payload_parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _generate_table_description(self, summary) -> str:
        column_lines = "\n".join(
            f"- {column.name}: type={column.type}, nullable={column.nullable}, default={column.default}"
            for column in summary.columns
        )
        fk_text = (
            "\n".join(f"- {_format_fk_detail(fk)}" for fk in summary.foreign_keys)
            if summary.foreign_keys
            else "none"
        )
        prompt = (
            "You are documenting relational database tables for analytics engineers. "
            "Write a concise 1-2 sentence description of the table's purpose and the kind of records it contains. "
            "Focus on business meaning, not SQL syntax. Avoid repeating column names unless necessary.\n\n"
            f"Table name: {summary.name}\n"
            f"Columns:\n{column_lines}\n"
            f"Foreign keys:\n{fk_text}\n"
        )
        description = None
        try:
            with self._llm_lock:
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a senior data analyst documenting a data warehouse. "
                                "Be accurate, concise, and use business-friendly language."
                            )
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
            description = getattr(response, "content", None)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("LLM description generation failed for %s: %s", summary.name, exc)
            description = None
        if description:
            return description.strip()
        # Fallback deterministic description
        column_descriptions = ", ".join(
            f"{column.name} ({column.type})" for column in summary.columns
        )
        fk_description = "; ".join(_format_fk_detail(fk) for fk in summary.foreign_keys) if summary.foreign_keys else "none"
        return (
            f"Table {summary.name} contains columns {column_descriptions}. "
            f"Foreign keys: {fk_description}."
        )

    def _collect_column_samples(self, summary) -> Dict[str, List[str]]:
        qualified_table = self._qualify_table(summary.name)
        samples: Dict[str, List[str]] = {}
        for column in summary.columns:
            try:
                query = self._sample_values_query(qualified_table, column.name)
                df = pd.read_sql_query(query, self.engine)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug(
                    "Distinct sample fetch failed for %s.%s: %s",
                    qualified_table,
                    column.name,
                    exc,
                )
                continue
            if df.empty:
                continue
            values = (
                df["value"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if values:
                samples[column.name] = values[:5]
        return samples

    def _build_value_anchors(self, tables: Sequence[TableCard]) -> str:
        anchors: List[str] = []
        column_samples: Dict[str, Dict[str, List[str]]] = {}
        for table_card in tables[:2]:
            qualified_table = self._qualify_table(table_card.name)
            query = self._sample_query(qualified_table, limit=3)
            try:
                sample_df = pd.read_sql_query(query, self.engine)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Value anchor query failed for %s: %s", qualified_table, exc)
                continue
            if sample_df.empty:
                continue
            table_sample_map = column_samples.setdefault(table_card.name, {})
            for column in sample_df.columns:
                unique_values = (
                    sample_df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if unique_values:
                    table_sample_map[column] = unique_values[:5]
            anchors.append(
                f"{qualified_table} sample rows:\n{sample_df.head(3).to_markdown(index=False)}"
            )
        if column_samples:
            schema_key = self._schema_key()
            for table_name, samples in column_samples.items():
                schema_hash = self.table_hashes.get(table_name)
                if schema_hash:
                    self.metadata_cache.update_samples(schema_key, table_name, schema_hash, samples)
            self.graph_context.update_column_samples(column_samples)
        return "\n\n".join(anchors) if anchors else "No value anchors collected."

    def _sample_query(self, qualified_table: str, limit: int = 3) -> str:
        dialect = self.dialect
        if dialect.startswith("oracle"):
            return f"SELECT * FROM {qualified_table} FETCH FIRST {limit} ROWS ONLY"
        if dialect in {"postgresql", "postgres", "mysql", "mariadb", "sqlite"}:
            return f"SELECT * FROM {qualified_table} LIMIT {limit}"
        if dialect in {"mssql", "sqlserver"}:
            return f"SELECT TOP ({limit}) * FROM {qualified_table}"
        return f"SELECT * FROM {qualified_table}"

    def _sample_values_query(self, qualified_table: str, column_name: str, limit: int = 25) -> str:
        dialect = self.dialect
        if dialect.startswith("oracle"):
            return (
                f"SELECT DISTINCT {column_name} AS value "
                f"FROM {qualified_table} "
                f"WHERE {column_name} IS NOT NULL "
                f"FETCH FIRST {limit} ROWS ONLY"
            )
        if dialect in {"postgresql", "postgres", "mysql", "mariadb", "sqlite"}:
            return (
                f"SELECT DISTINCT {column_name} AS value "
                f"FROM {qualified_table} "
                f"WHERE {column_name} IS NOT NULL "
                f"LIMIT {limit}"
            )
        if dialect in {"mssql", "sqlserver"}:
            return (
                f"SELECT DISTINCT TOP ({limit}) {column_name} AS value "
                f"FROM {qualified_table} "
                f"WHERE {column_name} IS NOT NULL"
            )
        return (
            f"SELECT DISTINCT {column_name} AS value "
            f"FROM {qualified_table} "
            f"WHERE {column_name} IS NOT NULL "
            f"LIMIT {limit}"
        )

    def _qualify_table(self, table: str) -> str:
        if self.db_config.schema and "." not in table:
            return f"{self.db_config.schema}.{table}"
        return table

    def _build_dialect_hint(self) -> str:
        dialect = self.dialect
        schema = self.db_config.schema
        guidance = ["You are working against a relational database."]
        if dialect == "oracle":
            owner_hint = schema.upper() if schema else "YOUR_SCHEMA"
            guidance = [
                "You are working against an Oracle database.",
                "- Use dual for scalar expressions.",
                "- Metadata queries:",
                "  * Current schema tables: SELECT table_name FROM user_tables ORDER BY table_name;",
                f"  * Specific schema tables: SELECT table_name FROM all_tables WHERE owner = '{owner_hint}';",
                f"  * Column info: SELECT column_name, data_type FROM all_tab_columns WHERE owner = '{owner_hint}' AND table_name = 'TABLE';",
                "- Use FETCH FIRST n ROWS ONLY instead of LIMIT.",
            ]
        elif dialect in {"postgresql", "postgres"}:
            guidance = [
                "You are working against a PostgreSQL database.",
                "- Metadata queries:",
                "  * Tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'schema';",
                "  * Columns: SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'schema' AND table_name = 'table';",
                "- Use LIMIT for sampling rows.",
            ]
        elif dialect in {"mysql", "mariadb"}:
            guidance = [
                f"You are working against a {dialect.title()} database.",
                "- Metadata queries:",
                "  * Tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'schema';",
                "  * Columns: SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'schema' AND table_name = 'table';",
                "- Use LIMIT for sampling rows.",
            ]
        elif dialect in {"sqlite"}:
            guidance = [
                "You are working against a SQLite database.",
                "- Metadata queries:",
                "  * Tables: SELECT name FROM sqlite_master WHERE type = 'table';",
                "  * Columns: PRAGMA table_info('table');",
            ]
        elif dialect in {"mssql", "sqlserver"}:
            guidance = [
                "You are working against a SQL Server database.",
                "- Metadata queries:",
                "  * Tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'schema';",
                "  * Columns: SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'schema' AND table_name = 'table';",
                "- Use TOP (n) for limiting rows.",
            ]
        else:
            guidance.append(f"The dialect reported by SQLAlchemy is '{dialect}'. Use ANSI SQL when unsure.")
        guidance.append("- Always produce a SQL statement that answers the question.")
        return "\n".join(guidance)


def _format_fk_detail(detail: ForeignKeyDetail) -> str:
    left = ",".join(detail.constrained_columns)
    right_schema = f"{detail.referred_schema}." if detail.referred_schema else ""
    right = ",".join(detail.referred_columns)
    return f"[{left}] -> {right_schema}{detail.referred_table}.[{right}]"

