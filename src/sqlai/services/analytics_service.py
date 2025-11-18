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
    load_vector_store_config,
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
from sqlai.services.graph_cache import GraphCache, GraphCardRecord
from sqlai.services.vector_store import VectorStoreManager
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
        self.vector_config = load_vector_store_config()
        self.app_config = load_app_config()
        self.graph_cache = GraphCache(self.app_config.cache_dir / "graph_cards.db")
        self._validate_runtime_requirements()
        self.vector_store = VectorStoreManager(
            self.vector_config,
            self.app_config.cache_dir,
            self.embedding_config,
        )
        self.semantic_retriever = SemanticRetriever(
            self.embedding_config,
            vector_store=self.vector_store,
        )
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
        self._sync_graph_cache_and_vectors()
        self.row_cap = self.db_config.sample_row_limit

        self.workflow = create_query_workflow(self.llm, self.engine)
        self.sql_database = build_sql_database(self.engine, self.db_config)
        self.conversation: List[Dict[str, Any]] = []

    def schema_markdown(self) -> str:
        """Build schema markdown from all schema summaries (legacy method)."""
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
    
    def _build_schema_markdown_from_tables(self, table_cards: List[TableCard]) -> str:
        """Build schema markdown from selected table cards only (respects schema preference)."""
        if not table_cards:
            return ""
        # Build summaries from table cards
        summaries = []
        for card in table_cards:
            # TableCard.name is just the table name, card.schema is the schema
            table_name = f"{card.schema}.{card.name}" if card.schema else card.name
            # Find matching schema summary to get FK details
            matching_summary = None
            for summary in self.schema_summaries:
                # Match by name and schema
                summary_schema = getattr(summary, "schema", None)
                if summary.name == card.name and (
                    (card.schema and summary_schema == card.schema) or
                    (not card.schema and not summary_schema)
                ):
                    matching_summary = summary
                    break
            
            fk_details = []
            if matching_summary:
                fk_details = [_format_fk_detail(fk) for fk in matching_summary.foreign_keys]
            
            summaries.append({
                "table": table_name,
                "columns": ", ".join(col.name for col in card.columns),
                "foreign_keys": ", ".join(fk_details) if fk_details else "",
                "row_estimate": str(card.row_estimate) if card.row_estimate else "",
            })
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

        # Build schema_markdown from selected tables only (schema preference already applied)
        selected_schema_markdown = self._build_schema_markdown_from_tables(retrieval.tables)
        
        result_state = self.workflow.invoke(
            {
                "question": question,
                "prompt_inputs": prompt_inputs,
                "schema_markdown": selected_schema_markdown,  # Use selected tables only
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
            "formatted_prompt": result_state.get("formatted_prompt"),  # Full prompt sent to planner LLM
            "final_sql": result_state.get("final_sql"),  # Final SQL after all intent repairs (before execution)
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

    def _validate_runtime_requirements(self) -> None:
        embed_provider = (self.embedding_config.provider or "").strip().lower()
        if not embed_provider:
            raise ValueError(
                "Embedding provider is required. Set SQLAI_EMBED_PROVIDER (e.g. 'huggingface') "
                "and restart the agent."
            )
        if not self.embedding_config.model:
            raise ValueError(
                "Embedding model is required. Set SQLAI_EMBED_MODEL "
                "(e.g. 'google/embeddinggemma-300m')."
            )
        if embed_provider == "huggingface" and not self.embedding_config.api_key:
            raise ValueError(
                "Hugging Face embeddings require an API key. Set SQLAI_EMBED_API_KEY with a valid token."
            )

    def _sync_graph_cache_and_vectors(self) -> None:
        if not self.schema_summaries:
            return
        schema_key = self._schema_key() or "(default)"
        existing_tables = set(self.graph_cache.list_tables(schema_key))
        current_tables = {summary.name for summary in self.schema_summaries}
        removed_tables = existing_tables - current_tables
        if removed_tables:
            LOGGER.info(
                "Removing graph cache entries for dropped tables: %s",
                ", ".join(sorted(removed_tables)),
            )
            self.graph_cache.delete_tables(schema_key, removed_tables)
            self.vector_store.delete_tables(schema_key, removed_tables)

        # Pre-load all cached hashes in one query
        cached_hashes = self.graph_cache.get_schema_hashes(schema_key)
        LOGGER.info(
            "Loaded %s cached graph card table(s) from schema '%s' for comparison.",
            len(cached_hashes),
            schema_key,
        )

        # Identify tables that need refresh (hash mismatch or missing)
        tables_to_refresh: List[str] = []
        summaries_to_process: List[TableSummary] = []
        for summary in self.schema_summaries:
            table_name = summary.name
            schema_hash = self.table_hashes.get(table_name)
            if not schema_hash:
                continue
            cached_hash = cached_hashes.get(table_name)
            if cached_hash == schema_hash:
                continue  # Cache hit, skip
            summaries_to_process.append(summary)
            tables_to_refresh.append(table_name)

        if not summaries_to_process:
            LOGGER.info(
                "All %s table(s) already have cached graph cards with matching schema hashes. Skipping graph card building.",
                len(self.schema_summaries),
            )
            return

        LOGGER.info(
            "Building graph cards for %s table(s) that need refresh.",
            len(summaries_to_process),
        )

        # Parallelize graph card building
        total_tables = len(summaries_to_process)
        if total_tables > 20:
            max_workers = min(8, total_tables)
            LOGGER.info(
                "Using ThreadPoolExecutor with %s worker(s) for graph card building.",
                max_workers,
            )

            def process_table(summary: TableSummary) -> None:
                table_name = summary.name
                schema_hash = self.table_hashes.get(table_name)
                if not schema_hash:
                    LOGGER.warning("No schema hash found for table '%s', skipping graph card building.", table_name)
                    return
                try:
                    records = self._build_graph_card_records(schema_key, table_name, schema_hash)
                    self.graph_cache.replace_table_cards(schema_key, table_name, schema_hash, records)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "Failed to build graph cards for table '%s': %s",
                        table_name,
                        exc,
                    )
                    raise

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_table, summary): summary.name
                    for summary in summaries_to_process
                }
                for future in as_completed(futures):
                    table_name = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error(
                            "Graph card building failed for table '%s': %s",
                            table_name,
                            exc,
                        )
        else:
            # Sequential for small batches
            for summary in summaries_to_process:
                table_name = summary.name
                schema_hash = self.table_hashes.get(table_name)
                if not schema_hash:
                    LOGGER.warning("No schema hash found for table '%s', skipping graph card building.", table_name)
                    continue
                try:
                    records = self._build_graph_card_records(schema_key, table_name, schema_hash)
                    self.graph_cache.replace_table_cards(schema_key, table_name, schema_hash, records)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "Failed to build graph cards for table '%s': %s",
                        table_name,
                        exc,
                    )
                    raise

        if tables_to_refresh:
            LOGGER.info(
                "Refreshing vector store for %s table(s): %s",
                len(tables_to_refresh),
                ", ".join(tables_to_refresh[:10]),
            )
            self.vector_store.refresh_tables(
                schema_key,
                tables_to_refresh,
                self.graph_cache,
                self.semantic_retriever.provider,
            )

    def _build_graph_card_records(
        self,
        schema: str,
        table_name: str,
        schema_hash: str,
    ) -> List[GraphCardRecord]:
        records: List[GraphCardRecord] = []
        table_card = next((card for card in self.graph_context.tables if card.name == table_name), None)
        if table_card:
            metadata: Dict[str, Any] = {}
            if table_card.row_estimate is not None:
                metadata["row_estimate"] = table_card.row_estimate
            if table_card.comment:
                metadata["comment"] = table_card.comment
            records.append(
                GraphCardRecord(
                    schema=schema,
                    table=table_name,
                    card_type="table",
                    identifier="__table__",
                    schema_hash=schema_hash,
                    text=table_card.render(),
                    metadata=metadata,
                )
            )

        column_cards = [card for card in self.graph_context.column_cards if card.table == table_name]
        for column_card in column_cards:
            column = column_card.column
            metadata = {
                "column": column.name,
                "type": str(column.type),
                "nullable": column.nullable,
            }
            if column.comment:
                metadata["comment"] = column.comment
            if column_card.sample_values:
                metadata["sample_values"] = column_card.sample_values[:5]
            records.append(
                GraphCardRecord(
                    schema=schema,
                    table=table_name,
                    card_type="column",
                    identifier=column.name,
                    schema_hash=schema_hash,
                    text=column_card.render(),
                    metadata=metadata,
                )
            )

        relationship_cards = [
            card for card in self.graph_context.relationships if card.table == table_name
        ]
        for rel_card in relationship_cards:
            identifier = (
                f"{','.join(rel_card.detail.constrained_columns)}->{rel_card.detail.referred_table}"
            )
            metadata = {
                "referred_table": rel_card.detail.referred_table,
                "constrained_columns": rel_card.detail.constrained_columns,
                "referred_columns": rel_card.detail.referred_columns,
            }
            records.append(
                GraphCardRecord(
                    schema=schema,
                    table=table_name,
                    card_type="relationship",
                    identifier=identifier,
                    schema_hash=schema_hash,
                    text=rel_card.render(),
                    metadata=metadata,
                )
            )
        return records

    def _hydrate_table_metadata(self) -> None:
        schema_key = self._schema_key()
        schema_cache_key = schema_key or "(default)"  # Consistent with other methods
        summaries = self.schema_summaries
        total_tables = len(summaries)
        if total_tables == 0:
            LOGGER.info("No tables detected during metadata hydration.")
            return

        LOGGER.info(
            "Hydrating metadata for %s tables in schema '%s'.",
            total_tables,
            schema_cache_key,
        )

        self.table_hashes = {summary.name: self._table_schema_hash(summary) for summary in summaries}
        cached_metadata = self.metadata_cache.fetch_schema(schema_cache_key)
        LOGGER.info(
            "Loaded %s cached table(s) from schema '%s' for comparison.",
            len(cached_metadata),
            schema_cache_key,
        )

        def process_summary(summary: TableSummary) -> TableSummary:
            table_name = summary.name
            schema_hash = self.table_hashes.get(table_name)
            if not schema_hash:
                LOGGER.warning("No schema hash found for table '%s', skipping metadata hydration.", table_name)
                return summary
            LOGGER.debug("Starting metadata hydration for table '%s'", table_name)
            try:
                cached = cached_metadata.get(table_name)
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

                self.metadata_cache.upsert(schema_cache_key, table_name, schema_hash, description, samples)
                LOGGER.debug("Persisted metadata for table '%s' into cache.", table_name)

                columns = [
                    replace(column, sample_values=samples.get(column.name))
                    for column in summary.columns
                ]
                hydrated_summary = replace(
                    summary,
                    columns=columns,
                    description=description,
                    comment=description or summary.comment,
                )
                LOGGER.debug("Finished metadata hydration for table '%s'", table_name)
                return hydrated_summary
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Metadata hydration failed for table '%s': %s",
                    table_name,
                    exc,
                    exc_info=True,
                )
                raise

        cache_hits = 0
        cache_misses = 0
        
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
                        # Count cache hits/misses
                        cached = cached_metadata.get(table_name)
                        if cached and cached.get("schema_hash") == self.table_hashes.get(table_name):
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception(
                            "Unhandled exception hydrating table '%s': %s",
                            table_name,
                            exc,
                        )
                        updated_map[table_name] = summaries[idx]
                        cache_misses += 1
            updated_summaries = [
                updated_map.get(summary.name, summary) for summary in summaries
            ]
        else:
            updated_summaries = []
            for summary in summaries:
                result = process_summary(summary)
                updated_summaries.append(result)
                # Count cache hits/misses
                cached = cached_metadata.get(summary.name)
                if cached and cached.get("schema_hash") == self.table_hashes.get(summary.name):
                    cache_hits += 1
                else:
                    cache_misses += 1

        self.schema_summaries = updated_summaries
        LOGGER.info(
            "Metadata hydration complete for %s table(s). Cache hits: %s, Cache misses: %s",
            total_tables,
            cache_hits,
            cache_misses,
        )

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
            # Use consistent FK formatting for hash stability
            payload_parts.extend(
                [f"fk:{_format_fk_detail(fk)}" for fk in summary.foreign_keys]
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
            LOGGER.error("LLM description generation failed for %s: %s", summary.name, exc, exc_info=True)
            raise RuntimeError(
                f"LLM description generation failed for table '{summary.name}'. "
                "Fix the LLM configuration before continuing."
            ) from exc
        if not description or not str(description).strip():
            raise RuntimeError(
                f"LLM returned an empty description for table '{summary.name}'. "
                "Cannot proceed without valid descriptions."
            )
        description_text = str(description).strip()
        LOGGER.info("Table description for %s: %s", summary.name, description_text)
        return description_text

    def _collect_column_samples(self, summary) -> Dict[str, List[str]]:
        qualified_table = self._qualify_table(summary.name)
        samples: Dict[str, List[str]] = {}
        limit = min(50, self.db_config.sample_row_limit or 50)
        query = self._sample_query(qualified_table, limit=limit)
        try:
            df = pd.read_sql_query(query, self.engine)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Sample fetch failed for %s: %s", qualified_table, exc)
            return samples

        if df.empty:
            return samples

        for column in summary.columns:
            if column.name not in df.columns:
                continue
            values = (
                df[column.name]
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

