"""
LangGraph workflow orchestrating question -> SQL -> analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypedDict

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.output_parsers import OutputFixingParser  # Newer LangChain
except Exception:  # pragma: no cover
    OutputFixingParser = None  # type: ignore[assignment]
from langchain_core.messages import AIMessage
import json
import re
import difflib
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
from sqlai.utils.sql_transpiler import transpile_sql
from sqlglot import parse_one, exp
import sqlglot

LOGGER = logging.getLogger(__name__)

MAX_INTENT_REPAIRS = 3


def _safe_invoke_json(chain: Any, payload: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Invoke a chain that is expected to return JSON. If parsing fails (e.g., trailing commas),
    attempt to repair the JSON using OutputFixingParser. As a last resort, return an empty,
    well-formed structure to keep the workflow moving.
    
    Note: The chain typically includes JsonOutputParser, so we need to extract raw response
    before parsing to capture what the LLM actually returned.
    """
    parser = JsonOutputParser()
    fixer = None
    if OutputFixingParser is not None:
        try:
            fixer = OutputFixingParser.from_llm(parser=parser, llm=llm)  # type: ignore[call-arg]
        except Exception:  # pragma: no cover
            fixer = None
    
    # Extract the prompt and LLM from chain to get raw response
    # Chain structure: prompt | llm | parser
    # We need to invoke prompt | llm to get raw response
    raw_content = ""
    try:
        # Try to get raw response by invoking without parser
        # Extract components from chain if possible
        if hasattr(chain, "first") and hasattr(chain, "middle") and hasattr(chain, "last"):
            # Chain is likely a RunnableSequence
            prompt_part = chain.first if hasattr(chain, "first") else None
            llm_part = chain.middle[0] if hasattr(chain, "middle") and len(chain.middle) > 0 else llm
            if prompt_part and llm_part:
                raw_response = (prompt_part | llm_part).invoke(payload)
                raw_content = getattr(raw_response, "content", str(raw_response)) or ""
        else:
            # Fallback: try invoking the full chain and see if we can extract raw content
            result = chain.invoke(payload)
            # If result is already parsed (dict), we lost the raw content
            if isinstance(result, dict):
                # Try to get raw by re-invoking without parser
                # This is a best-effort approach
                pass
            elif hasattr(result, "content"):
                raw_content = getattr(result, "content", None) or ""
    except Exception:
        pass  # Will try chain.invoke below
    
    try:
        result = chain.invoke(payload)
        # Attach raw content if we captured it
        if isinstance(result, dict):
            if raw_content:
                result["__raw"] = raw_content
            return result
        # If result is not a dict, it might be an AIMessage
        if hasattr(result, "content"):
            raw_content = getattr(result, "content", None) or raw_content
            # Try to parse it
            try:
                parsed = parser.parse(raw_content) if raw_content else {}
                if isinstance(parsed, dict):
                    parsed["__raw"] = raw_content
                return parsed
            except Exception:
                pass
        return result
    except Exception as exc:
        # Try to obtain raw content and repair it
        if not raw_content:
            try:
                # Try to extract prompt and llm from chain
                if hasattr(chain, "first") and hasattr(chain, "middle"):
                    prompt_part = chain.first
                    llm_part = chain.middle[0] if len(chain.middle) > 0 else llm
                    if prompt_part and llm_part:
                        raw_response = (prompt_part | llm_part).invoke(payload)
                        raw_content = getattr(raw_response, "content", str(raw_response)) or ""
            except Exception:
                pass
        
        if not raw_content:
            LOGGER.warning("Could not extract raw LLM response for JSON repair. Exception: %s", exc)
            return {"verdict": "reject", "reasons": [], "repair_hints": [], "__raw": None}
        
        try:
            if fixer is not None:
                parsed = fixer.parse(raw_content)
                # Attach raw text for downstream logging
                if isinstance(parsed, dict):
                    parsed["__raw"] = raw_content
                return parsed
            # Local best-effort repair if OutputFixingParser is unavailable
            repaired = _best_effort_json_repair(raw_content)
            repaired["__raw"] = raw_content
            return repaired
        except Exception as repair_exc:
            LOGGER.debug("JSON repair failed: %s. Raw content: %s", repair_exc, raw_content[:500] if raw_content else "<<empty>>")
            return {"verdict": "reject", "reasons": [], "repair_hints": [], "__raw": raw_content}


def _best_effort_json_repair(text: str) -> Dict[str, Any]:
    """
    Minimal, generic JSON repair:
    - Extract first {...} block
    - Replace single quotes with double quotes when safe
    - Remove trailing commas before } or ]
    - Attempt json.loads; fallback to an empty structure
    """
    # Extract JSON object if extra prose present
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    # Replace smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    # Naive single-quote to double-quote for keys only (simple heuristic)
    text = re.sub(r"'\s*([A-Za-z0-9_\-]+)\s*'\s*:", r'"\1":', text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except Exception:
        return {"verdict": "reject", "reasons": [], "repair_hints": []}


class QueryState(TypedDict, total=False):
    question: str
    prompt_inputs: Dict[str, Any]
    formatted_prompt: str  # Full formatted prompt sent to planner LLM
    final_sql: str  # Final SQL that will be executed (after all intent repairs)
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


def _fail():
    def node(state: QueryState) -> QueryState:
        # No further processing; execution_error already set by router
        message = state.get("execution_error") or "Query failed during intent validation."
        LOGGER.info("Failing early due to unresolved intent issues: %s", message)
        # Produce an answer payload so callers don't KeyError
        return {
            "answer": {
                "text": message,
                "chart": None,
                "followups": [],
            },
            "executions": [],
        }

    return node


def _remap_missing_columns(sql_text: str, graph: Optional[GraphContext]) -> str:
    """
    If a qualified column references a table alias whose table does not contain the column,
    but another joined alias does contain it, remap the qualifier to that alias.
    This is a generic, schema-aware normalization; it does NOT add or drop joins.
    """
    if not isinstance(graph, GraphContext) or not sql_text:
        return sql_text
    try:
        ast = parse_one(sql_text)
    except Exception:
        return sql_text

    # Build alias -> table and table -> columns maps
    alias_to_table: Dict[str, str] = {}
    for tbl in ast.find_all(exp.Table):
        alias_expr = tbl.args.get("alias")
        alias = None
        if alias_expr is not None:
            alias = getattr(alias_expr, "this", None)
            alias = getattr(alias, "name", alias)  # normalize identifier
        name = getattr(getattr(tbl, "this", None), "name", None)
        if name:
            alias_key = (alias or name).lower()
            alias_to_table[alias_key] = name.lower()

    table_to_columns: Dict[str, set] = {}
    for table_card in graph.tables:
        table_to_columns[table_card.name.lower()] = {c.name.lower() for c in table_card.columns}

    changed = False
    for col in ast.find_all(exp.Column):
        qualifier = col.table
        col_name = col.name.lower()
        if qualifier:
            qual_key = qualifier.lower()
            table_name = alias_to_table.get(qual_key)
            if table_name and col_name not in table_to_columns.get(table_name, set()):
                # Try to find another alias that has this column
                for alias_key, tname in alias_to_table.items():
                    if col_name in table_to_columns.get(tname, set()):
                        col.set("table", exp.to_identifier(alias_key))
                        changed = True
                        break
    return ast.sql() if changed else sql_text


def _repair_unknown_columns(sql_text: str, graph: Optional[GraphContext]) -> str:
    """
    If a column name does not exist on the referenced table (or on any joined table),
    try to map it to the closest valid column name across the joined tables using
    a string similarity heuristic. If a close match is found, replace the column
    name (and re-qualify with the alias that owns that column).
    """
    if not isinstance(graph, GraphContext) or not sql_text:
        return sql_text
    try:
        ast = parse_one(sql_text)
    except Exception:
        return sql_text

    # alias -> table and table -> columns maps
    alias_to_table: Dict[str, str] = {}
    for tbl in ast.find_all(exp.Table):
        alias_expr = tbl.args.get("alias")
        alias = None
        if alias_expr is not None:
            alias = getattr(alias_expr, "this", None)
            alias = getattr(alias, "name", alias)
        name = getattr(getattr(tbl, "this", None), "name", None)
        if name:
            alias_key = (alias or name).lower()
            alias_to_table[alias_key] = name.lower()

    table_to_columns: Dict[str, set] = {}
    all_columns: Dict[str, str] = {}  # column_name -> table_name (first occurrence wins)
    for table_card in graph.tables:
        tname = table_card.name.lower()
        cols = {c.name.lower() for c in table_card.columns}
        table_to_columns[tname] = cols
        for cname in cols:
            all_columns.setdefault(cname, tname)

    changed = False
    for col in ast.find_all(exp.Column):
        col_name = col.name.lower()
        qualifier = col.table.lower() if col.table else None

        # Determine current table (if any) for this qualified column
        current_table = alias_to_table.get(qualifier) if qualifier else None
        current_has = current_table and (col_name in table_to_columns.get(current_table, set()))
        global_has = col_name in all_columns

        if current_has or global_has:
            continue  # already valid

        # Find closest column name across all known columns
        candidates = list(all_columns.keys())
        best = difflib.get_close_matches(col_name, candidates, n=1, cutoff=0.8)
        if not best:
            continue

        best_name = best[0]
        target_table = all_columns[best_name]

        # Find an alias that maps to the target table; if none, skip
        target_alias = None
        for alias_key, tname in alias_to_table.items():
            if tname == target_table:
                target_alias = alias_key
                break
        if not target_alias:
            continue

        # Apply the repair: rename column and re-qualify with the alias that owns it
        col.set("this", exp.to_identifier(best_name))
        col.set("table", exp.to_identifier(target_alias))
        changed = True

    return ast.sql() if changed else sql_text


def _sanitize_sql_list(sql_statements: List[str]) -> List[str]:
    """
    Remove non-data 'probe' statements (e.g., SELECT 'pie' FROM dual) and keep only
    statements that reference actual tables/joins. Heuristics:
    - Drop statements with no FROM clause
    - For Oracle, drop SELECT ... FROM dual unless selecting from a function that touches data
    - Keep first 1-2 meaningful statements max
    """
    cleaned: List[str] = []
    for sql in sql_statements:
        try:
            ast = sqlglot.parse_one(sql, read=None)  # auto-detect
        except Exception:
            # If can't parse, keep as-is to let guardrails decide
            cleaned.append(sql)
            continue
        has_from = any(isinstance(node, exp.From) for node in ast.find_all(exp.From))
        if not has_from:
            # e.g., SELECT 'pie'; drop
            continue
        # Detect FROM dual
        from_tables = [t.this.name.lower() for t in ast.find_all(exp.Table) if hasattr(t.this, "name")]
        if "dual" in from_tables and len(from_tables) == 1:
            # likely a probe, drop
            continue
        cleaned.append(sql)
        if len(cleaned) >= 2:
            break
    return cleaned or sql_statements[:1]


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
    graph.add_node("fail", _fail())

    graph.set_entry_point("plan")
    # Route after planning: if no SQL, fail early with a clear message
    def _route_after_plan(state: QueryState) -> str:
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if not sql or (isinstance(sql, str) and not sql.strip()) or (isinstance(sql, list) and not any((s or "").strip() for s in sql)):
            # ensure execution_error is set by planner node; if not, set a generic one
            if not state.get("execution_error"):
                state["execution_error"] = "Planner failed to produce SQL for the question."
            return "fail"
        return "intent_critic"
    graph.add_conditional_edges("plan", _route_after_plan, {"intent_critic": "intent_critic", "fail": "fail"})
    graph.add_conditional_edges(
        "intent_critic",
        _route_after_intent,
        {
            "repair": "intent_repair",
            "execute": "execute",
            "fail": "fail",
        },
    )
    # After an intent repair, conditionally continue or fail to avoid an extra 4th critic call
    def _route_after_intent_repair(state: QueryState) -> str:
        attempts = state.get("intent_attempts", 0)
        last_critic = state.get("intent_critic") or {}
        if attempts >= MAX_INTENT_REPAIRS:
            reasons = last_critic.get("reasons") or []
            sql = state.get("plan", {}).get("sql")
            state["execution_error"] = (
                "Unable to produce SQL that satisfies the question after "
                f"{attempts} intent repair attempt(s).\n"
                f"Reasons: {reasons}\n\nLast SQL:\n{sql}"
            )
            return "fail"
        return "intent_critic"
    graph.add_conditional_edges("intent_repair", _route_after_intent_repair, {"intent_critic": "intent_critic", "fail": "fail"})
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
    graph.add_edge("fail", END)

    return graph.compile()


def _plan_sql(llm: Any):
    parser = JsonOutputParser()
    prompt = agent_prompt()
    chain = prompt | llm | parser

    def node(state: QueryState) -> QueryState:
        LOGGER.debug("Planning SQL for question=%s", state.get("question"))
        prompt_inputs = dict(state.get("prompt_inputs") or {})
        prompt_inputs.setdefault("user_question", state.get("question", ""))
        
        # Log what's being passed to the prompt (summary)
        LOGGER.info("Prompt inputs summary: table_cards=%d chars, column_cards=%d chars, relationship_map=%d chars",
                   len(prompt_inputs.get("table_cards", "") or ""),
                   len(prompt_inputs.get("column_cards", "") or ""),
                   len(prompt_inputs.get("relationship_map", "") or ""))
        
        # Format the prompt to get the actual messages that will be sent to LLM
        formatted_prompt_text = ""
        try:
            formatted_messages = prompt.format_messages(**prompt_inputs)
            
            # Log the full prompt
            LOGGER.info("=" * 80)
            LOGGER.info("PLANNER PROMPT - Full prompt sent to LLM:")
            LOGGER.info("=" * 80)
            for msg in formatted_messages:
                role = msg.__class__.__name__
                content = msg.content if hasattr(msg, "content") else str(msg)
                LOGGER.info("\n[%s MESSAGE]", role.upper())
                LOGGER.info("-" * 80)
                # Handle both text-only (string) and multimodal (list) content
                if isinstance(content, list):
                    # Multimodal content: list of content blocks
                    for idx, block in enumerate(content):
                        if isinstance(block, dict):
                            block_type = block.get("type", "unknown")
                            if block_type == "text":
                                LOGGER.info("Text block %d: %s", idx + 1, block.get("text", ""))
                            elif block_type == "image_url":
                                image_url = block.get("image_url", {})
                                if isinstance(image_url, dict):
                                    LOGGER.info("Image block %d: %s", idx + 1, image_url.get("url", ""))
                                else:
                                    LOGGER.info("Image block %d: %s", idx + 1, image_url)
                            else:
                                LOGGER.info("Content block %d (type=%s): %s", idx + 1, block_type, block)
                        else:
                            LOGGER.info("Content block %d: %s", idx + 1, block)
                else:
                    # Text-only content: simple string
                    LOGGER.info("%s", content)
                LOGGER.info("-" * 80)
            LOGGER.info("=" * 80)
            
            # Store formatted prompt for UI display (handle multimodal)
            prompt_parts = []
            for msg in formatted_messages:
                role = msg.__class__.__name__.upper()
                content = msg.content if hasattr(msg, "content") else str(msg)
                if isinstance(content, list):
                    # Format multimodal content
                    content_str = "\n".join([
                        f"  [{idx+1}] {block.get('type', 'unknown')}: {block.get('text') or block.get('image_url', {}).get('url', '') if isinstance(block, dict) else block}"
                        for idx, block in enumerate(content)
                    ])
                    prompt_parts.append(f"[{role}]\n{content_str}")
                else:
                    prompt_parts.append(f"[{role}]\n{content}")
            formatted_prompt_text = "\n\n".join(prompt_parts)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to format prompt for logging: %s", exc)
            formatted_prompt_text = f"[Error formatting prompt: {exc}]"
        
        # Invoke with robust JSON handling: try strict parser first, then auto-fix
        raw_planner_text = ""
        try:
            # Capture raw response first for diagnostics
            raw_response = (prompt | llm).invoke(prompt_inputs)
            raw_planner_text = getattr(raw_response, "content", str(raw_response)) or ""
            LOGGER.debug("Planner raw LLM response (before parsing): %s", raw_planner_text[:500] if raw_planner_text else "<<empty>>")
            if not raw_planner_text.strip():
                LOGGER.warning("Planner LLM returned empty response. This may indicate a model compatibility issue.")
            plan = parser.parse(raw_planner_text)
        except Exception as parse_exc:
            LOGGER.debug("Planner JSON parsing failed: %s. Raw response: %s", parse_exc, raw_planner_text[:500] if raw_planner_text else "<<empty>>")
            # Fallback: attempt repair
            try:
                if not raw_planner_text:
                    # Re-invoke if we don't have raw text yet
                    raw_response = (prompt | llm).invoke(prompt_inputs)
                    raw_planner_text = getattr(raw_response, "content", str(raw_response)) or ""
                if raw_planner_text.strip():
                    if OutputFixingParser is not None:
                        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)  # type: ignore[call-arg]
                        plan = fixing_parser.parse(raw_planner_text)
                    else:
                        plan = _best_effort_json_repair(raw_planner_text)  # type: ignore[assignment]
                else:
                    LOGGER.warning("Planner LLM returned empty response after retry. Model may not support this prompt format.")
                    plan = {"sql": "", "plan": {}, "rationale_summary": "", "tests": [], "summary": "", "followups": []}
            except Exception as repair_exc:
                LOGGER.warning("Planner JSON repair also failed: %s. Raw response: %s", repair_exc, raw_planner_text[:500] if raw_planner_text else "<<empty>>")
                plan = {"sql": "", "plan": {}, "rationale_summary": "", "tests": [], "summary": "", "followups": []}
        if isinstance(plan, dict):
            sql_statements = plan.get("sql")
            if not sql_statements:
                # Log raw response for debugging when SQL is empty
                if raw_planner_text:
                    LOGGER.info("Planner returned empty SQL. Raw LLM response: %s", raw_planner_text[:1000])
                else:
                    LOGGER.warning("Planner returned empty SQL and no raw response was captured. Model may have failed silently.")
                # Planner failed to produce SQL; bootstrap a minimal SQL using the repair prompt
                LOGGER.info("Planner returned no SQL. Bootstrapping baseline SQL from Graph Context.")
                try:
                    bootstrap_chain = REPAIR_PROMPT | llm | JsonOutputParser()
                    graph_text = state.get("schema_markdown", "")
                    result = _safe_invoke_json(
                        bootstrap_chain,
                        {
                            "dialect": state.get("dialect", ""),
                            "graph": graph_text,
                            "sql": "",
                            "error": "Planner produced no SQL; synthesize a minimal, correct SQL answering the question from Graph Context and analysis hints.",
                            "repair_hints": [
                                "Use FK paths from RELATIONSHIP_MAP to join necessary tables",
                                "Select explicit columns, no SELECT *",
                                "Apply GROUP BY/COUNT and ORDER BY DESC with row cap per dialect",
                            ],
                            "plan": {"note": "bootstrap from question and graph"},
                        },
                        llm,
                    )
                    bootstrap_raw = result.get("__raw", "")
                    if bootstrap_raw:
                        LOGGER.debug("Bootstrap raw LLM response: %s", bootstrap_raw[:500])
                    patched_sql = result.get("patched_sql") or ""
                    if patched_sql.strip():
                        plan["sql"] = patched_sql
                        plan.setdefault("notes", []).append("Bootstrapped baseline SQL due to empty planner output.")
                        LOGGER.debug("Bootstrapped SQL: %s", patched_sql)
                    else:
                        if bootstrap_raw:
                            LOGGER.warning("Bootstrap returned empty patched_sql. Raw response: %s", bootstrap_raw[:1000])
                        else:
                            LOGGER.warning("Bootstrap returned empty patched_sql and no raw response captured.")
                        plan["sql_generation_note"] = "Planner returned no SQL and bootstrap produced no patch."
                        # Surface a clear error for routing to fail node
                        state["execution_error"] = "Planner failed to produce SQL, and bootstrap also returned no patch."
                except Exception as e:
                    LOGGER.debug("Bootstrap baseline SQL failed: %s", e)
                    plan["sql_generation_note"] = "Planner returned no SQL and bootstrap failed."
                    state["execution_error"] = "Planner failed to produce SQL, and bootstrap failed."
            else:
                plan = _post_process_plan(plan, state)
            LOGGER.debug("LLM plan: %s", plan)
        return {
            "plan": plan,
            "prompt_inputs": prompt_inputs,
            "formatted_prompt": formatted_prompt_text,  # Store for UI
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
        attempts = state.get("intent_attempts", 0)
        LOGGER.info("Intent critic iteration %d/%d (MAX)", attempts + 1, MAX_INTENT_REPAIRS)
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if isinstance(sql, list):
            sql_text = "\n".join(sql)
        else:
            sql_text = sql or ""
        plan_summary = plan.get("plan") or {}
        graph_text = state.get("schema_markdown", "")
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "question": state.get("question", ""),
                "plan": plan_summary,
                "sql": sql_text,
            },
            llm,
        )
        LOGGER.info("Intent critic iteration %d verdict: %s", attempts + 1, result.get("verdict"))
        if result.get("reasons"):
            LOGGER.info("Intent critic reasons: %s", result.get("reasons"))
        if result.get("repair_hints"):
            LOGGER.info("Intent critic repair_hints: %s", result.get("repair_hints"))
        if not (result.get("reasons") or result.get("repair_hints")):
            raw = result.get("__raw")
            if raw:
                LOGGER.info("Intent critic returned empty reasons/hints. Raw response: %s", raw)
        # Visualization-only complaints should not block execution
        try:
            reasons = [str(r).lower() for r in (result.get("reasons") or [])]
            hints = [str(h).lower() for h in (result.get("repair_hints") or [])]
            viz_keywords = ("chart", "visualization", "pie chart")
            only_viz_reasons = bool(reasons) and all(any(k in r for k in viz_keywords) for r in reasons)
            only_viz_hints = bool(hints) and all(any(k in h for k in viz_keywords) for h in hints)
            if (result.get("verdict") or "").lower() == "reject" and (only_viz_reasons or only_viz_hints):
                LOGGER.info("Intent critic: overriding reject to accept (visualization-only complaints).")
                result = {"verdict": "accept", "reasons": [], "repair_hints": []}
        except Exception:  # pragma: no cover
            pass
        state["intent_critic"] = result
        # Clear any previous execution error when we are re-evaluating intent
        state.pop("execution_error", None)
        return state

    return node


def _intent_repair(llm: Any):
    parser = JsonOutputParser()
    chain = REPAIR_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        attempts = state.get("intent_attempts", 0)
        if attempts >= MAX_INTENT_REPAIRS:
            LOGGER.info("Reached MAX intent repair iterations (%d). Skipping further repair.", MAX_INTENT_REPAIRS)
            return state
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        if isinstance(sql, list):
            sql_text = sql[0]
        else:
            sql_text = sql or ""
        critic = state.get("intent_critic") or {}
        error_text = "; ".join(critic.get("reasons", []))
        graph_text = state.get("schema_markdown", "")
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "sql": sql_text,
                "error": error_text,
                "repair_hints": critic.get("repair_hints", []),
                "plan": plan.get("plan") or {},
            },
            llm,
        )
        LOGGER.info("Intent repair attempt %d result received", attempts + 1)
        patched_sql = result.get("patched_sql")
        if patched_sql and patched_sql.strip() and patched_sql.strip() != sql_text.strip():
            plan["sql"] = patched_sql
            plan.setdefault("notes", []).append(
                f"Intent repair iteration {attempts + 1} applied."
            )
            state["plan"] = plan
        else:
            LOGGER.info("Intent repair attempt %d produced no effective change", attempts + 1)
        state["intent_attempts"] = attempts + 1
        state["intent_critic"] = {}
        state.pop("execution_error", None)
        return state

    return node


def _execute_sql(engine: Engine):
    def node(state: QueryState) -> QueryState:
        plan = state["plan"]
        sql_statements = plan.get("sql")
        
        # Log the final SQL and prompt that will be used for execution (after all intent critic/repair iterations)
        LOGGER.info("=" * 80)
        LOGGER.info("FINAL PROMPT & SQL BEFORE EXECUTION (after all intent critic/repair iterations):")
        LOGGER.info("=" * 80)
        intent_attempts = state.get("intent_attempts", 0)
        if intent_attempts > 0:
            LOGGER.info("Intent repair attempts: %d", intent_attempts)
        
        # Re-print the full prompt that was used to generate this SQL
        formatted_prompt = state.get("formatted_prompt")
        if formatted_prompt:
            LOGGER.info("\n[FULL PLANNER PROMPT - This is what generated the SQL below]")
            LOGGER.info("-" * 80)
            LOGGER.info("%s", formatted_prompt)
            LOGGER.info("-" * 80)
        
        # Log the final SQL that will be executed
        LOGGER.info("\n[FINAL SQL TO EXECUTE - After all intent repairs]")
        LOGGER.info("-" * 80)
        if isinstance(sql_statements, str):
            LOGGER.info("%s", sql_statements)
        elif isinstance(sql_statements, list):
            for idx, sql in enumerate(sql_statements, 1):
                LOGGER.info("\n-- Statement %d:\n%s", idx, sql)
        else:
            LOGGER.info("No SQL statements found in plan")
        LOGGER.info("-" * 80)
        LOGGER.info("=" * 80)
        
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

        # Sanitize away non-data/probe statements (e.g., SELECT 'pie' FROM dual)
        if isinstance(sql_statements, list):
            original_len = len(sql_statements)
            sql_statements = _sanitize_sql_list(sql_statements)
            if len(sql_statements) != original_len:
                LOGGER.info("Sanitized SQL list: removed %d non-data statement(s)", original_len - len(sql_statements))

        # Transpile SQL to target dialect if needed (safety net in case LLM generated wrong dialect)
        target_dialect = state.get("dialect") or ""
        for idx, sql in enumerate(sql_statements):
            transpiled_sql, was_transpiled = transpile_sql(sql, target_dialect=target_dialect)
            if was_transpiled:
                sql_statements[idx] = transpiled_sql
                LOGGER.info("Transpiled SQL statement %d to %s dialect", idx + 1, target_dialect)

        # First, attempt to repair unknown/misspelled columns in a schema-aware way
        for idx, sql in enumerate(sql_statements):
            repaired = _repair_unknown_columns(sql, graph)
            if repaired != sql:
                sql_statements[idx] = repaired
                LOGGER.info("Repaired unknown column names in statement %d based on schema context", idx + 1)

        # Schema-aware alias normalization to fix common qualifier mistakes
        for idx, sql in enumerate(sql_statements):
            normalized = _remap_missing_columns(sql, graph)
            if normalized != sql:
                sql_statements[idx] = normalized
                LOGGER.info("Normalized column qualifiers in statement %d based on schema context", idx + 1)

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
        
        # Update final SQL text after all processing (validation, repair, etc.)
        if isinstance(sql_statements, str):
            final_sql_text = sql_statements
        elif isinstance(sql_statements, list) and sql_statements:
            final_sql_text = "\n\n".join(f"-- Statement {idx}\n{sql}" for idx, sql in enumerate(sql_statements, 1))
        else:
            final_sql_text = ""
        
        return {
            "executions": executions,
            "plan": plan,
            "execution_error": execution_error,
            "executions_available": True,
            "final_sql": final_sql_text,  # Final SQL that was executed (after all repairs)
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
                "inference and explicitly note the gap rather than refusing.\n\n"
                "**CRITICAL: Self-critique your summary**\n"
                "- Verify that your summary directly answers the user's question.\n"
                "- Check that the key information requested in the question is present in your answer.\n"
                "- Ensure your answer matches the user's intent, not just the SQL results.\n"
                "- If the results don't fully answer the question, acknowledge what's missing.\n"
                "- Before finalizing, ask yourself: 'Does this answer what the user asked for?'",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Plan rationale: {rationale}\n\n"
                "SQL:\n{sql}\n\n"
                "Table preview:\n{preview}\n\n"
                "If a chart specification was provided, repeat it and explain it briefly.\n\n"
                "**Remember: Your summary must directly answer the user's question above.**",
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
    qualification_applied = False
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

    # Ensure tables are qualified with the default schema (if configured)
    graph_context = state.get("graph_context")
    known_tables: Set[str] = set()
    if isinstance(graph_context, GraphContext):
        known_tables = {card.name.lower() for card in graph_context.tables}
    if schema_name and adjusted:
        qualified_sqls = []
        for sql_text in adjusted:
            qualified_sql, qualified = _ensure_schema_qualification(
                sql_text,
                schema_name,
                known_tables,
            )
            if qualified:
                changed = True
                qualification_applied = True
            qualified_sqls.append(qualified_sql)
        adjusted = qualified_sqls

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
    if qualification_applied:
        note = f"Auto-qualified tables with schema {schema_name}."
        if note not in notes:
            notes.append(note)

    return plan


def _ensure_schema_qualification(
    sql_text: str,
    schema_name: str,
    known_tables: Set[str],
) -> Tuple[str, bool]:
    """
    Ensure all table references are qualified with the default schema.
    Only applies to tables that are part of the known graph context.
    """
    if not schema_name:
        return sql_text, False
    try:
        ast = sqlglot.parse_one(sql_text, read=None)
    except Exception:
        return sql_text, False

    cte_names: Set[str] = set()
    for cte in ast.find_all(exp.CTE):
        alias = getattr(getattr(cte, "alias", None), "name", None)
        name = alias or getattr(getattr(cte, "this", None), "name", None)
        if name:
            cte_names.add(name.lower())

    changed = False
    for table in ast.find_all(exp.Table):
        if table.args.get("db"):
            continue
        identifier = getattr(table, "this", None)
        if not isinstance(identifier, exp.Identifier):
            continue
        table_name = (identifier.name or "").lower()
        if not table_name:
            continue
        if table_name in cte_names:
            continue
        if known_tables and table_name not in known_tables:
            continue
        table.set("db", exp.to_identifier(schema_name))
        changed = True
    return (ast.sql() if changed else sql_text), changed


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
    if attempts >= MAX_INTENT_REPAIRS:
        LOGGER.info("Post-exec repair: reached MAX iterations (%d). Stopping repair loop.", MAX_INTENT_REPAIRS)
        return False
    return critic.get("verdict") == "reject"


def _critic_sql(llm: Any):
    parser = JsonOutputParser()
    chain = CRITIC_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        attempts = state.get("repair_attempts", 0)
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
        graph_text = state.get("schema_markdown", "")
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "sql": sql_text,
                "plan": plan_summary,
                "error": state.get("execution_error", ""),
            },
            llm,
        )
        LOGGER.info("Post-exec critic iteration %d/%d (MAX) verdict: %s", attempts + 1, MAX_INTENT_REPAIRS, result.get("verdict"))
        if result.get("reasons"):
            LOGGER.info("Post-exec critic reasons: %s", result.get("reasons"))
        if result.get("repair_hints"):
            LOGGER.info("Post-exec critic repair_hints: %s", result.get("repair_hints"))
        if not (result.get("reasons") or result.get("repair_hints")):
            raw = result.get("__raw")
            if raw:
                LOGGER.info("Post-exec critic returned empty reasons/hints. Raw response: %s", raw)
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
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": state.get("schema_markdown", ""),
                "sql": sql_text,
                "error": state.get("execution_error", ""),
                "repair_hints": critic.get("repair_hints", []),
                "plan": plan,
            },
            llm,
        )
        LOGGER.info("Post-exec repair attempt %d/%d (MAX) result received", attempts + 1, MAX_INTENT_REPAIRS)
        patched_sql = result.get("patched_sql")
        what_changed = result.get("what_changed")
        why_changed = result.get("why")
        if what_changed:
            LOGGER.info("Post-exec repair what_changed: %s", what_changed)
        if why_changed:
            LOGGER.info("Post-exec repair why: %s", why_changed)
        if not patched_sql:
            raw = result.get("__raw")
            if raw:
                LOGGER.info("Post-exec repair returned no patched_sql. Raw response: %s", raw)
        if patched_sql:
            plan["sql"] = patched_sql
            plan.setdefault("notes", []).append(
                f"Applied repair iteration {attempts + 1}."
            )
            LOGGER.debug("Patched SQL (iteration %d): %s", attempts + 1, patched_sql)
        state["plan"] = plan
        state["repair_attempts"] = attempts + 1
        state.pop("execution_error", None)
        state["executions"] = []
        return state

    return node

