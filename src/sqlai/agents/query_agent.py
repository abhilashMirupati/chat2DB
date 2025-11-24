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
    ANALYSIS_PROMPT,
)
from sqlai.agents.guard import repair_sql, validate_sql
from sqlai.graph.context import GraphContext
from sqlai.services.domain_fallbacks import maybe_generate_domain_sql
from sqlai.utils.sql_transpiler import transpile_sql
from sqlglot import parse_one, exp
import sqlglot

LOGGER = logging.getLogger(__name__)

# Default max repairs (can be overridden by config)
MAX_INTENT_REPAIRS = 2
RE_REASONING_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
RE_LEADING_FENCE = re.compile(r"^```(?:json)?", re.IGNORECASE)


def _format_desired_columns_hint(desired_columns: Optional[List[str]], raw_columns: Optional[List[str]] = None) -> str:
    """Format desired columns for prompt hints."""
    # Prefer raw user input (allows synonyms), fallback to validated
    columns_to_show = raw_columns if raw_columns else desired_columns
    if not columns_to_show:
        return "No specific column preferences. Include all relevant columns needed to answer the question."
    
    hint_text = (
        f"The user explicitly requested these columns in the results: {', '.join(columns_to_show)}. "
        f"**CRITICAL: Map these to actual column names from the Graph Context and ensure ALL of them are included in the SELECT clause** "
        f"(with appropriate table aliases/qualifiers if needed). If a column requires a JOIN, add that JOIN using FK paths.\n\n"
        f"**SYNONYM MAPPING GUIDANCE:**\n"
        f"- If the user used synonyms or natural language (e.g., 'test set name' instead of 'name'), map them to the correct column names from the available tables.\n"
        f"- Consider table context: if the user says 'test set' and there's a table like 'test_sets_real_data', map to columns from that table.\n"
        f"- Use semantic matching: 'test set name' likely refers to a 'name' column in a test_sets-related table.\n"
        f"- Match based on the question context and the selected tables in the Graph Context."
    )
    
    # Add validated hints if available and different from raw
    if desired_columns and raw_columns and set(desired_columns) != set(raw_columns):
        hint_text += f"\n\nValidated column matches: {', '.join(desired_columns)} (use these as reference for mapping)."
    
    return hint_text


def _format_desired_columns_section(desired_columns: Optional[List[str]], raw_columns: Optional[List[str]] = None) -> str:
    """Format desired columns as a section in prompts."""
    # Prefer raw user input (allows synonyms), fallback to validated
    columns_to_show = raw_columns if raw_columns else desired_columns
    if not columns_to_show:
        return ""
    
    section = f"\n\n[USER DESIRED COLUMNS]\nThe user explicitly wants these columns in the results: {', '.join(columns_to_show)}\n\n"
    section += "**IMPORTANT - SYNONYM & CONTEXT MAPPING:**\n"
    section += "1. Map these to actual column names from the Graph Context (the user may have used synonyms or natural language).\n"
    section += "2. **Consider table context**: If the user says 'test set' and there's a table like 'test_sets_real_data' in the selected tables, "
    section += "map to columns from that table (e.g., 'test set name' → 'test_sets_real_data.name').\n"
    section += "3. Use semantic matching: Match user terms to actual column names based on:\n"
    section += "   - The question context (what tables are relevant)\n"
    section += "   - The selected tables in Graph Context\n"
    section += "   - Column names and their meanings from TABLE CARDS and COLUMN CARDS\n"
    section += "4. Include ALL of these columns in the SELECT clause with appropriate table aliases/qualifiers.\n"
    section += "5. If a column requires a JOIN, add that JOIN using FK paths from RELATIONSHIP_MAP.\n"
    
    # Add validated hints if available
    if desired_columns and raw_columns and set(desired_columns) != set(raw_columns):
        section += f"\nValidated matches (for reference): {', '.join(desired_columns)}\n"
    
    return section


def _format_query_analysis_section(query_analysis: Dict[str, Any]) -> str:
    """Format query analysis (TODO list and checklist) for prompts."""
    if not query_analysis:
        return ""
    
    parts = []
    
    # Analysis summary
    analysis = query_analysis.get("analysis", "")
    if analysis:
        parts.append(f"[QUERY ANALYSIS]\n{analysis}\n")
    
    # TODO list
    todo_list = query_analysis.get("todo_list", [])
    if todo_list:
        parts.append("[TODO LIST - Steps to Complete]")
        for idx, todo in enumerate(todo_list, 1):
            parts.append(f"{idx}. {todo}")
        parts.append("")
    
    # Verification checklist
    checklist = query_analysis.get("verification_checklist", [])
    if checklist:
        parts.append("[VERIFICATION CHECKLIST - Verify After SQL Generation]")
        for item in checklist:
            parts.append(f"✓ {item}")
        parts.append("")
    
    # Potential pitfalls
    pitfalls = query_analysis.get("potential_pitfalls", [])
    if pitfalls:
        parts.append("[POTENTIAL PITFALLS - Avoid These Mistakes]")
        for pitfall in pitfalls:
            parts.append(f"⚠ {pitfall}")
        parts.append("")
    
    # Complex requirements
    complex_reqs = query_analysis.get("complex_requirements", [])
    if complex_reqs:
        parts.append("[COMPLEX REQUIREMENTS - SQL Features Needed]")
        for req in complex_reqs:
            parts.append(f"→ {req}")
        parts.append("")
    
    if not parts:
        return ""
    
    return "\n".join(parts)


def _strip_reasoning_wrappers(text: str) -> str:
    """
    Remove common reasoning wrappers (e.g., DeepSeek's <think> blocks, ```json fences)
    so downstream JSON parsing sees a clean object.
    """
    if not text:
        return text
    # Normalize bytes to string (works with any LLM model)
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    elif not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    if "<think>" in cleaned.lower():
        cleaned = RE_REASONING_BLOCK.sub("", cleaned)
    cleaned = RE_LEADING_FENCE.sub("", cleaned).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first balanced {...} block from the text. Returns None if no block found.
    """
    if not text:
        return None
    # Normalize bytes to string (works with any LLM model)
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    elif not isinstance(text, str):
        text = str(text)
    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]
    return None


def _extract_sql_from_malformed_json(text: str, field_name: str = "sql") -> Optional[str]:
    """
    Robustly extract SQL from malformed JSON text.
    Handles:
    - Complex SQL with nested quotes (single and double)
    - Multi-line SQL
    - Escaped characters
    - SQL in array format: ["SELECT ..."]
    - Arrays of objects with statement field: [{"description": "...", "statement": "SQL"}]
    - Malformed JSON (missing closing quotes, braces, etc.)
    - SQL with CTEs, subqueries, etc.
    - Null/empty values
    - Wrong field types (number, object, etc.)
    - Multiple array elements (takes first)
    
    Args:
        text: The raw JSON text to extract from
        field_name: The JSON field name to extract (default: "sql", can be "patched_sql")
    """
    if not text:
        return None
    
    # Normalize bytes to string (works with any LLM model)
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    elif not isinstance(text, str):
        text = str(text)
    
    # Normalize text (handle None, empty, whitespace)
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: Find the specified field and extract value using balanced quote matching
    sql_key_pattern = rf'"{field_name}"\s*:\s*'
    sql_key_match = re.search(sql_key_pattern, text, re.IGNORECASE)
    if sql_key_match:
        start_pos = sql_key_match.end()
        remaining = text[start_pos:].lstrip()
        
        # Handle null/empty: "sql": null or "sql": ""
        if remaining.startswith('null') or remaining.startswith('""'):
            return None
        
        # Try array format first: ["SQL"] or ["SQL1", "SQL2"] or [{"description": "...", "statement": "SQL"}]
        if remaining.startswith('['):
            # First, try to extract array of strings: ["SQL"]
            quote_start = remaining.find('"', 1)
            if quote_start > 0:
                extracted = _extract_quoted_string(remaining, quote_start)
                if extracted and _is_valid_sql(extracted):
                    return extracted
            # Try to extract first element from array with multiple string elements
            # Pattern: ["SQL1", "SQL2", ...]
            array_match = re.search(r'\[\s*"((?:[^"\\]|\\.)*)"', remaining, re.DOTALL)
            if array_match:
                extracted = _unescape_json_string(array_match.group(1))
                if _is_valid_sql(extracted):
                    return extracted
            # Try to extract from array of objects with statement field: [{"description": "...", "statement": "SQL"}]
            # Pattern: [{"description": "...", "statement": "SQL"}]
            statement_pattern = r'\[\s*\{[^}]*"statement"\s*:\s*"((?:[^"\\]|\\.)*)"'
            statement_match = re.search(statement_pattern, remaining, re.DOTALL | re.IGNORECASE)
            if statement_match:
                extracted = _unescape_json_string(statement_match.group(1))
                if _is_valid_sql(extracted):
                    LOGGER.debug("Extracted SQL from object with 'statement' field in array (robust extraction)")
                    return extracted
        
        # Try string format: "SQL"
        if remaining.startswith('"'):
            extracted = _extract_quoted_string(remaining, 0)
            if extracted and _is_valid_sql(extracted):
                return extracted
        
        # Handle wrong types: "sql": 123 or "sql": {...}
        # Skip these as they're not valid SQL
        if remaining.startswith(('{', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 'true', 'false')):
            LOGGER.debug("Field '%s' has wrong type (not string), skipping", field_name)
            return None
    
    # Strategy 2: Use regex patterns for common formats (fallback)
    # Array of objects with statement field: "field_name": [{"description": "...", "statement": "SQL"}]
    statement_obj_pattern = rf'"{field_name}"\s*:\s*\[\s*\{{[^}}]*"statement"\s*:\s*"((?:[^"\\]|\\.)*)"'
    sql_patterns = [
        # Array of objects with statement field
        statement_obj_pattern,
        # Array format: "field_name": ["SELECT ..."] or ["SQL1", "SQL2"]
        rf'"{field_name}"\s*:\s*\[\s*"((?:[^"\\]|\\.)*)"',
        # String format with proper escaping: "field_name": "SELECT ..."
        rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"',
        # Simple format (may fail on complex SQL): "field_name": "SELECT ..."
        rf'"{field_name}"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in sql_patterns:
        sql_match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if sql_match:
            extracted = sql_match.group(1)
            # Unescape common escape sequences
            extracted = _unescape_json_string(extracted)
            if extracted and _is_valid_sql(extracted):
                return extracted
    
    return None


def _extract_quoted_string(text: str, start_pos: int) -> Optional[str]:
    """
    Extract a quoted string starting at start_pos, handling escaped quotes and newlines.
    Returns the unescaped string content.
    """
    if start_pos >= len(text) or text[start_pos] != '"':
        return None
    
    i = start_pos + 1  # Skip opening quote
    result = []
    escaped = False
    
    while i < len(text):
        char = text[i]
        
        if escaped:
            # Handle escape sequences
            if char == 'n':
                result.append('\n')
            elif char == 't':
                result.append('\t')
            elif char == 'r':
                result.append('\r')
            elif char in ['"', '\\', '/']:
                result.append(char)
            else:
                # Unknown escape, keep as-is
                result.append('\\')
                result.append(char)
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            # Found closing quote
            extracted = ''.join(result)
            if _is_valid_sql(extracted):
                return extracted
            return None
        else:
            result.append(char)
        
        i += 1
    
    # No closing quote found (malformed JSON), but return what we have if it looks like SQL
    extracted = ''.join(result)
    if _is_valid_sql(extracted):
        return extracted
    
    return None


def _unescape_json_string(text: str) -> str:
    """Unescape common JSON escape sequences."""
    if not text:
        return text
    # Handle common escape sequences
    text = text.replace('\\"', '"')
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    text = text.replace('\\r', '\r')
    text = text.replace('\\\\', '\\')
    text = text.replace('\\/', '/')
    return text.strip()


def _is_valid_sql(text: str) -> bool:
    """
    Check if extracted text looks like valid SQL.
    Handles edge cases: empty, None, wrong types, JSON artifacts, error messages.
    """
    # Handle None, empty, wrong types
    if not text or not isinstance(text, str):
        return False
    
    text_stripped = text.strip()
    if len(text_stripped) < 10:
        return False
    
    text_upper = text_stripped.upper()
    
    # Must start with a SQL keyword (after any leading whitespace/newlines)
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
    if not any(text_upper.startswith(kw) for kw in sql_keywords):
        return False
    
    # Must contain FROM (for SELECT) or other indicators
    if 'SELECT' in text_upper and 'FROM' not in text_upper:
        # Could be a subquery, check for other patterns
        if 'WHERE' not in text_upper and 'JOIN' not in text_upper and 'UNION' not in text_upper:
            # Check if it's a CTE (WITH ... AS ...)
            if 'WITH' in text_upper and 'AS' in text_upper:
                return True
            return False
    
    # Should not be just JSON structure or error messages
    invalid_patterns = [
        '"SQL"', '{"SQL"', '"SQL":',  # JSON artifacts
        'ERROR', 'EXCEPTION', 'FAILED',  # Error messages
        'INVALID JSON', 'FOR TROUBLESHOOTING',  # LangChain error messages
        'JSON', 'PARSE',  # Parsing artifacts
    ]
    # Only reject if these patterns appear at the start (likely error messages)
    # Allow them in the middle (could be in SQL strings/comments)
    for pattern in invalid_patterns:
        if text_upper.startswith(pattern) or text_upper[:50].startswith(pattern):
            return False
    
    # Should contain at least one table reference or common SQL pattern
    sql_indicators = ['FROM', 'INTO', 'TABLE', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'UNION']
    if not any(indicator in text_upper for indicator in sql_indicators):
        # Might be a simple expression, check for basic SQL structure
        if 'SELECT' in text_upper and ('(' in text_stripped or ')' in text_stripped):
            # Could be SELECT with function calls
            return True
        return False
    
    return True


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
    
    # CRITICAL: Always get raw LLM response FIRST before parsing
    # This ensures we have the actual LLM output even if JSON parsing fails
    raw_content = ""
    prompt_part = None
    llm_part = None
    
    # Try multiple strategies to extract prompt and LLM from chain
    try:
        # Strategy 1: Check if chain has first/middle/last (RunnableSequence)
        if hasattr(chain, "first") and hasattr(chain, "middle") and hasattr(chain, "last"):
            prompt_part = chain.first
            if hasattr(chain, "middle") and len(chain.middle) > 0:
                llm_part = chain.middle[0]
            else:
                llm_part = llm
            LOGGER.debug("Extracted prompt/llm using Strategy 1 (first/middle/last)")
        # Strategy 2: Check if chain has steps (alternative structure)
        elif hasattr(chain, "steps") and isinstance(chain.steps, (list, tuple)) and len(chain.steps) >= 2:
            prompt_part = chain.steps[0]
            llm_part = chain.steps[1]
            LOGGER.debug("Extracted prompt/llm using Strategy 2 (steps)")
        # Strategy 3: Try to access internal structure (for LangChain RunnableSequence)
        elif hasattr(chain, "__dict__"):
            # Look for prompt and llm in the chain's internal structure
            chain_dict = chain.__dict__
            if "first" in chain_dict:
                prompt_part = chain_dict["first"]
            if "middle" in chain_dict and isinstance(chain_dict["middle"], (list, tuple)) and len(chain_dict["middle"]) > 0:
                llm_part = chain_dict["middle"][0]
            if prompt_part or llm_part:
                LOGGER.debug("Extracted prompt/llm using Strategy 3 (__dict__)")
        
        # ALWAYS invoke prompt|llm first to get raw response (before parsing)
        if prompt_part and llm_part:
            try:
                raw_response = (prompt_part | llm_part).invoke(payload)
                # Robust extraction: try multiple attributes and methods
                if hasattr(raw_response, "content"):
                    raw_content = raw_response.content or ""
                elif hasattr(raw_response, "text"):
                    raw_content = raw_response.text or ""
                elif hasattr(raw_response, "message"):
                    msg = raw_response.message
                    if hasattr(msg, "content"):
                        raw_content = msg.content or ""
                    elif isinstance(msg, str):
                        raw_content = msg
                elif isinstance(raw_response, str):
                    raw_content = raw_response
                elif isinstance(raw_response, dict):
                    raw_content = raw_response.get("content") or raw_response.get("text") or raw_response.get("message") or ""
                else:
                    raw_content = str(raw_response) if raw_response else ""
                
                # Normalize to string (handle bytes)
                if raw_content:
                    if isinstance(raw_content, bytes):
                        raw_content = raw_content.decode('utf-8', errors='ignore')
                    elif not isinstance(raw_content, str):
                        raw_content = str(raw_content)
                
                if not raw_content:
                    LOGGER.warning("Raw response extraction failed. Response type: %s, has content: %s, has text: %s",
                                 type(raw_response).__name__,
                                 hasattr(raw_response, "content"),
                                 hasattr(raw_response, "text"))
                
                LOGGER.debug("Captured raw LLM response BEFORE parsing (length: %d)", len(raw_content) if raw_content else 0)
            except Exception as raw_exc:
                LOGGER.warning("Failed to get raw response from prompt|llm: %s", raw_exc)
        else:
            LOGGER.warning("Could not extract prompt/llm from chain structure. Chain type: %s, has first: %s, has middle: %s, has steps: %s", 
                         type(chain).__name__, 
                         hasattr(chain, "first"), 
                         hasattr(chain, "middle"), 
                         hasattr(chain, "steps"))
    except Exception as extract_exc:
        LOGGER.warning("Failed to extract prompt/llm from chain: %s", extract_exc)
    
    # Now try to parse (this may throw an exception, but we already have raw_content)
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
            sanitized = _strip_reasoning_wrappers(raw_content)
            try:
                parsed = parser.parse(sanitized) if sanitized else {}
                if isinstance(parsed, dict):
                    parsed["__raw"] = raw_content
                return parsed
            except Exception:
                pass
        return result
    except Exception as exc:
        # Try to obtain raw content and repair it
        if not raw_content:
            # Last resort: try to invoke prompt|llm again if we have them
            if prompt_part and llm_part:
                try:
                    raw_response = (prompt_part | llm_part).invoke(payload)
                    # Use same robust extraction logic
                    if hasattr(raw_response, "content"):
                        raw_content = raw_response.content or ""
                    elif hasattr(raw_response, "text"):
                        raw_content = raw_response.text or ""
                    elif isinstance(raw_response, str):
                        raw_content = raw_response
                    else:
                        raw_content = str(raw_response) if raw_response else ""
                except Exception:
                    pass
        
        # Try to extract raw response from exception if we don't have it yet
        if not raw_content:
            # LangChain's OutputParserException may have the response in various attributes
            # Try partial_output first (common in OutputParserException)
            if hasattr(exc, "partial_output") and exc.partial_output:
                if isinstance(exc.partial_output, str):
                    raw_content = exc.partial_output
                elif hasattr(exc.partial_output, "content"):
                    raw_content = getattr(exc.partial_output, "content", "")
            # Try llm_output attribute
            elif hasattr(exc, "llm_output") and exc.llm_output:
                if isinstance(exc.llm_output, dict) and "text" in exc.llm_output:
                    raw_content = exc.llm_output["text"]
                elif isinstance(exc.llm_output, str):
                    raw_content = exc.llm_output
            # Try response attribute
            elif hasattr(exc, "response") and exc.response:
                if hasattr(exc.response, "content"):
                    raw_content = getattr(exc.response, "content", "")
                elif isinstance(exc.response, str):
                    raw_content = exc.response
            # Check if exception args contain the response (first arg is often the response)
            elif hasattr(exc, "args") and exc.args and len(exc.args) > 0:
                first_arg = exc.args[0]
                if isinstance(first_arg, str) and len(first_arg) > 100 and "Invalid json output" not in first_arg:
                    # If first arg is a long string and not just an error message, it might be the response
                    raw_content = first_arg
                elif hasattr(first_arg, "content"):
                    raw_content = getattr(first_arg, "content", "")
            # Check if exception has a message that contains the raw response
            elif hasattr(exc, "message") and exc.message:
                msg_str = str(exc.message)
                # Only use message if it's not just an error message
                if len(msg_str) > 100 and "Invalid json output" not in msg_str and "For troubleshooting" not in msg_str:
                    raw_content = msg_str
            # Check if exception has raw_response attribute
            elif hasattr(exc, "raw_response"):
                raw_content = str(exc.raw_response)
        
        # Normalize raw_content to string (handle bytes)
        if raw_content:
            if isinstance(raw_content, bytes):
                raw_content = raw_content.decode('utf-8', errors='ignore')
            elif not isinstance(raw_content, str):
                raw_content = str(raw_content)
        
        # Filter out error messages - don't use exception string if it's just an error message
        if raw_content and ("Invalid json output" in raw_content or "For troubleshooting" in raw_content):
            # This is just an error message, not the actual LLM response
            LOGGER.warning("Raw content appears to be an error message, not LLM response. Length: %d", len(raw_content))
            raw_content = ""  # Clear it so we try other methods
        
        sanitized = _strip_reasoning_wrappers(raw_content)
        if not sanitized:
            LOGGER.warning("Could not extract raw LLM response for JSON repair. Exception type: %s, Exception: %s", type(exc).__name__, exc)
            # If we still don't have raw_content, try one more time to invoke prompt|llm directly
            if not raw_content and prompt_part and llm_part:
                try:
                    LOGGER.debug("Making final attempt to get raw response from prompt|llm")
                    raw_response = (prompt_part | llm_part).invoke(payload)
                    # Use same robust extraction logic
                    if hasattr(raw_response, "content"):
                        raw_content = raw_response.content or ""
                    elif hasattr(raw_response, "text"):
                        raw_content = raw_response.text or ""
                    elif isinstance(raw_response, str):
                        raw_content = raw_response
                    else:
                        raw_content = str(raw_response) if raw_response else ""
                    
                    # Normalize to string (handle bytes)
                    if raw_content:
                        if isinstance(raw_content, bytes):
                            raw_content = raw_content.decode('utf-8', errors='ignore')
                        elif not isinstance(raw_content, str):
                            raw_content = str(raw_content)
                    
                    if raw_content:
                        LOGGER.info("Successfully captured raw response on final attempt (length: %d)", len(raw_content))
                        sanitized = _strip_reasoning_wrappers(raw_content)
                except Exception as final_exc:
                    LOGGER.warning("Final attempt to get raw response also failed: %s", final_exc)
            
            if not sanitized:
                # Return a structure that matches what the caller expects based on context
                # For repair calls, return patched_sql structure; for critic, return verdict structure
                return {"patched_sql": "", "what_changed": [], "why": f"Failed to extract raw LLM response. Exception: {type(exc).__name__}: {exc}", "__raw": None}
        
        try:
            if fixer is not None:
                parsed = fixer.parse(sanitized)
                # Attach raw text for downstream logging
                if isinstance(parsed, dict):
                    parsed["__raw"] = raw_content
                return parsed
            # Local best-effort repair if OutputFixingParser is unavailable
            repaired = _best_effort_json_repair(sanitized)
            repaired["__raw"] = raw_content
            return repaired
        except Exception as repair_exc:
            LOGGER.debug("JSON repair failed: %s. Raw content: %s", repair_exc, raw_content[:500] if raw_content else "<<empty>>")
            # Return appropriate structure based on context
            return {"patched_sql": "", "what_changed": [], "why": f"JSON repair failed: {repair_exc}", "__raw": raw_content}


def _best_effort_json_repair(text: str) -> Dict[str, Any]:
    """
    Minimal, generic JSON repair:
    - Extract first {...} block
    - Replace single quotes with double quotes when safe
    - Remove trailing commas before } or ]
    - Attempt json.loads; fallback to an empty structure
    """
    cleaned = _strip_reasoning_wrappers(text)
    candidate = _extract_first_json_object(cleaned)
    if candidate:
        text = candidate
    else:
        text = cleaned
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
    desired_columns: List[str]  # Validated column names (for hints)
    desired_columns_raw: List[str]  # Raw user input (for LLM to map synonyms/aliases)
    query_analysis: Dict[str, Any]  # Analysis results: todo_list, verification_checklist, potential_pitfalls, etc.


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


def _build_rich_graph_context_for_repair(state: QueryState) -> str:
    """
    Build a rich Graph Context for repair steps that includes:
    - Table cards with full column details
    - Column cards with types, nullable, sample values
    - Column facts (distinct counts, min/max)
    - Relationship map (FK paths)
    - Value anchors (sample values for filters)
    
    This provides much more context than the simplified schema_markdown.
    """
    prompt_inputs = state.get("prompt_inputs") or {}
    
    # Build rich context from prompt_inputs (same as planner gets)
    parts = []
    
    # Table cards (with full column details)
    table_cards = prompt_inputs.get("table_cards", "")
    if table_cards and table_cards != "None":
        parts.append("TABLE CARDS:")
        parts.append(table_cards)
        parts.append("")
    
    # Column cards (with types, nullable, sample values)
    column_cards = prompt_inputs.get("column_cards", "")
    if column_cards and column_cards != "None":
        parts.append("COLUMN CARDS:")
        parts.append(column_cards)
        parts.append("")
    
    # Column facts (distinct counts, min/max, null%)
    column_facts = prompt_inputs.get("column_facts", "")
    if column_facts and column_facts != "None":
        parts.append("COLUMN FACTS (type, null%, distinct, min/max):")
        parts.append(column_facts)
        parts.append("")
    
    # Relationship map (FK paths)
    relationship_map = prompt_inputs.get("relationship_map", "")
    if relationship_map and relationship_map != "None":
        parts.append("FOREIGN-KEY PATHS (high-confidence joins):")
        parts.append(relationship_map)
        parts.append("")
    
    # Value anchors (sample values for realistic filters)
    value_anchors = prompt_inputs.get("value_anchors", "")
    if value_anchors and value_anchors != "No value anchors collected.":
        parts.append("VALUE ANCHORS (representative values for realistic filters):")
        parts.append(value_anchors)
        parts.append("")
    
    # Fallback to simplified schema_markdown if rich context not available
    if not parts:
        schema_markdown = state.get("schema_markdown", "")
        if schema_markdown:
            parts.append("SCHEMA SUMMARY:")
            parts.append(schema_markdown)
    
    return "\n".join(parts) if parts else "No Graph Context available."


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


def create_query_workflow(llm: Any, engine: Engine, max_repair_iterations: int = 2) -> Any:
    """
    Compile a LangGraph for the SQL query workflow.
    
    Args:
        llm: The language model to use for SQL generation and repair
        engine: The database engine for SQL execution
        max_repair_iterations: Maximum number of repair iterations (default: 2)
    """
    # Set the global max repairs for this workflow instance
    global MAX_INTENT_REPAIRS
    MAX_INTENT_REPAIRS = max_repair_iterations

    graph = StateGraph(QueryState)
    graph.add_node("analyze", _analyze_query(llm))
    graph.add_node("plan", _plan_sql(llm))
    graph.add_node("intent_critic", _intent_critic(llm))
    graph.add_node("intent_repair", _intent_repair(llm))
    graph.add_node("execute", _execute_sql(engine))
    graph.add_node("critic", _critic_sql(llm))
    graph.add_node("repair", _repair_sql(llm))
    graph.add_node("summarise", _summarise(llm))
    graph.add_node("fail", _fail())

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "plan")
    # Route after planning: if no SQL, fail early with a clear message
    def _route_after_plan(state: QueryState) -> str:
        """
        Route after planner node.
        - If SQL is empty after all extraction attempts (planner, robust extraction, bootstrap, fallback):
          FAIL EARLY - no point going to critic/repair if we have no SQL at all.
        - If SQL exists (even if incorrect/wrong), proceed to critic/repair - they will handle fixing it.
        """
        plan = state.get("plan") or {}
        sql = plan.get("sql")
        
        # Check if SQL is empty after all extraction attempts
        sql_is_empty = False
        if not sql:
            sql_is_empty = True
        elif isinstance(sql, str):
            sql_is_empty = not sql.strip()
        elif isinstance(sql, list):
            sql_is_empty = not any((s or "").strip() for s in sql)
        
        if sql_is_empty:
            # Planner failed to produce SQL even after robust extraction, bootstrap, and fallback
            error_msg = plan.get("sql_generation_note") or "Planner did not produce proper SQL. The LLM failed to generate SQL even after multiple extraction attempts."
            state["execution_error"] = error_msg
            LOGGER.warning("Planner returned empty SQL after all extraction attempts. Failing early instead of proceeding to critic/repair. Error: %s", error_msg)
            return "fail"
        
        # SQL exists (even if it might be incorrect) - let critic/repair handle it
        LOGGER.debug("Planner produced SQL (will be validated by critic/repair). SQL preview: %s", 
                    (sql[0] if isinstance(sql, list) else sql)[:200] if sql else "")
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


def _analyze_query(llm: Any):
    """
    Analyze the user question and Graph Context to create a TODO list and verification checklist.
    This helps the planner and repair steps avoid common mistakes.
    """
    parser = JsonOutputParser()
    chain = ANALYSIS_PROMPT | llm | parser

    def node(state: QueryState) -> QueryState:
        prompt_inputs = state.get("prompt_inputs") or {}
        question = state.get("question", "")
        desired_columns = state.get("desired_columns") or []
        desired_columns_raw = state.get("desired_columns_raw") or []
        
        # Build rich Graph Context for analysis
        graph_text = _build_rich_graph_context_for_repair(state)
        
        result = _safe_invoke_json(
            chain,
            {
                "question": question,
                "graph": graph_text,
                "dialect": state.get("dialect", ""),
                "desired_columns_section": _format_desired_columns_section(desired_columns, desired_columns_raw),
            },
            llm,
        )
        
        # Store analysis results in state
        analysis = {
            "analysis": result.get("analysis", ""),
            "todo_list": result.get("todo_list", []),
            "verification_checklist": result.get("verification_checklist", []),
            "potential_pitfalls": result.get("potential_pitfalls", []),
            "complex_requirements": result.get("complex_requirements", []),
        }
        
        LOGGER.info("Query analysis completed: %d TODO items, %d checklist items, %d pitfalls identified",
                   len(analysis.get("todo_list", [])),
                   len(analysis.get("verification_checklist", [])),
                   len(analysis.get("potential_pitfalls", [])))
        
        return {
            "query_analysis": analysis,
        }
    
    return node


def _plan_sql(llm: Any):
    parser = JsonOutputParser()
    prompt = agent_prompt()
    chain = prompt | llm | parser

    def node(state: QueryState) -> QueryState:
        LOGGER.debug("Planning SQL for question=%s", state.get("question"))
        prompt_inputs = dict(state.get("prompt_inputs") or {})
        prompt_inputs.setdefault("user_question", state.get("question", ""))
        
        # Add desired columns formatting (use raw input for LLM synonym mapping)
        desired_columns = state.get("desired_columns") or []
        desired_columns_raw = state.get("desired_columns_raw") or []
        prompt_inputs["desired_columns_hint"] = _format_desired_columns_hint(desired_columns, desired_columns_raw)
        prompt_inputs["desired_columns_section"] = _format_desired_columns_section(desired_columns, desired_columns_raw)
        
        # Add query analysis (TODO list and checklist) to help planner avoid mistakes
        query_analysis = state.get("query_analysis") or {}
        prompt_inputs["query_analysis_section"] = _format_query_analysis_section(query_analysis)
        
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
        # CRITICAL: Always capture raw response BEFORE parsing to ensure we have actual LLM output
        raw_planner_text = ""
        sanitized_planner_text = ""
        try:
            # Capture raw response first for diagnostics (BEFORE parsing)
            raw_response = (prompt | llm).invoke(prompt_inputs)
            
            # Log the raw response object for debugging BEFORE extraction
            LOGGER.debug("Planner response object type: %s, repr: %s", type(raw_response).__name__, repr(raw_response)[:500])
            
            # Robust extraction: try multiple attributes and methods
            if hasattr(raw_response, "content"):
                # Handle both string and list content (multimodal)
                content = raw_response.content
                if isinstance(content, list):
                    # Extract text from multimodal content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "image_url":
                                # Skip images, but log them
                                LOGGER.debug("Skipping image block in response")
                        elif isinstance(block, str):
                            text_parts.append(block)
                    raw_planner_text = "".join(text_parts)
                elif isinstance(content, str):
                    raw_planner_text = content
                else:
                    raw_planner_text = str(content) if content else ""
                # Normalize to string (handle bytes)
                if raw_planner_text:
                    if isinstance(raw_planner_text, bytes):
                        raw_planner_text = raw_planner_text.decode('utf-8', errors='ignore')
                    elif not isinstance(raw_planner_text, str):
                        raw_planner_text = str(raw_planner_text)
                
                LOGGER.debug("Extracted from .content attribute (length: %d, type: %s)", 
                           len(raw_planner_text) if raw_planner_text else 0, type(content).__name__)
            elif hasattr(raw_response, "text"):
                raw_planner_text = raw_response.text or ""
                LOGGER.debug("Extracted from .text attribute (length: %d)", len(raw_planner_text))
            elif hasattr(raw_response, "message"):
                msg = raw_response.message
                if hasattr(msg, "content"):
                    raw_planner_text = msg.content or ""
                    LOGGER.debug("Extracted from .message.content (length: %d)", len(raw_planner_text))
                elif isinstance(msg, str):
                    raw_planner_text = msg
                    LOGGER.debug("Extracted from .message as string (length: %d)", len(raw_planner_text))
            elif isinstance(raw_response, str):
                raw_planner_text = raw_response
                LOGGER.debug("Response is string (length: %d)", len(raw_planner_text))
            elif isinstance(raw_response, dict):
                # Try common keys
                raw_planner_text = raw_response.get("content") or raw_response.get("text") or raw_response.get("message") or ""
                LOGGER.debug("Extracted from dict (length: %d), keys: %s", len(raw_planner_text), list(raw_response.keys())[:10])
            else:
                # Last resort: convert to string
                raw_planner_text = str(raw_response) if raw_response else ""
                LOGGER.debug("Extracted via str() conversion (length: %d)", len(raw_planner_text))
            
            # Also check response_metadata for raw response (some LangChain versions store it there)
            if not raw_planner_text and hasattr(raw_response, "response_metadata"):
                try:
                    metadata = raw_response.response_metadata
                    if isinstance(metadata, dict):
                        # Check for raw response in metadata
                        if "raw_response" in metadata:
                            raw_resp = metadata["raw_response"]
                            if hasattr(raw_resp, "choices") and len(raw_resp.choices) > 0:
                                choice = raw_resp.choices[0]
                                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                                    raw_planner_text = choice.message.content or ""
                                    LOGGER.debug("Extracted from response_metadata.raw_response.choices[0].message.content (length: %d)", len(raw_planner_text))
                except Exception as meta_exc:
                    LOGGER.debug("Failed to extract from response_metadata: %s", meta_exc)
            
            # Normalize raw_planner_text to string (handle bytes) before any string operations
            if raw_planner_text:
                if isinstance(raw_planner_text, bytes):
                    raw_planner_text = raw_planner_text.decode('utf-8', errors='ignore')
                elif not isinstance(raw_planner_text, str):
                    raw_planner_text = str(raw_planner_text)
            
            # Log response type for debugging if extraction failed
            if not raw_planner_text:
                LOGGER.warning("Planner response extraction failed. Response type: %s, has content: %s, has text: %s, str length: %d",
                             type(raw_response).__name__,
                             hasattr(raw_response, "content"),
                             hasattr(raw_response, "text"),
                             len(str(raw_response)) if raw_response else 0)
                # Try to inspect the object more deeply
                if hasattr(raw_response, "__dict__"):
                    LOGGER.debug("Response object attributes: %s", list(raw_response.__dict__.keys()))
                    # Log all attribute values (truncated)
                    for attr in list(raw_response.__dict__.keys())[:10]:
                        try:
                            val = getattr(raw_response, attr)
                            if isinstance(val, str):
                                LOGGER.debug("  %s = %s (length: %d)", attr, val[:200], len(val))
                            else:
                                LOGGER.debug("  %s = %s (type: %s)", attr, repr(val)[:200], type(val).__name__)
                        except Exception:
                            pass
                # Also try dir() to see all available methods/attributes
                try:
                    all_attrs = [x for x in dir(raw_response) if not x.startswith('_')]
                    LOGGER.debug("Response object public attributes/methods: %s", all_attrs[:20])
                except Exception:
                    pass
            
            # Log the actual content (first 1000 chars) to see if SQL is there
            if raw_planner_text:
                LOGGER.info("Planner raw LLM response (length: %d): %s", 
                           len(raw_planner_text),
                           raw_planner_text[:1000])
                # Check if SQL keywords are present
                sql_keywords = ['SELECT', 'WITH', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY']
                found_keywords = [kw for kw in sql_keywords if kw in raw_planner_text.upper()]
                if found_keywords:
                    LOGGER.info("SQL keywords found in response: %s", found_keywords)
                else:
                    LOGGER.warning("No SQL keywords found in raw response")
            else:
                LOGGER.warning("Planner raw LLM response is EMPTY (length: 0)")
            
            # Filter out error messages - don't use if it's just an error message
            if raw_planner_text and ("Invalid json output" in raw_planner_text or "For troubleshooting" in raw_planner_text):
                LOGGER.warning("Planner raw response appears to be an error message, not LLM output. Attempting to re-invoke.")
                # Try to get actual response
                try:
                    raw_response = (prompt | llm).invoke(prompt_inputs)
                    # Use same robust extraction logic
                    if hasattr(raw_response, "content"):
                        raw_planner_text = raw_response.content or ""
                    elif hasattr(raw_response, "text"):
                        raw_planner_text = raw_response.text or ""
                    elif isinstance(raw_response, str):
                        raw_planner_text = raw_response
                    else:
                        raw_planner_text = str(raw_response) if raw_response else ""
                except Exception:
                    pass
            
            sanitized_planner_text = _strip_reasoning_wrappers(raw_planner_text)
            if not sanitized_planner_text.strip():
                LOGGER.warning("Planner LLM returned empty response. This may indicate a model compatibility issue.")
            plan = parser.parse(sanitized_planner_text)
        except Exception as parse_exc:
            LOGGER.debug("Planner JSON parsing failed: %s. Raw response length: %d", parse_exc, len(raw_planner_text) if raw_planner_text else 0)
            # Fallback: attempt repair using raw response we already captured
            try:
                if not raw_planner_text:
                    # Re-invoke if we don't have raw text yet (shouldn't happen, but safety check)
                    LOGGER.debug("Re-invoking planner to get raw response")
                    raw_response = (prompt | llm).invoke(prompt_inputs)
                    # Use same robust extraction logic
                    if hasattr(raw_response, "content"):
                        raw_planner_text = raw_response.content or ""
                    elif hasattr(raw_response, "text"):
                        raw_planner_text = raw_response.text or ""
                    elif isinstance(raw_response, str):
                        raw_planner_text = raw_response
                    else:
                        raw_planner_text = str(raw_response) if raw_response else ""
                
                # Normalize to string (handle bytes) before string operations
                if raw_planner_text:
                    if isinstance(raw_planner_text, bytes):
                        raw_planner_text = raw_planner_text.decode('utf-8', errors='ignore')
                    elif not isinstance(raw_planner_text, str):
                        raw_planner_text = str(raw_planner_text)
                
                # Filter out error messages
                if raw_planner_text and ("Invalid json output" in raw_planner_text or "For troubleshooting" in raw_planner_text):
                    LOGGER.warning("Planner raw response is an error message. This should not happen if we captured it before parsing.")
                    raw_planner_text = ""  # Clear it
                
                sanitized_planner_text = _strip_reasoning_wrappers(raw_planner_text) if raw_planner_text else ""
                if sanitized_planner_text.strip():
                    if OutputFixingParser is not None:
                        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)  # type: ignore[call-arg]
                        plan = fixing_parser.parse(sanitized_planner_text)
                    else:
                        plan = _best_effort_json_repair(sanitized_planner_text)  # type: ignore[assignment]
                else:
                    LOGGER.warning("Planner LLM returned empty response after retry. Model may not support this prompt format.")
                    plan = {"sql": "", "plan": {}, "rationale_summary": "", "tests": [], "summary": "", "followups": []}
            except Exception as repair_exc:
                LOGGER.warning("Planner JSON repair also failed: %s. Raw response length: %d", repair_exc, len(raw_planner_text) if raw_planner_text else 0)
                plan = {"sql": "", "plan": {}, "rationale_summary": "", "tests": [], "summary": "", "followups": []}
        if isinstance(plan, dict):
            sql_statements = plan.get("sql")
            
            # Normalize SQL: handle arrays, null, wrong types, etc.
            if sql_statements is not None:
                # Handle array format: ["SELECT ..."] or ["SQL1", "SQL2"] or [{"description": "...", "statement": "SQL"}]
                if isinstance(sql_statements, list):
                    if len(sql_statements) > 0:
                        first_elem = sql_statements[0]
                        # Handle array of strings: ["SELECT ..."]
                        if isinstance(first_elem, str) and first_elem.strip():
                            sql_statements = first_elem
                        # Handle array of objects with statement field: [{"description": "...", "statement": "SQL"}]
                        elif isinstance(first_elem, dict) and "statement" in first_elem:
                            statement = first_elem.get("statement", "")
                            if isinstance(statement, str) and statement.strip():
                                LOGGER.info("Extracted SQL from object with 'statement' field in array (description: %s)", first_elem.get("description", "N/A"))
                                sql_statements = statement
                            else:
                                LOGGER.warning("Object in SQL array has 'statement' field but it's not a valid string, using robust extraction")
                                sql_statements = None
                        else:
                            LOGGER.warning("SQL array contains non-string/non-object elements (type: %s), using robust extraction", type(first_elem).__name__)
                            sql_statements = None
                    else:
                        sql_statements = None
                # Handle wrong types: number, object, etc.
                elif not isinstance(sql_statements, str):
                    LOGGER.warning("SQL field has wrong type (%s), expected string. Using robust extraction.", type(sql_statements).__name__)
                    sql_statements = None
                # Handle empty/whitespace strings
                elif not sql_statements.strip():
                    sql_statements = None
            
            # If SQL is empty but raw response contains SQL, try to extract it
            if not sql_statements and raw_planner_text:
                # Try to extract SQL from raw response using multiple strategies
                # Strategy 1: Try to extract first JSON object and parse it manually
                json_obj_text = _extract_first_json_object(sanitized_planner_text if sanitized_planner_text else raw_planner_text)
                if json_obj_text:
                    try:
                        # Try to parse as JSON (might fail if SQL has unescaped quotes)
                        temp_plan = json.loads(json_obj_text)
                        if temp_plan.get("sql"):
                            extracted_sql = temp_plan["sql"]
                            # Normalize: handle arrays, null, wrong types, objects with statement field
                            if isinstance(extracted_sql, list) and len(extracted_sql) > 0:
                                first_elem = extracted_sql[0]
                                # Handle array of strings: ["SELECT ..."]
                                if isinstance(first_elem, str):
                                    extracted_sql = first_elem
                                # Handle array of objects with statement field: [{"description": "...", "statement": "SQL"}]
                                elif isinstance(first_elem, dict) and "statement" in first_elem:
                                    extracted_sql = first_elem.get("statement", "")
                                    LOGGER.info("Extracted SQL from object with 'statement' field in fallback extraction")
                                else:
                                    extracted_sql = None
                            if isinstance(extracted_sql, str) and extracted_sql.strip():
                                LOGGER.info("Extracted SQL from JSON object after initial parse failed: %s", extracted_sql[:200])
                                plan["sql"] = extracted_sql
                                sql_statements = extracted_sql
                            else:
                                LOGGER.debug("Extracted SQL from JSON object is invalid (type: %s, empty: %s)", 
                                           type(extracted_sql).__name__, not extracted_sql if isinstance(extracted_sql, str) else True)
                    except Exception as json_exc:
                        LOGGER.debug("Failed to parse extracted JSON object: %s", json_exc)
                
                # Strategy 2: Use robust extraction to find SQL field (handles malformed JSON)
                if not sql_statements:
                    extracted_sql = _extract_sql_from_malformed_json(raw_planner_text)
                    if extracted_sql:
                        LOGGER.info("Extracted SQL from raw response using robust parser: %s", extracted_sql[:200])
                        plan["sql"] = extracted_sql
                        sql_statements = extracted_sql
            if not sql_statements:
                # Log raw response for debugging when SQL is empty
                if raw_planner_text:
                    LOGGER.info("Planner returned empty SQL. Raw LLM response: %s", raw_planner_text[:1000])
                else:
                    LOGGER.warning("Planner returned empty SQL and no raw response was captured. Model may have failed silently.")
                
                # Planner failed to produce SQL after all attempts (parsing + robust extraction)
                # Try domain fallback as last resort (regex-based pattern matching, no LLM call)
                LOGGER.info("Planner returned no SQL. Trying domain-specific fallback SQL generation (regex pattern matching)...")
                question = state.get("question", "")
                graph_context = state.get("graph_context")
                schema = state.get("schema_name")
                dialect = state.get("dialect", "")
                fallback_sql = maybe_generate_domain_sql(question, graph_context, schema, dialect)
                if fallback_sql:
                    plan["sql"] = fallback_sql[0] if isinstance(fallback_sql, list) else fallback_sql
                    plan.setdefault("notes", []).append("Used domain-specific fallback SQL (regex pattern matching) due to planner failure.")
                    LOGGER.info("Domain fallback generated SQL: %s", plan["sql"][:200])
                else:
                    plan["sql_generation_note"] = "Planner returned no SQL and domain fallback found no matching pattern."
                    # Surface a clear error for routing to fail node
                    state["execution_error"] = "Planner failed to produce SQL after all parsing attempts, and domain fallback (regex pattern matching) found no match for the question pattern."
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
        desired_columns = state.get("desired_columns") or []
        desired_columns_raw = state.get("desired_columns_raw") or []
        query_analysis = state.get("query_analysis") or {}
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "question": state.get("question", ""),
                "plan": plan_summary,
                "sql": sql_text,
                "desired_columns_section": _format_desired_columns_section(desired_columns, desired_columns_raw),
                "query_analysis_section": _format_query_analysis_section(query_analysis),
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
        # Use rich Graph Context (table cards, column cards, relationships, value anchors)
        graph_text = _build_rich_graph_context_for_repair(state)
        desired_columns = state.get("desired_columns") or []
        desired_columns_raw = state.get("desired_columns_raw") or []
        query_analysis = state.get("query_analysis") or {}
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "sql": sql_text,
                "error": error_text,
                "repair_hints": critic.get("repair_hints", []),
                "plan": plan.get("plan") or {},
                "desired_columns_section": _format_desired_columns_section(desired_columns, desired_columns_raw),
                "query_analysis_section": _format_query_analysis_section(query_analysis),
            },
            llm,
        )
        LOGGER.info("Intent repair attempt %d result received", attempts + 1)
        patched_sql = result.get("patched_sql")
        raw_response = result.get("__raw")
        
        # Normalize patched_sql: handle arrays, null, wrong types, etc.
        if patched_sql is not None:
            # Handle array format: ["SELECT ..."]
            if isinstance(patched_sql, list):
                if len(patched_sql) > 0:
                    first_elem = patched_sql[0]
                    if isinstance(first_elem, str) and first_elem.strip():
                        patched_sql = first_elem
                    else:
                        LOGGER.warning("patched_sql array contains non-string elements, using robust extraction")
                        patched_sql = None
                else:
                    patched_sql = None
            # Handle wrong types: number, object, etc.
            elif not isinstance(patched_sql, str):
                LOGGER.warning("patched_sql has wrong type (%s), expected string. Using robust extraction.", type(patched_sql).__name__)
                patched_sql = None
            # Handle empty/whitespace strings
            elif not patched_sql.strip():
                patched_sql = None
        
        # If patched_sql is empty but we have raw response, try robust extraction
        if (not patched_sql or not patched_sql.strip()) and raw_response:
            # Normalize raw_response to string (handle bytes from any LLM model)
            if isinstance(raw_response, bytes):
                raw_response = raw_response.decode('utf-8', errors='ignore')
            elif not isinstance(raw_response, str):
                raw_response = str(raw_response)
            LOGGER.debug("Intent repair returned empty patched_sql, attempting robust extraction from raw response (length: %d)", len(raw_response) if raw_response else 0)
            extracted_sql = _extract_sql_from_malformed_json(raw_response, field_name="patched_sql")
            if extracted_sql:
                LOGGER.info("Extracted patched_sql from raw response using robust parser in intent repair: %s", extracted_sql[:200])
                patched_sql = extracted_sql
            else:
                # Log what we tried to extract from for debugging
                LOGGER.debug("Robust extraction failed. Raw response preview: %s", raw_response[:500] if raw_response and len(raw_response) > 500 else raw_response)
        
        if patched_sql and patched_sql.strip() and patched_sql.strip() != sql_text.strip():
            plan["sql"] = patched_sql
            plan.setdefault("notes", []).append(
                f"Intent repair iteration {attempts + 1} applied."
            )
            LOGGER.info("Intent repair attempt %d successfully patched SQL", attempts + 1)
            state["plan"] = plan
        else:
            if not patched_sql or not patched_sql.strip():
                if raw_response:
                    LOGGER.warning("Intent repair attempt %d returned empty patched_sql. Raw LLM response (first 1000 chars): %s", attempts + 1, raw_response[:1000] if len(raw_response) > 1000 else raw_response)
                else:
                    LOGGER.warning("Intent repair attempt %d returned empty patched_sql and no raw response captured", attempts + 1)
            else:
                LOGGER.info("Intent repair attempt %d produced no effective change (SQL unchanged)", attempts + 1)
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
        
        # Fix incorrect schema prefixes (e.g., on CTEs, double prefixes, column references)
        schema_name = state.get("schema_name")
        if schema_name:
            for idx, sql in enumerate(sql_statements):
                # Log before fix for debugging
                if "AGENT_DEMO.AGENT_DEMO" in sql or f"{schema_name}.{schema_name}" in sql:
                    LOGGER.debug("Detected multiple schema prefixes in SQL before fix. SQL preview: %s", sql[:300])
                fixed = _fix_incorrect_schema_prefixes(sql, schema_name)
                if fixed != sql:
                    sql_statements[idx] = fixed
                    LOGGER.info("Fixed incorrect schema prefixes in statement %d. Preview: %s", idx + 1, fixed[:200])
                else:
                    # Log if no changes were made but we expected changes
                    if "AGENT_DEMO.AGENT_DEMO" in sql or f"{schema_name}.{schema_name}" in sql:
                        LOGGER.warning("Expected to fix schema prefixes but no changes were made. SQL: %s", sql[:300])

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
    Skips CTEs and already-qualified tables.
    """
    if not schema_name:
        return sql_text, False
    try:
        ast = sqlglot.parse_one(sql_text, read=None)
    except Exception:
        return sql_text, False

    # Collect CTE names (including any that might have incorrect schema prefixes)
    cte_names: Set[str] = set()
    for cte in ast.find_all(exp.CTE):
        alias = getattr(getattr(cte, "alias", None), "name", None)
        name = alias or getattr(getattr(cte, "this", None), "name", None)
        if name:
            # Normalize: remove schema prefix if present (CTEs shouldn't have them)
            normalized_name = name.lower()
            if "." in normalized_name:
                # Extract just the CTE name (last part after dots)
                normalized_name = normalized_name.split(".")[-1]
            cte_names.add(normalized_name)

    changed = False
    for table in ast.find_all(exp.Table):
        # Skip if already has schema qualification
        if table.args.get("db"):
            # Check if it's a double-qualification (schema.schema.table) and fix it
            existing_db = str(table.args.get("db", "")).lower()
            if existing_db == schema_name.lower():
                # Already correctly qualified, skip
                continue
            # If it has a different schema, leave it alone
            continue
        
        identifier = getattr(table, "this", None)
        if not isinstance(identifier, exp.Identifier):
            continue
        table_name = (identifier.name or "").lower()
        if not table_name:
            continue
        
        # Skip if this is a CTE (check both full name and base name)
        if table_name in cte_names:
            continue
        # Also check if table_name is the base part of any CTE
        if any(cte_name.endswith(f".{table_name}") or cte_name == table_name for cte_name in cte_names):
            continue
        
        # Only qualify if it's a known table
        if known_tables and table_name not in known_tables:
            continue
        
        table.set("db", exp.to_identifier(schema_name))
        changed = True
    
    # Also fix CTEs that incorrectly have schema prefixes
    for cte in ast.find_all(exp.CTE):
        alias = getattr(cte, "alias", None)
        if alias and hasattr(alias, "name"):
            cte_name = alias.name
            # If CTE name has schema prefix, remove it
            if "." in cte_name and not cte_name.startswith("("):
                # Extract just the CTE name
                parts = cte_name.split(".")
                # If it looks like schema.schema.cte_name or schema.cte_name, use last part
                if len(parts) > 1:
                    # Check if first part matches schema (likely incorrect qualification)
                    if parts[0].upper() == schema_name.upper():
                        alias.name = parts[-1]  # Use just the CTE name
                        changed = True
    
    return (ast.sql() if changed else sql_text), changed


def _fix_incorrect_schema_prefixes(sql_text: str, schema_name: str) -> str:
    """
    Fix common schema qualification errors:
    - Remove schema prefixes from CTE names (e.g., AGENT_DEMO.recent_test_sets -> recent_test_sets)
    - Fix double/triple schema prefixes (e.g., AGENT_DEMO.AGENT_DEMO.table -> AGENT_DEMO.table)
    - Fix schema prefixes on column references (e.g., e.AGENT_DEMO.run_at -> e.run_at)
    - Fix CTE references in FROM/JOIN clauses that have schema prefixes
    
    Uses regex preprocessing to fix CTE names before sqlglot parsing, then uses sqlglot
    for more precise fixes on table and column references.
    """
    if not schema_name:
        return sql_text
    
    import re
    
    # STEP 1: Regex preprocessing to fix CTE names with schema prefixes BEFORE parsing
    # This handles cases where sqlglot might not parse malformed CTE names correctly
    # Pattern: WITH schema.schema.schema.cte_name AS or WITH schema.cte_name AS
    # Match: WITH followed by one or more schema. prefixes, then CTE name, then AS
    # IMPORTANT: Match ALL CTEs in the WITH clause, not just the first one
    # Pattern matches: WITH schema.cte1 AS ..., schema.cte2 AS ... (handles comma-separated CTEs)
    # Use \s* to handle optional whitespace/newlines
    cte_pattern = rf'\b((?:WITH|,)\s*)((?:{re.escape(schema_name)}\.)+)(\w+)\s+AS\b'
    
    def fix_cte_name(match):
        prefix = match.group(1)  # "WITH " or ", "
        schema_prefix = match.group(2)  # e.g., "AGENT_DEMO.AGENT_DEMO.AGENT_DEMO."
        cte_name = match.group(3)  # e.g., "monthly_stats"
        # Preserve whitespace in prefix
        return f'{prefix}{cte_name} AS'
    
    sql_text = re.sub(cte_pattern, fix_cte_name, sql_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Also fix CTE references in FROM/JOIN clauses (but only if they're not actual tables)
    # We'll collect CTE names first, then fix references
    cte_names_set = set()
    # Extract CTE names from WITH clauses (after our fix above)
    # Match both: WITH cte1 AS ... and , cte2 AS ... (handles comma-separated CTEs)
    cte_def_pattern = r'\b(?:WITH|,)\s+(\w+)\s+AS\b'
    for match in re.finditer(cte_def_pattern, sql_text, re.IGNORECASE):
        cte_names_set.add(match.group(1).lower())
    
    # Fix CTE references with schema prefixes in table contexts (FROM/JOIN/UPDATE/DELETE)
    # Pattern: FROM schema.schema.cte_name or JOIN schema.cte_name (where cte_name is a known CTE)
    # Also fix in subqueries: (SELECT ... FROM schema.cte_name)
    for cte_name in cte_names_set:
        # Match schema prefix(es) followed by the CTE name in FROM/JOIN contexts
        # Use word boundaries to avoid partial matches
        cte_ref_pattern = rf'\b((?:FROM|JOIN|UPDATE|DELETE\s+FROM)\s+)((?:{re.escape(schema_name)}\.)+){re.escape(cte_name)}\b'
        def fix_cte_ref(match):
            keyword = match.group(1).strip()  # "FROM", "JOIN", etc.
            return f'{keyword} {cte_name}'
        sql_text = re.sub(cte_ref_pattern, fix_cte_ref, sql_text, flags=re.IGNORECASE)
        
        # Also fix CTE references in subqueries and other table contexts
        # Pattern: schema.schema.cte_name when used as a table (after FROM, JOIN, or in subqueries)
        # Be more specific: only replace if it's followed by AS (alias) or whitespace/comma/end
        cte_ref_table_pattern = rf'\b((?:{re.escape(schema_name)}\.)+){re.escape(cte_name)}(?=\s+(?:AS\s+\w+|\w+|,|\)|$))'
        sql_text = re.sub(cte_ref_table_pattern, cte_name, sql_text, flags=re.IGNORECASE)
    
    # Fix multiple schema prefixes on actual tables: schema.schema.schema.table -> schema.table
    # Pattern: schema.schema.table (where schema repeats)
    table_pattern = rf'\b({re.escape(schema_name)}\.)+({re.escape(schema_name)}\.)+(\w+)\b'
    sql_text = re.sub(table_pattern, rf'{schema_name}.\3', sql_text, flags=re.IGNORECASE)
    
    # STEP 2: Use sqlglot for more precise fixes on column references and edge cases
    try:
        ast = parse_one(sql_text, read=None)
        changed = False
        
        # Collect CTE names from the AST (after regex preprocessing)
        cte_names: Set[str] = set()
        for cte in ast.find_all(exp.CTE):
            alias = getattr(cte, "alias", None)
            if alias and hasattr(alias, "name"):
                cte_name = alias.name
                # Double-check: remove any remaining schema prefixes
                if "." in cte_name:
                    parts = cte_name.split(".")
                    if parts[0].upper() == schema_name.upper():
                        alias.name = parts[-1]
                        changed = True
                        cte_name = alias.name
                cte_names.add(cte_name.lower())
        
        # Fix table references that are CTEs (in FROM/JOIN clauses)
        for table in ast.find_all(exp.Table):
            identifier = getattr(table, "this", None)
            if not isinstance(identifier, exp.Identifier):
                continue
            
            table_name = identifier.name
            if not table_name:
                continue
            
            # Check if this is a CTE reference with schema prefix
            if "." in table_name:
                parts = table_name.split(".")
                if len(parts) > 1 and parts[-1].lower() in cte_names:
                    identifier.name = parts[-1]
                    if hasattr(table, "db"):
                        table.set("db", None)
                    changed = True
                    continue
            
            # Fix multiple schema prefixes on actual tables
            if "." in table_name:
                parts = table_name.split(".")
                if len(parts) > 2 and parts[0].upper() == schema_name.upper():
                    # Remove duplicate schema prefixes
                    cleaned_parts = [parts[0]]
                    i = 1
                    while i < len(parts) - 1:
                        if parts[i].upper() != schema_name.upper():
                            cleaned_parts.append(parts[i])
                        i += 1
                    cleaned_parts.append(parts[-1])
                    identifier.name = ".".join(cleaned_parts)
                    changed = True
                elif len(parts) == 2 and parts[0].upper() == schema_name.upper() and parts[1].lower() in cte_names:
                    # This is a CTE with schema prefix
                    identifier.name = parts[1]
                    if hasattr(table, "db"):
                        table.set("db", None)
                    changed = True
        
        # Fix column references with schema prefixes (e.g., e.AGENT_DEMO.run_at -> e.run_at)
        for column in ast.find_all(exp.Column):
            table = getattr(column, "table", None)
            if table:
                table_str = table if isinstance(table, str) else (getattr(table, "name", None) if hasattr(table, "name") else str(table))
                if table_str and isinstance(table_str, str):
                    if "." in table_str:
                        parts = table_str.split(".")
                        # Check if it's a CTE reference with schema prefix
                        if len(parts) > 1 and parts[-1].lower() in cte_names:
                            column.set("table", parts[-1])
                            changed = True
                        elif len(parts) > 1 and parts[-1].upper() == schema_name.upper():
                            # Remove schema part from column reference: e.AGENT_DEMO -> e
                            column.set("table", parts[0])
                            changed = True
        
        return ast.sql() if changed else sql_text
    except Exception as e:
        # If sqlglot parsing fails, return the regex-preprocessed SQL
        LOGGER.debug("sqlglot parsing failed in _fix_incorrect_schema_prefixes, using regex-only fix: %s", e)
        return sql_text


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
        desired_columns = state.get("desired_columns") or []
        desired_columns_raw = state.get("desired_columns_raw") or []
        query_analysis = state.get("query_analysis") or {}
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "question": state.get("question", ""),
                "sql": sql_text,
                "plan": plan_summary,
                "error": state.get("execution_error", ""),
                "desired_columns_section": _format_desired_columns_section(desired_columns, desired_columns_raw),
                "query_analysis_section": _format_query_analysis_section(query_analysis),
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
        # Use rich Graph Context (table cards, column cards, relationships, value anchors)
        graph_text = _build_rich_graph_context_for_repair(state)
        desired_columns = state.get("desired_columns") or []
        desired_columns_raw = state.get("desired_columns_raw") or []
        query_analysis = state.get("query_analysis") or {}
        result = _safe_invoke_json(
            chain,
            {
                "dialect": state.get("dialect", ""),
                "graph": graph_text,
                "sql": sql_text,
                "error": state.get("execution_error", ""),
                "repair_hints": critic.get("repair_hints", []),
                "plan": plan,
                "desired_columns_section": _format_desired_columns_section(desired_columns, desired_columns_raw),
                "query_analysis_section": _format_query_analysis_section(query_analysis),
            },
            llm,
        )
        LOGGER.info("Post-exec repair attempt %d/%d (MAX) result received", attempts + 1, MAX_INTENT_REPAIRS)
        patched_sql = result.get("patched_sql")
        raw_response = result.get("__raw")
        what_changed = result.get("what_changed")
        why_changed = result.get("why")
        if what_changed:
            LOGGER.info("Post-exec repair what_changed: %s", what_changed)
        if why_changed:
            LOGGER.info("Post-exec repair why: %s", why_changed)
        
        # Normalize patched_sql: handle arrays, null, wrong types, etc.
        if patched_sql is not None:
            # Handle array format: ["SELECT ..."]
            if isinstance(patched_sql, list):
                if len(patched_sql) > 0:
                    first_elem = patched_sql[0]
                    if isinstance(first_elem, str) and first_elem.strip():
                        patched_sql = first_elem
                    else:
                        LOGGER.warning("patched_sql array contains non-string elements, using robust extraction")
                        patched_sql = None
                else:
                    patched_sql = None
            # Handle wrong types: number, object, etc.
            elif not isinstance(patched_sql, str):
                LOGGER.warning("patched_sql has wrong type (%s), expected string. Using robust extraction.", type(patched_sql).__name__)
                patched_sql = None
            # Handle empty/whitespace strings
            elif not patched_sql.strip():
                patched_sql = None
        
        # If patched_sql is empty but we have raw response, try robust extraction
        if (not patched_sql or not patched_sql.strip()) and raw_response:
            # Normalize raw_response to string (handle bytes from any LLM model)
            if isinstance(raw_response, bytes):
                raw_response = raw_response.decode('utf-8', errors='ignore')
            elif not isinstance(raw_response, str):
                raw_response = str(raw_response)
            LOGGER.debug("Post-exec repair returned empty patched_sql, attempting robust extraction from raw response (length: %d)", len(raw_response) if raw_response else 0)
            extracted_sql = _extract_sql_from_malformed_json(raw_response, field_name="patched_sql")
            if extracted_sql:
                LOGGER.info("Extracted patched_sql from raw response using robust parser in post-exec repair: %s", extracted_sql[:200])
                patched_sql = extracted_sql
            else:
                # Log what we tried to extract from for debugging
                LOGGER.debug("Robust extraction failed. Raw response preview: %s", raw_response[:500] if raw_response and len(raw_response) > 500 else raw_response)
        
        if not patched_sql and raw_response:
            LOGGER.warning("Post-exec repair returned no patched_sql. Raw response (first 1000 chars): %s", raw_response[:1000] if len(raw_response) > 1000 else raw_response)
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

