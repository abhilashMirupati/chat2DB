# SQLAI End-to-End Query Walkthrough

This guide explains what happens when SQLAI answers a realistic question end-to-end. It covers both **cold start** (first time a schema is seen) and **warm start** (metadata already cached) so you can trace the flow and the I/O of every component.

---

## Sample schema

| Table       | Relevant columns                                                                     |
|-------------|---------------------------------------------------------------------------------------|
| `test_sets` | `id`, `name`, `release`, `created_at`                                                |
| `test_cases`| `id`, `test_set_id` (FK → `test_sets.id`), `name`, `severity`                         |
| `executions`| `id`, `test_case_id` (FK → `test_cases.id`), `status`, `failure_reason`, `run_at`     |

## Business question

> “For the **Checkout smoke** test set, list each test case’s **top failure reason** over the last seven days and how many times it happened.”

This requires joins, date filters, grouping, and ordering—perfect for demonstrating the planner, guards, embeddings, and caching layers.

---

## High-level flow (cold start vs warm start)

```mermaid
flowchart TD
    subgraph Cold start
        UICold[Streamlit UI]
        UICold --> ASCold[AnalyticsService]
        ASCold --> MetaIntrospect[Schema introspector]
        MetaIntrospect --> CacheStore[(SQLite metadata cache)]
        ASCold --> RetrieverCold[SemanticRetriever (embeddings)]
        RetrieverCold --> PlannerCold[LangGraph plan node]
        PlannerCold --> CriticCold[Intent critic / repair]
        CriticCold --> GuardCold[SQL guard]
        GuardCold --> ExecuteCold[Execute node]
        ExecuteCold --> PostCold[Post-critique / repair]
        PostCold --> SummaryCold[Summarise node]
        SummaryCold --> UIRenderCold[Render in UI]
    end

    subgraph Warm start
        UIWarm[Streamlit UI]
        UIWarm --> ASWarm[AnalyticsService]
        ASWarm --> CacheRead[(SQLite cache hit)]
        ASWarm --> RetrieverWarm[SemanticRetriever (embeddings)]
        RetrieverWarm --> PlannerWarm[LangGraph plan node]
        PlannerWarm --> CriticWarm[Intent critic / repair]
        CriticWarm --> GuardWarm[SQL guard]
        GuardWarm --> ExecuteWarm[Execute node]
        ExecuteWarm --> PostWarm[Post-critique / repair]
        PostWarm --> SummaryWarm[Summarise node]
        SummaryWarm --> UIRenderWarm[Render in UI]
    end
```

- **Cold start** performs schema introspection and LLM metadata generation, storing results in `.cache/table_metadata.db` (SQLite) and local assets (value anchors, table descriptions, column samples).
- **Warm start** loads existing metadata from the cache instantly; only missing pieces trigger new LLM calls.

---

## Detailed step-by-step (Inputs → Outputs)

### 1. Streamlit UI
- **Input**: DB/LLM/embedding credentials, question text, selected schema, row cap.
- **Output**: `AnalyticsService.ask(question)` call with the current configuration and session state; optionally stores the branding/logo for display.

### 2. AnalyticsService initialisation
- Loads DB engine (`SQLAlchemy`), ensures connectivity.
- Loads LLM/embedding providers (via `sqlai.llm.providers` and `sqlai.semantic.retriever`).
- **Cold start**: `SchemaIntrospector.introspect_database` queries system catalogs for tables/columns/FKs; stores samples/descriptions in SQLite cache via `MetadataCache`.
- **Warm start**: retrieves cached descriptions and column samples directly from the SQLite cache.
- **Output**: `GraphContext` representing tables, columns, relationships, value anchors, and embedding-vectors-ready cards.

### 3. Semantic retrieval (embeddings + heuristics)
- **Input**: Question text, graph context.
- `SemanticRetriever` ranks cards using:
  - Keyword heuristics (direct column/table name matches).
  - Embedding similarity (Hugging Face or Ollama). Results are cached in-memory for the session.
- **Output**: Top-N table cards, column cards, relationship cards, plus a JSON blob of retrieval scores.

### 4. Prompt preparation
- `prompt_inputs = GraphContext.prepare_prompt_inputs(...)`
  - Includes: dialect guide, table/column cards, FK map, value anchors, sensitive columns, schema name, row cap.
  - Adds `analysis_hints` (e.g., “Use COUNT/GROUP BY”) derived from the question.
  - Captures metadata snapshot for summariser.
- **Output**: Structured dictionary for LangGraph plan node.

### 5. Plan node (`_plan_sql`)
- **Input**: Prompt inputs + question.
- LLM (planner) produces JSON adhering to the contract:
  - `plan.steps`, `plan.notes`
  - `sql` (string or array)
  - `rationale_summary`
  - `tests` (self-checks)
  - Optional `chart`
- **Output**: Planner result inserted into LangGraph state.

### 6. Intent critic (`_intent_critic`) and repair (`_intent_repair`)
- **Input**: Planned SQL and steps.
- Critic LLM verifies intent (correct tables, filters). If verdict is “reject”, `_intent_repair` iteratively patches the SQL up to 5 times.
- **Output**: Approved plan or annotated errors causing early exit.

### 7. Guard & validation (`validate_sql` + `repair_sql`)
- **Input**: Approved SQL, graph context, row cap, dialect, sensitive columns.
- Steps:
  1. Ensure tables exist and belong to the allowed schema.
  2. Verify no `SELECT *` across multiple tables (unless explicitly allowed).
  3. Enforce row caps (`FETCH FIRST`, `LIMIT`, `TOP` depending on dialect).
  4. Match literals to value anchors (`FAIL` vs `failed`).
  5. Normalize `LIKE` to case-insensitive when user typed lower-case strings.
- **Output**: Patched SQL passed to execution; or guard errors that trigger repair loop.

### 8. Execution node (`_execute_sql`)
- **Input**: Guard-approved SQL statements.
- Uses `pandas.read_sql_query` against the SQLAlchemy engine.
- Profiles each dataframe (`row_count`, top values per column) via `_profile_dataframe` for summariser context.
- **Output**: `ExecutionResult` objects containing SQL, dataframe, preview markdown, row counts, and stats.

### 9. Post-execution critic / repair
- If the database raises errors (syntax, permissions), the critic/repair loop attempts up to 3 LLM-guided fixes, each revalidated by guards.

### 10. Summariser (`_summarise`)
- **Input**: Question, plan rationale, executed SQL, preview markdown, structured column stats.
- Summariser LLM returns natural-language insight, optional chart spec, and follow-up suggestions.
- If the LLM refuses (e.g., placeholder warning), fallback summary uses profiled stats.
- **Output**: Summary text, chart spec, follow-up array.

### 11. UI rendering & caching
- Displays answer, plan JSON, chart, data table (first 20 rows), and summary stats.
- Stores successful Q&A in `conversation_cache` (SQLite) for quick replay (saved queries section).

---

## Example result (first 10 rows)

| test_case_name                  | top_failure_reason                 | failure_count |
|---------------------------------|------------------------------------|---------------|
| User can add item to cart       | Timeout waiting for cart service   | 7             |
| Cart persists across refresh    | Session cookie expired             | 5             |
| Apply coupon at checkout        | Pricing service unavailable         | 4             |
| Guest checkout flow             | Payment gateway declined            | 3             |
| Saved addresses load            | Address service timeout             | 3             |
| Checkout totals update          | Tax calculation mismatch            | 2             |
| Promo banner renders            | CDN asset missing                   | 2             |
| Order confirmation email        | Email service throttled             | 2             |
| Loyalty points applied          | Points balance out-of-sync          | 1             |
| PayPal handoff                  | Third-party redirect failure        | 1             |

The summariser uses the stats to note the leading failure reasons and highlight that `User can add item to cart` is particularly flaky.

---

## Key artifacts written during the flow

| Artifact                          | Producer                                    | Warm-start benefit                                         |
|----------------------------------|---------------------------------------------|------------------------------------------------------------|
| `.cache/table_metadata.db`       | `MetadataCache`                             | Cached table descriptions, column samples, schema hashes   |
| `.cache/conversation_history.db` | `ConversationCache`                         | Saved queries for one-click replay                         |
| Embedding vectors (in-memory)    | `SemanticRetriever`                         | Reused across subsequent questions in the same session     |
| `assets/logo.*` (optional)       | Branding hook (sidebar)                     | Displayed automatically if present                         |

---

## Troubleshooting tips

1. **Planner returns no SQL** → Check schema name, ensure metadata cache contains the tables, and verify embeddings are configured.
2. **Guard keeps rejecting** → Look at log lines from `validate_sql` to see unknown tables or literal mismatches.
3. **Row cap warnings** → The summary notes “row limit reached.” Refine the question or plan a future enhancement for pagination.
4. **Saved query replay** → Uses cached SQL; still runs through guard/execute to ensure freshness.

---

Need a visual aide in the README? Refer to this document alongside the “Architecture at a glance” section for a deeper dive.
