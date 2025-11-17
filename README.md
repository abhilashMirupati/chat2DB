# SQLAI

Production-ready Graph-RAG agent that connects to your databases, generates safe SQL, and explains the results in a Streamlit UI.

---
## Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Breakdown](#component-breakdown)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [Prewarm & Runtime Workflow](#prewarm--runtime-workflow)
7. [Retrieval & Guardrails](#retrieval--guardrails)
8. [Observability & Production Checklist](#observability--production-checklist)
9. [Repository Layout](#repository-layout)
10. [Contributing & License](#contributing--license)

---
## Overview

SQLAI is designed for enterprise teams who need trustworthy, self-healing SQL generation:

- **Graph-RAG context** from live schema introspection, cached in SQLite, embedded in Chroma.
- **LangGraph workflow** with planner, pre-execution critic/repair, deterministic guardrails, post-execution critic/repair, and summariser.
- **Streamlit UI** with saved profiles, connection testers, saved query replay, and detailed plan/SQL previews.
- **Headless prewarm** script for production deployments so cold starts are fast.

---
## Architecture

```mermaid
flowchart TD
    A[Launch UI or prewarm script] --> B[Load configs & validate]
    B --> C[Introspect schema\n(sqlalchemy.inspect)]
    C --> D[Hydrate metadata\nLLM descriptions + samples]
    D --> E[Build GraphContext\nTable/Column/Relationship cards]
    E --> F[Persist graph cards\nSQLite cache]
    F --> G[Embed table cards only\nChroma namespace per model]
    G --> H[User question\nStreamlit UI]
    H --> I[SemanticRetriever\nTable-only search]
    I --> I1[Stage 1: Semantic search\nTable cards only]
    I1 --> I2[Stage 2: FK expansion\nAdd referenced tables]
    I2 --> I3[Stage 3: Get ALL columns\nfor selected tables]
    I3 --> I4[Stage 4: Get ALL relationships\nfor selected tables]
    I4 --> J[Planner LLM\nLangGraph plan node]
    J --> K[Intent critic/repair\npre-exec gatekeeper]
    K --> K1[SQLGlot Transpile\nConvert to target dialect]
    K1 --> L[Execute SQL\nvalidate_sql + pandas]
    L --> M[Post critic/repair\nruntime gatekeeper]
    M --> N[Summariser LLM\nanswer + chart + follow-ups]
    N --> O[Persist answer\nconversation history + saved queries]
```

**Graph Context Structure:**
- **TableCard**: Table name, schema, LLM-generated description, row estimate, trimmed column list
- **ColumnCard**: Column name, type, nullable/default, sample values (5), comments
- **RelationshipCard**: Foreign key relationships (FK edges between tables)

**Caches & storage**
- `.cache/table_metadata.db` – table descriptions + column samples (LLM-generated).
- `.cache/graph_cards.db` – rendered cards (table/column/relationship) with schema hashes.
- `.cache/vector_store/<provider>__<model>/` – Chroma index per embedding model (table cards only).
- `.cache/conversation_history.db` – successful interactions for replay.

---
## Component Breakdown

| Layer | Responsibilities | Key files |
|-------|------------------|-----------|
| Streamlit UI | Profiles, connection tests, question handling, result rendering, saved query replay | `src/sqlai/ui/app.py`, `profile_store.py` |
| AnalyticsService | DB engine, metadata hydration, GraphContext build, LangGraph orchestration, cache sync | `src/sqlai/services/analytics_service.py` |
| Graph & vector caches | Persist rendered cards, embed table cards via Chroma, support warm restarts | `graph_cache.py`, `vector_store.py` |
| LangGraph workflow | Planner, intent critic/repair, execution node, post critic/repair, summariser | `src/sqlai/agents/query_agent.py` |
| Guardrails | SQLGlot transpilation, deterministic SQL validation/repair (row caps, dialect rewrites, literal alignment) | `src/sqlai/guards/sql_guard.py`, `src/sqlai/utils/sql_transpiler.py` |
| Retrieval | Two-stage: table-only semantic search → FK expansion → get ALL columns/relationships | `src/sqlai/semantic/retriever.py` |
| Graph Context | TableCard, ColumnCard, RelationshipCard builders and formatters | `src/sqlai/graph/context.py` |

---
## Installation & Setup

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10 / 3.11 | `python --version` |
| Git | optional but useful |
| Oracle Instant Client | only for Oracle; update PATH/LD_LIBRARY_PATH |
| Database access | SQLAlchemy URL (Oracle, Postgres, MySQL, SQL Server, SQLite, …) |
| LLM credentials | Ollama (local) or API key (OpenAI / Anthropic / Hugging Face / Azure) |
| Embedding provider | Required for Graph-RAG (Hugging Face token or Ollama model) |

### Create virtual environment
```bash
git clone https://github.com/your-org/sqlai.git
cd sqlai
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate
pip install -e ".[openai,anthropic,ollama,huggingface]"
pip install chromadb
```

---
## Configuration

Create `.env` in the repo root:

```
SQLAI_DB_URL=oracle+oracledb://user:password@host:1521/?service_name=XEPDB1
SQLAI_DB_SCHEMA=AGENT_DEMO

SQLAI_LLM_PROVIDER=huggingface
SQLAI_LLM_MODEL=defog/llama-3-sqlcoder-8b:featherless-ai
SQLAI_LLM_BASE_URL=https://router.huggingface.co/v1
SQLAI_LLM_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

SQLAI_EMBED_PROVIDER=huggingface
SQLAI_EMBED_MODEL=google/embeddinggemma-300m
SQLAI_EMBED_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

SQLAI_VECTOR_PROVIDER=chroma
# Optional overrides:
# SQLAI_VECTOR_PATH=.cache/vector_store
# SQLAI_VECTOR_COLLECTION=graph_cards

SQLAI_LOG_LEVEL=INFO
SQLAI_BRAND_NAME=Silpa Analytics Lab
# SQLAI_BRAND_LOGO_PATH=C:\path\to\logo.png
```

Database, chat LLM, and embedding LLM settings are mandatory. `defog/*` models automatically route to the correct Hugging Face endpoint; others fall back to `https://router.huggingface.co/hf-inference`.

---
## Prewarm & Runtime Workflow

### Prewarm (recommended for prod)
```bash
python scripts/prewarm_metadata.py
# Windows PowerShell
.\scripts\prewarm_metadata.ps1n
```
Hydrates table descriptions, column samples, graph cards, and vector embeddings without launching Streamlit.

### Run the UI
```bash 
python run_app.py
```
Open the printed URL (default `http://localhost:8501`).

### Sidebar workflow
1. Load or create a profile.
2. Configure DB URL + schema; **Test database connection**.
3. Configure chat LLM; **Test LLM connection**.
4. Configure embeddings; **Test embedding connection**.
5. Click **Initialise Agent** (uses prewarm caches if available).
6. Ask questions; inspect plan JSON, SQL preview, Table preview, charts, and follow-ups.
7. Use **Saved queries** to replay cached SQL from `.cache/conversation_history.db`.

---
## Retrieval & Guardrails

### Graph Context Building

**Graph cards** are structured representations of your database schema:

- **`TableCard`**: 
  - Table name, schema, row estimate
  - LLM-generated description (business-friendly explanation)
  - Trimmed column list (first 12 columns for overview)
  
- **`ColumnCard`**: 
  - Column name, type, nullable/default
  - Sample values (5 representative values from actual data)
  - Column comments (if available)
  
- **`RelationshipCard`**: 
  - Foreign key relationships (FK edges)
  - Format: `schema.table[column] -> schema.referred_table[referred_column]`

Rendered cards are cached in `.cache/graph_cards.db` and **only table cards are embedded** in ChromaDB under `.cache/vector_store/<provider>__<model>/`.

### Two-Stage Semantic Retrieval

**Why table-only search?** 
- Prevents noise from irrelevant table columns
- Ensures columns belong to relevant tables
- More semantically correct: tables first, then their columns

**Retrieval Process:**

1. **Stage 1: Table Selection** (Semantic search on table cards only)
   - Heuristic ranking: keyword matching on table/column names
   - On-the-fly semantic similarity: embedding-based table ranking (fallback)
   - Vector store search: queries ChromaDB with `where={"card_type": "table"}`
   - Merge results: combines heuristic + semantic + vector scores
   - Select top-K tables (default: 6)

2. **Stage 2: FK Expansion** (1-hop expansion)
   - Automatically includes tables referenced by foreign keys
   - **Outgoing FKs** (priority): If selected table has FK → other table, add that table
   - **Incoming FKs** (secondary): If other table has FK → selected table, add that table
   - Safety limit: Max 5 additional tables to prevent context explosion
   - **Why?** Ensures complete join paths even if referenced tables weren't semantically matched

3. **Stage 3: Column Retrieval** (No semantic search)
   - Gets **ALL columns** for selected tables (including expanded)
   - No semantic filtering - ensures no important columns are missed
   - Includes foreign keys, join columns, and all other columns
   - Case-insensitive matching for table names
   - Limited to `max_columns` if exceeds limit (prioritizes by heuristic)

4. **Stage 4: Relationship Retrieval**
   - Gets **ALL relationships** for selected tables (including expanded)
   - Includes both directions (outgoing and incoming FKs)
   - Ensures complete join paths are available to LLM

**What gets passed to LLM:**

The `GraphContext` passed to the planner includes:
- **Selected tables**: Top-K semantically matched tables + FK-expanded tables
- **All columns**: Every column from selected tables (not just top-K)
- **All relationships**: Every FK relationship involving selected tables
- **Value anchors**: Sample values from columns for realistic filters
- **Retrieval details**: Metadata about which tables/columns were selected and why

**Example:**
```
Query: "What are the test sets with max failures?"

Stage 1: Semantic search → [test_sets, executions] (top 2)
Stage 2: FK expansion → executions → test_cases (+1) → [test_sets, executions, test_cases]
Stage 3: Get ALL columns → 13 columns from 3 tables
Stage 4: Get ALL relationships → 2 FK relationships

Passed to LLM:
- 3 tables (with descriptions)
- 13 columns (with types, samples, nullable)
- 2 relationships (complete join paths)
```

### Graph Context Formatting for LLM

The `GraphContext.prepare_prompt_inputs()` method formats the retrieved cards into structured text for the LLM prompt:

**Table Cards Format:**
```
schema.table_name | rows≈estimate
  comment: LLM-generated description
  columns: col1 (TYPE), col2 (TYPE, nullable), ...
```

**Column Cards Format:**
```
schema.table.column | type=TYPE | nullable=Tr ue/False | default=VALUE | values≈[sample1, sample2, ...]
```

**Relationship Cards Format:**
```
schema.table[column] -> schema.referred_table[referred_column]
```

**Prompt Structure:**
- `table_cards`: All selected table cards (one per line, separated by blank lines)
- `column_cards`: All selected column cards (one per line)
- `relationship_map`: All selected relationships (one per line)
- `column_facts`: Column metadata summary
- `value_anchors`: Sample values for realistic filters
- `k_tables`: Count of selected tables
- `k_columns`: Count of selected columns

This structured format ensures the LLM has complete schema information, sample values, and join paths to generate accurate SQL.

### Guarded execution
- **SQLGlot transpilation** (mandatory): Automatically converts SQL to the target database dialect if LLM generates wrong-dialect SQL. Supports Oracle, PostgreSQL, MySQL, SQL Server, SQLite, Snowflake, BigQuery, and more.
- `validate_sql` enforces row caps, read-only queries, literal alignment, and dialect normalization.
- `repair_sql` patches missing limits, metadata rewrites, and dialect quirks.
- **Intent critic** (pre-execution) ensures SQL answers the question before hitting the DB (max 5 iterations).
- **Post critic** inspects runtime errors, provides fixes, and retries (max 3 iterations).
- **Robust JSON parsing** for planner, intent critic/repair, and post critic/repair: malformed JSON (e.g., trailing commas) is auto-fixed so critique and repair always run.
- **Schema-aware SQL normalization (universal):**
  - Column-name repair: if a referenced column does not exist, map to the closest valid column from the joined tables (e.g., `result` → `status`) using string similarity against the GraphContext.
  - Alias remap: if a qualified column belongs to a different joined table, re-qualify automatically (e.g., `tc.status` → `e.status` when `status` exists only on `executions`).
  - Runs before validation/execution; database-agnostic, no query-specific rules.
- Summariser LLM blends plan rationale, SQL, preview tables, and execution stats into a business-friendly answer and chart.

---
## Observability & Production Checklist

### Logging & debugging
Set `SQLAI_LOG_LEVEL=INFO` or `DEBUG` to see:

**Retrieval logs (INFO level):**
- Table selection process (heuristic, semantic, vector store)
- FK expansion details (which tables added, why)
- Column retrieval (counts per table, total columns)
- Relationship retrieval (FK paths found)
- Final summary (tables/columns/relationships passed to LLM)

**Detailed logs (DEBUG level):**
- Retrieval scores for each table/column
- Planner JSON output
- Critic verdicts and repair hints
- Guardrail patches
- Executed SQL
- Metadata hydration progress

UI visibility:
- The full planner prompt and the final SQL (after all intent repairs) are shown in a Streamlit expander under the answer.

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Unknown tables referenced` | Schema cache stale; rerun prewarm or initialise after selecting schema. |
| `ORA-00904` / `ORA-00933` | Dialect mismatch; guardrail output shows patched SQL. |
| Hugging Face `401/404/410` | Token lacks access or wrong router; accept license, ensure `.env` base URL correct. |
| Metadata hydration slow | Run prewarm; caches reduce cold-start time dramatically. |
| Retrieval misses tables | Embedding provider not configured/tested; run sidebar "Test embedding connection". |
| Missing FK-referenced tables | FK expansion should auto-include them; check logs for expansion details. If limit reached (5 tables), increase `max_expansion` in retriever. |
| No columns found for selected tables | Check table name matching (case-insensitive); verify graph_cards.db has columns for those tables. |

### Production deployment checklist
1. **Secrets management** – inject DB URLs/API keys from Vault/Key Vault/Secrets Manager.
2. **Prewarm on deploy** – run `scripts/prewarm_metadata.py` in CI/CD or container entrypoint.
3. **Persistent cache** – mount `.cache/` on durable storage (EFS/Azure Files/host volume).
4. **Monitoring** – route logs to ELK, CloudWatch, or Log Analytics; add `/healthz` probes to Streamlit container.
5. **Rate limits** – verify Hugging Face/Ollama quotas match expected concurrency; scale inference services accordingly.
6. **Rollback plan** – pin repo to tags, snapshot caches, and keep previous images for quick fallback.

---
## Repository Layout

```
README.md                # this file
run_app.py               # Streamlit entry point
scripts/
    prewarm_metadata.py  # hydrate metadata/vector caches
src/sqlai/
    agents/              # LangGraph nodes/workflow
    database/            # connectors, schema introspection
    graph/               # GraphContext builders
    llm/                 # provider factory + prompts
    semantic/            # retriever + similarity providers
    services/            # AnalyticsService, caches, visualisation
    ui/                  # Streamlit app + profile store
    utils/               # logging helpers
    guards/              # SQL guardrails
```

---
## Contributing & License

1. Fork → branch → PR.
2. When filing issues include DB dialect/URL (sanitised), full question, DEBUG log snippets, and repro steps.
3. Tests should live under `tests/` and mock DB/LLM dependencies.

Licensed under **MIT**. Built so teams can "talk to their data" without wiring bespoke dashboards.

> Tip: keep `SQLAI_LOG_LEVEL=INFO` or `DEBUG` enabled during initial setup to watch the entire reasoning chain, including detailed retrieval logs showing table selection, FK expansion, and column/relationship retrieval.
