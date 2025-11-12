# SQLAI

Natural language analytics agent that connects to your databases, generates SQL, runs it safely, and presents the results in a Streamlit UI.

## 1. What this project does

Imagine asking “Which test set had the most failures last week?” and getting:

1. **A SQL query** that actually works for your database dialect.
2. **A data preview, chart, and plain language summary.**
3. **Safety checks** so we do not touch tables we do not know about, the SQL stays read only, and every query is capped.

That is the goal of SQLAI. You provide the connection details and the LLM/embedding provider, and the agent handles the rest:

- Introspects the schema (tables, columns, foreign keys, sample rows).
- Builds a graph-based context for the LLM (Graph-RAG).
- Generates a plan, guarded SQL, and summaries.
- If the database returns an error, a **gatekeeper loop** critiques the SQL, repairs it automatically, and tries again (up to three attempts).

## 2. Architecture at a glance

```
Streamlit UI
    ↓ sidebar collects DB + LLM settings
AnalyticsService (src/sqlai/services/analytics_service.py)
    ↓ creates SQLAlchemy engine, loads schema, graph context
LangGraph workflow (src/sqlai/agents/query_agent.py)
    plan → intent critic/repair loop → execute → post-exec critic/repair loop → summarise
        plan node        : LLM + Graph-RAG context builds plan JSON
        intent critic    : second LLM checks the SQL before execution (does it answer the question?)
        intent repair    : optional loop (max 5 tries) to rewrite SQL before it ever hits the database
        guard            : validate_sql / repair_sql add limits, sanity checks
        execute node     : runs sql via pandas.read_sql_query
        post critic node : gatekeeper LLM judges runtime failures, suggests fixes
        post repair node : LLM rewrites SQL using critic hints (max 3 tries)
        summarise node   : LLM explains results, charts, follow-ups
Supporting modules
    embeddings       : Hugging Face or Ollama similarity for retrieval
    database         : connectors, external table helpers, metadata filters
    ui/app.py        : Streamlit layout, status messages, result rendering
```

Detailed walkthrough: see [`docs/query_flow_example.md`](docs/query_flow_example.md) for a step-by-step example of a complex query flowing through each node.

## 3. Prerequisites

| Requirement                   | Notes                                                                 |
|------------------------------|-----------------------------------------------------------------------|
| Python 3.10 or 3.11          | Check with `python --version`.                                        |
| Git                          | Optional but recommended.                                             |
| Oracle Instant Client        | Only if you use Oracle. Place DLLs/so files on the machine running the agent. |
| Databases                    | Any SQLAlchemy URL works (Oracle, Postgres, MySQL, SQL Server, SQLite, etc.). |
| LLM provider credentials     | Ollama (local) or API key for OpenAI / Anthropic / Azure / Hugging Face. |
| Embedding provider           | Optional but recommended. Hugging Face token or Ollama model name.     |

### Windows specifics

1. Install Python from https://www.python.org/. During setup tick “Add Python to PATH”.
2. Install Git from https://git-scm.com/ if you want version control.
3. For Oracle: install the Instant Client (Basic or Basic Light) and set `PATH` to include its directory.

### macOS specifics

1. Install Homebrew (optional) then `brew install python git`.
2. For Oracle 21c Instant Client: download the mac package and follow Oracle instructions. Set `DYLD_LIBRARY_PATH`.

### Linux specifics

1. Use your package manager to install Python (`sudo apt-get install python3 python3-venv`).
2. For Oracle: install Instant Client RPM/zip and set `LD_LIBRARY_PATH`.

## 4. Installation (fresh laptop walk-through)

The project ships with all Python dependencies specified in `pyproject.toml`. Installing with `pip install -e .[extras]` ensures a reproducible environment across Windows/macOS/Linux. For a first-time setup on a clean machine, follow the steps below.

```bash
# Clone or download the repo
git clone https://github.com/your-org/sqlai.git
cd sqlai

# Create isolated environment
python -m venv .venv

# Activate the environment
# Windows PowerShell
.\.venv\Scripts\Activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies (core + optional extras)
pip install -e ".[openai,anthropic,ollama,huggingface]"
```

The editable install (`pip install -e`) means code changes are picked up immediately.

### Environment variables (optional but helpful)
Create a `.env` file in the project root:

```
SQLAI_DB_URL=oracle+oracledb://user:password@host:1521/?service_name=ORCLPDB1
SQLAI_DB_SCHEMA=AGENT_DEMO
SQLAI_LLM_PROVIDER=huggingface
SQLAI_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
SQLAI_LLM_BASE_URL=https://router.huggingface.co/v1
SQLAI_LLM_API_KEY=hf_xxxxx
SQLAI_EMBED_PROVIDER=huggingface
SQLAI_EMBED_MODEL=google/embeddinggemma-300m
SQLAI_EMBED_API_KEY=hf_xxxxx
SQLAI_LOG_LEVEL=DEBUG
SQLAI_BRAND_NAME=Silpa Analytics Lab
# Optional: point to a PNG/SVG/JPEG logo that appears in the sidebar
SQLAI_BRAND_LOGO_PATH=C:\path\to\logo.png
```

All entries are optional. The Streamlit UI lets you override them at runtime. `SQLAI_BRAND_NAME` / `SQLAI_BRAND_LOGO_PATH` show your branding in the sidebar (defaults to `assets/logo.*` if present).

### Oracle connection strings

SQLAlchemy requires the `oracle+oracledb://user:password@host:port/?service_name=...` format. If you have a JDBC string such as

```
jdbc:oracle:thin:@(DESCRIPTION=(ADDRESS=(PROTOCOL=tcp)(HOST=hostname)(PORT=1521))(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME=erdtqcp)))
```

convert it by extracting host/port/service and supplying credentials:

```
oracle+oracledb://user:password@hostname:1521/?service_name=erdtqcp
```

## 5. Running the application

```bash
# From the repo root with the venv activated
python run_app.py
```

Streamlit prints a local URL (default `http://localhost:8501`). Open it in your browser.

> **Tip:** To avoid the UI doing this work on the first launch, you can pre-warm the table metadata cache:
>
> ```bash
> # macOS / Linux
> python scripts/prewarm_metadata.py
> ```
>
> ```powershell
> # Windows PowerShell
> .\scripts\prewarm_metadata.ps1
> ```
>
> The cache lives at `.cache/table_metadata.db`.

### Sidebar workflow

1. **Load/Save profiles** – store DB + LLM + embedding settings to reuse later.
2. **Database section**
   - Pick the database type (Oracle, Postgres…). The connection string template updates accordingly.
   - Optional Oracle thick mode options (Instant Client path, TNS config).
   - Click “Test database connection” to verify credentials. If Oracle external tables error (ORA-29913 / ORA-29400), resolve the underlying file paths first.
3. **LLM section**
   - Choose provider & model. For Hugging Face, either supply API key (router endpoint) or base URL for a custom deployment.
   - “Test LLM connection” sends a ping request; any authentication issues surface here.
4. **Embeddings section**
   - Optional but recommended for better table/column retrieval.
   - For Hugging Face, the test button calls `sentence_similarity` exactly like their official snippet. If you use router models (e.g., `google/embeddinggemma-300m`), no base URL is needed.
5. **Initialise Agent**
   - Builds the engine, introspects schema (tables, columns, FKs), sets up the LangGraph workflow, and opens a session.
6. Ask questions. Results appear in the main pane together with plan JSON, SQL statements, previews, charts, and follow-ups.

### Saved queries & replay

- Every successful question is saved per schema in `.cache/conversation_history.db` with its SQL, plan JSON, and summary.
- The main pane now has a **Saved queries** expander. Use **Run** to replay the stored SQL instantly (skips the LLM) or **Load** to drop the original question back into the text box for refinement.
- Cached runs still honour the guardrails (row limits, schema filtering) because the SQL is re-executed against the current database.
- Delete the cache file if you need a clean slate.

## 6. How a question is answered (end-to-end)

1. **Semantic retrieval** – `SemanticRetriever.select_cards()` combines heuristic matches with embedding similarity to pick the most relevant table/column “cards”. The details are logged (`Retrieval details` in DEBUG log).
2. **Planner LLM** – `plan` node invokes the Graph-RAG prompt, producing JSON with plan steps, SQL, tests, summary, chart hints, etc.
3. **Guardrails** – `validate_sql` ensures all tables are known, row caps are enforced, dialect mismatches are normalized (e.g. convert Oracle `FETCH FIRST` to Postgres `LIMIT`), and metadata views are handled safely.
4. **Execution** – Pandas executes the SQL. Dataframes are stored in `ExecutionResult` objects.
5. **Gatekeeper loop**
   - If execution fails, the **critic** prompt inspects the SQL, plan and error, providing reasons and repair hints.
   - The **repair** prompt rewrites the SQL using those hints.
   - The loop tries up to **three repairs**. After each repair the guard re-checks the SQL before execution.
   - If all attempts fail, the final error message shows the last SQL and critic reasons.
6. **Summariser** – takes the question, plan rationale, executed SQL, preview tables, and returns a readable answer plus chart instructions. If an error remained, it reports the failure instead of a summary.

## 7. Logging & debugging

Set `SQLAI_LOG_LEVEL=DEBUG` to see:
- Retrieval scores (`semantic_tables`, `semantic_columns`).
- Planned SQL and rewrite notes (e.g. “Adjusted metadata query to use ALL_TABLES…”).
- Critic verdicts and repair attempts.
- Exact SQL sent to the database and any errors (tracebacks).

Common Oracle issues:
- **ORA-00904** – usually alias or column typo. The critic will call this out. The repair loop should fix it, but if not, capture the failing SQL and adjust your schema prompts or add domain templates.
- **ORA-29913 / ORA-29400 / KUP-04044** – external table file missing/inaccessible. Fix the external table configuration; no amount of SQL rewriting can solve it.
- **ORA-00933** – invalid row-limit clause. The guard now removes stray `ROWNUM` usage and normalises the clause.

## 8. Project structure

```
README.md              ← you are here
run_app.py             ← Streamlit entry point
pyproject.toml         ← dependencies / extras
src/sqlai/
    agents/            ← LangGraph nodes, guard, repair loop
    database/          ← SQLAlchemy connectors, schema introspection
    graph/             ← Graph context builders (table/column cards)
    llm/               ← Providers + prompt templates
    semantic/          ← Embedding-based retriever
    services/          ← AnalyticsService orchestrator, visualisation
    ui/                ← Streamlit app and profile store
    utils/             ← Logging helpers
```

## 9. Customisation points

- **Domain fallbacks**: `services/domain_fallbacks.py` contains templates for frequently used metadata queries (count tables, list columns, failure summaries). Extend this with your domain-specific SQL patterns.
- **Embeddings**: add new providers by implementing `SimilarityProvider` in `semantic/retriever.py`.
- **LLM providers**: plug in new LangChain chat models via `llm/providers.py`.
- **Visualisations**: extend `services/visualization.py` to map chart specs to Plotly (or other libraries).
- **Branding**: set `SQLAI_BRAND_NAME` / `SQLAI_BRAND_LOGO_PATH` (PNG/JPG/SVG) to add your own name and logo to the Streamlit sidebar. If no path is provided, the app automatically looks for `assets/logo.(png|jpg|jpeg|svg|webp)`.

## 10. Testing ideas

We recommend adding automated tests under `tests/`:
- Mock database metadata and ensure the planner emits the expected SQL.
- Verify the critic/repair loop corrects common errors (alias mismatch, limit clause).
- Check metadata rewrites (`user_tables` → `all_tables`) for Oracle and no-op for other dialects.

## 11. Troubleshooting checklist

| Symptom                                  | Likely cause / fix                                                   |
|-----------------------------------------|-----------------------------------------------------------------------|
| “Unknown tables referenced”              | Table not in schema cache; reinitialise agent after selecting schema. |
| ORA-00904 “invalid identifier”           | Alias/column mismatch; critic should highlight. Repair loop will fix. |
| ORA-00933 “SQL command not properly ended” | Row-limit clause in wrong dialect; guard now normalises automatically. |
| ORA-29913 / ORA-29400 / KUP-04044        | Oracle external table cannot access file. Fix file path/permissions.  |
| Hugging Face 401                         | Token missing scope or model access. Accept model licence, regenerate token. |
| Hugging Face 410                         | Model not served on public inference API. Deploy your own endpoint or choose another model. |
| LLM returns no SQL                       | Plan JSON will include `sql_generation_note`. Provide clearer question or ensure schema/metadata exists. |
| Semantic retrieval missing tables        | Check embedding provider configuration, run “Test embedding connection” to confirm success. |

## 12. Contributing & support

1. Fork the repo, create a topic branch, and open a pull request.
2. File issues with:
   - Database dialect and URL
   - Exact question asked
   - DEBUG log snippet (plan, critic verdict, repair attempts)
   - Steps to reproduce

## 13. Credits & license

MIT License. Built for teams who want to “talk to their data” without wiring up bespoke dashboards for every question.

---
*Tip: enable `SQLAI_LOG_LEVEL=DEBUG` during initial setup to see exactly how the agent reasons about your schema and queries.*

