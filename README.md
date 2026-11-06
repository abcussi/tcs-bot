# TCS Bot — Vendor Risk AI

[![CI](https://github.com/abcussi/tcs-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/abcussi/tcs-bot/actions/workflows/ci.yml)

AI-based **vendor risk assessment report generator for capital markets**.
Ingests vendor financial reports, news and regulatory filings (text/PDF), retrieves the relevant
chunks with `text-embedding-3-small` + numpy cosine similarity, and runs an LCEL pipeline
(`ChatPromptTemplate | ChatOpenAI(gpt-4.1-mini, temperature=0).with_structured_output(RiskReport)`)
to produce structured JSON reports: overall risk score, red flags, financial-health summary and
compliance recommendations. Exposed as a **FastAPI** backend and an **MCP tool**. Market context
comes from a **Google Finance crawler** (cron) that indexes quotes into **Elasticsearch**.

> All vendor data in this repo is synthetic. Tickers on seed vendors are listed *market proxies*.

## Architecture
```
            ┌────────────── crawler/ (cron, supercronic/APScheduler) ──────────────┐
            │ Google Finance ──► parse Quote ──► Elasticsearch [market-quotes]     │
            └──────────────────────────────────────────────────────────────────────┘
 client / dashboard                                   ▲ latest_quote(ticker)
   │  HTTP (FastAPI)  or  MCP (FastMCP tool)          │
   ▼                                                  │
 api/ ──► services/ReportService ──► rag/ (embed + numpy cosine top-k)
                │          │            llm/ LCEL chain ──► RiskReport (Pydantic)
                │          └──► cache/ L1 in-process TTL/LRU ─► L2 Redis
                └──► db/ MySQL (pool + transaction())  [vendors, documents, chunks, reports, audit]
                     search/ Elasticsearch [vendor-chunks dense_vector]
```

## Quick start
```bash
cp .env.example .env              # set OPENAI_API_KEY (or LLM_PROVIDER=fake to run offline)
uv sync
uv run tcs-bot all         # docker compose up mysql/redis/es -> migrate -> seed -> index -> crawl -> API
```
Open http://localhost:8000/docs. Generate a report:
```bash
curl -s -XPOST localhost:8000/reports -H 'content-type: application/json' -d '{"vendor_id": 1}'
```

Everything in containers (API + crawler + infra):
```bash
docker compose --profile app up -d --build
```

### Steps
| Command | What it does |
|---|---|
| `tcs-bot up` / `down` / `status` | Start/stop MySQL, Redis, Elasticsearch via Docker Compose and wait for health |
| `tcs-bot migrate` | Create all tables (tracked in `schema_migrations`; compensating rollback on failure) |
| `tcs-bot seed` | Load `data/seed/vendors.json`; falls back to built-in synthetic seed if missing |
| `tcs-bot index` | Ensure ES indices; chunk + embed + index every un-embedded document |
| `tcs-bot crawl` | One crawler run (non-fatal) |
| `tcs-bot api` / `mcp` | Serve FastAPI / MCP (stdio) |

### MCP
`.mcp.json` registers the server for Claude Code and other MCP clients:
`uv run tcs-bot-mcp` (stdio) or `uv run tcs-bot-mcp --transport streamable-http --port 8001`.
Tools: `generate_vendor_risk_report`, `list_vendors`.

## TCS risk medallion pipeline (spec 005)
Crawls news and communities about Tata Consultancy Services (TCS) and turns them into a risk view
across ethics, compliance, legal, privacy/cyber, labour/HR, financial, operational, reputational,
ESG and geopolitical/immigration. Limits: crawl depth <= 2, <= 500 documents per run.

| Layer | Store | Code | Content |
|---|---|---|---|
| Bronze | MySQL `bronze_documents`, `crawl_runs` | `crawler/src/gfinance_crawler/news/`, `medallion/bronze.py` | Raw documents as fetched, with crawl lineage (query, parent URL, depth, run id) |
| Silver | Elasticsearch `silver-tcs-risk` | `medallion/silver/` | Clean text, entities, sentiment, TCS relevance, risk signals by category, risk score 0-100 |
| Gold | Elasticsearch kNN `gold-tcs-risk-{docs,chunks}--<model>` + `data/gold/` | `medallion/gold/` | Embeddings, cosine k-NN graph, Louvain/KMeans clusters, interactive graph |

```bash
docker compose up -d mysql redis elasticsearch
uv run tcs-bot migrate && uv run tcs-bot seed && uv run tcs-bot index
uv run tcs-bot medallion run --max-docs 50 --max-depth 2   # bronze -> silver -> gold
uv run tcs-bot medallion bronze|silver|gold                # one layer at a time
uv run tcs-bot medallion schedule --cron "0 */6 * * *"     # recurring (Compose service `medallion`, profile `app`)
```
Offline input: drop `.html/.txt/.json/.pdf` files into `data/bronze/inbox/`.

**View the graph:** open `data/gold/tcs_risk_graph.html` in a browser (self-contained, no network).
The same data is served at `GET /tcs/risks/{search?q=&k=,summary,graph}` and by the MCP tool
`search_tcs_risks`.

**Embeddings:** OpenAI `text-embedding-3-small` when `OPENAI_API_KEY` is set and
`LLM_PROVIDER=openai`; otherwise an offline TF-IDF+SVD ("LSA") embedder. Index names include the
model, so vector spaces never mix. Gold vectors are Elasticsearch `dense_vector` fields (cosine, up
to 4096 dims, so LSA 256 and OpenAI 1536 both fit) searched with the `knn` API; the index prefix is
`GOLD_INDEX_PREFIX`. Tune the graph with `GOLD_GRAPH_K` and `GOLD_GRAPH_MIN_COSINE`,
and the relevance filter with `MEDALLION_RELEVANCE_THRESHOLD`.

**Sources and politeness:** Bing News RSS, GDELT, Hacker News and the local inbox work. robots.txt is
respected and never bypassed: Google News (`/rss/search`) and Reddit disallow generic crawlers, so
those sources are reported as `blocked` on every run. HTTP 429 is not retried and also counts as
blocked. Some publishers answer 403 to the bot user agent; their Bing RSS snippet is kept instead.

## Development (spec-driven)
Specs live in `specs/` (constitution + one folder per feature with `spec.md`, `plan.md`, `tasks.md`).
Shared AI context for all models: [`AGENTS.md`](AGENTS.md). Claude subagents: `.claude/agents/`
(architect-reviewer, developer, qa-engineer, crawler-engineer). Skills: `.claude/skills/`.

```bash
uv run pytest tests/unit crawler/tests     # offline
uv run pytest -m integration               # needs `tcs-bot up`
uv run pytest -m e2e
uv run ruff check .
OPENAI_API_KEY=sk-... uv run pytest -m openai   # real-model smoke test
```

### CI (`.github/workflows/ci.yml`)
Every push/PR: ruff + unit tests with coverage -> integration + e2e against MySQL/Redis/Elasticsearch
service containers -> Docker image builds. Add the repository secret `OPENAI_API_KEY`
(Settings -> Secrets and variables -> Actions) to also run the real gpt-4.1-mini smoke test.

## Requirements
- [uv](https://docs.astral.sh/uv/) and Python 3.13
- Docker Engine + Compose v2 (Docker Desktop on macOS/Windows)
- Optional: `tesseract-ocr` + `poppler` for OCR of scanned PDFs (included in the Docker image)
