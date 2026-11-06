# TCS Bot — Vendor Risk AI

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

## Development (spec-driven)
Specs live in `specs/` (constitution + one folder per feature with `spec.md`, `plan.md`, `tasks.md`).
Shared AI context for all models: [`AGENTS.md`](AGENTS.md). Claude subagents: `.claude/agents/`
(architect-reviewer, developer, qa-engineer, crawler-engineer). Skills: `.claude/skills/`.

```bash
uv run pytest tests/unit crawler/tests     # offline
uv run pytest -m integration               # needs `tcs-bot up`
uv run pytest -m e2e
uv run ruff check .
```

## Requirements
- [uv](https://docs.astral.sh/uv/) and Python 3.13
- Docker Engine + Compose v2 (Docker Desktop on macOS/Windows)
- Optional: `tesseract-ocr` + `poppler` for OCR of scanned PDFs (included in the Docker image)
