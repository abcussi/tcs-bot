# AGENTS.md — shared context for every AI model (Claude, Codex, Gemini, Copilot, Cursor)

This is the **single source of truth** for any LLM or human working in this repo.
`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md` and `.cursor/rules` all point here.
Keep it short and correct; update it in the same commit as any architectural change.

## 1. Product
**Vendor Risk AI** — generates AI-based vendor (third-party) risk assessment reports for
capital-markets firms. It ingests vendor financial reports, news and regulatory filings
(text/PDF), indexes them as dense embeddings, retrieves relevant chunks with numpy cosine
similarity, and asks an LLM (LCEL pipeline) for a **structured JSON risk report**: overall
risk score, red flags, financial-health summary and compliance recommendations. The generator
is exposed as a FastAPI backend **and** as an MCP tool for enterprise risk dashboards.
Market context (quotes) is scraped from Google Finance by a separate cron crawler.

## 2. Workflow: Spec-Driven Development (SDD)
1. **Spec first.** Every feature lives in `specs/NNN-name/` with `spec.md` (what/why,
   acceptance criteria), `plan.md` (how, contracts) and `tasks.md` (checklist).
2. Code only what a spec asks for. If you need to deviate, update the spec in the same change.
3. Tests map to acceptance criteria IDs (e.g. `AC-001.3`) in test docstrings.
4. `specs/000-constitution.md` holds non-negotiable rules. Read it before coding.

## 3. Layout
```
src/vendor_risk_ai/
  config.py            Settings (pydantic-settings, reads .env) -> get_settings()
  errors.py            AppError hierarchy (framework-neutral; re-exported by api/errors.py)
  main.py              CLI orchestrator: `uv run vendor-risk-ai all` boots everything
  api/                 FastAPI app factory, routes, custom errors + handlers, deps
  schemas/             Pydantic I/O models (api.py, report.py, domain.py)
  db/                  MySQL connection pool, transaction(), schema.sql, migrate, seed, repositories
  cache/               Two-tier cache: L1 in-process TTL/LRU + L2 Redis (graceful fallback)
  search/              Elasticsearch client + index management (quotes, chunks)
  ingestion/           PDF/text extraction (pypdf, OCR fallback), normalization, chunking
  rag/                 Embedders (OpenAI text-embedding-3-small | offline hash) + numpy retriever
  llm/                 LCEL chain: ChatPromptTemplate | ChatOpenAI(gpt-4.1-mini, T=0).with_structured_output
  services/            ReportService — orchestrates retrieval + LLM + persistence + cache
  mcp_server/          FastMCP server exposing `generate_vendor_risk_report`
crawler/               Separate uv workspace member `gfinance-crawler` (Google Finance -> ES), cron
data/seed/             Synthetic, anonymized vendor profiles + documents (fallback seed is built-in)
specs/                 SDD specs (constitution + features)
tests/{unit,integration,e2e}
docker-compose.yml     mysql, redis, elasticsearch, api, crawler
```

## 4. Commands
```bash
uv sync                                   # install (workspace incl. crawler)
uv run vendor-risk-ai all                 # compose up infra -> migrate -> seed -> index -> crawl -> API
uv run vendor-risk-ai up|down|status|migrate|seed|index|crawl|api|mcp
uv run vendor-risk-ai all --no-infra      # same pipeline, infra already running
uv run pytest tests/unit crawler/tests -m "not integration"   # fast, no infra
uv run pytest -m integration              # needs `uv run vendor-risk-ai up`
uv run pytest -m e2e                      # full stack
uv run ruff check . && uv run ruff format --check .
docker compose --profile app up -d --build   # everything in containers (api+crawler are in the `app` profile; needs .env)
```

## 5. Engineering rules (summary — full list in specs/000-constitution.md)
- Python 3.13, `uv` only (never pip). Type hints everywhere; ruff clean.
- Config only via `get_settings()`; secrets only in `.env` (never committed).
- MySQL access only through `db.mysql.transaction()` / pool — no ad-hoc connections.
- Every external dependency (OpenAI, Redis, ES) has an offline/fallback path so unit tests run with
  no network: `LLM_PROVIDER=fake`, in-memory cache, fixture HTML for the crawler.
- LLM output is **always** a validated Pydantic `RiskReport`; never return free text.
- API errors are `AppError` subclasses mapped by handlers to `{"error": {"code","message","details"}}`.
- Only synthetic/anonymized vendor data in the repo.

## 6. Agent roster (`.claude/agents/`)
| Agent | Owns |
|---|---|
| `architect-reviewer` | Reviews architecture, contracts between modules, integration of features; edits specs, not code |
| `developer` | Implements the monolithic app per specs (src/vendor_risk_ai) |
| `qa-engineer` | Unit, integration and e2e tests mapped to acceptance criteria |
| `crawler-engineer` | `crawler/` package, scraping, ES sink, scheduling |

Skills (`.claude/skills/`): `spec-driven` (create/advance a spec), `run-stack` (boot & verify the stack).
