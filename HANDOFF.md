# HANDOFF — tcs-bot (paste the prompt below into Claude Code on your PC)

Repo: https://github.com/abcussi/tcs-bot  (branch `main`)  ·  Status date: 2026-11-06

---

## PROMPT (copy everything below this line)

You are taking over an in-progress Python project, **tcs-bot** (`https://github.com/abcussi/tcs-bot`, branch `main`). Work on the user's own computer, in a normal folder they can see and share — for example `~/Documents/repo/tcs-bot` — never in a temp directory. The previous agent worked in a cloud sandbox with no internet, so some things were never run against the real internet. Do not fabricate results. Report what you actually ran.

### How to operate (you are Claude Opus running Claude Code locally in the cloned repo)
- Start `claude` **inside the cloned repo folder** so it loads `CLAUDE.md` → `AGENTS.md`, the custom subagents in `.claude/agents/` (`developer`, `qa-engineer`, `architect-reviewer`, `crawler-engineer`) and the skills in `.claude/skills/` (`spec-driven`, `run-stack`). Use them.
- Plan first (short), then **run independent work in parallel with subagents** (e.g. real-crawl parser fixes, silver rule tuning, gold graph viz, CI/Docker). Give each subagent exclusive file ownership; you do all git commits.
- You have a real machine: use it for what the sandbox could not — real internet, Docker, a browser for screenshots. Verify with real runs before claiming anything is done.

### 0. Set up (do this first)
```bash
git clone https://github.com/abcussi/tcs-bot.git ~/Documents/repo/tcs-bot
cd ~/Documents/repo/tcs-bot
cp .env.example .env        # .env is git-ignored; set OPENAI_API_KEY if available, else keep LLM_PROVIDER=fake
uv sync                     # Python 3.13, uv only (never pip)
docker compose up -d mysql redis elasticsearch qdrant
uv run tcs-bot migrate && uv run tcs-bot seed && uv run tcs-bot index
```
Read first, in this order: `AGENTS.md` (shared rules), `specs/000-constitution.md`, `specs/005-tcs-risk-medallion/{spec,plan,tasks}.md`. The project uses **spec-driven development**: update the spec/plan in the same commit as any contract change, and keep commits small (one concern per commit).

### 1. What the project is
- **App A — Vendor Risk AI (finished, verified):** FastAPI backend + LCEL chain (`gpt-4.1-mini`, T=0, structured `RiskReport`) + numpy cosine retrieval (`text-embedding-3-small`) + MCP tool. MySQL pool/transactions, Redis+in-process cache, Elasticsearch, Google Finance crawler (`crawler/`, cron). Run: `uv run tcs-bot all`.
- **App B — TCS risk medallion pipeline (spec 005, IN PROGRESS):** crawl news + communities about **Tata Consultancy Services (TCS)** to find risks (ethics, compliance, legal, privacy/cyber, labour/HR, financial, operational, reputational, ESG, geopolitical/immigration). Hard limits: **crawl depth ≤ 2, ≤ 500 documents per run**.
  - **Bronze → MySQL** (raw, unstructured): `crawler/src/gfinance_crawler/news/` (sources: Google News RSS, Bing News RSS, GDELT, Hacker News, Reddit, local inbox folder `data/bronze/inbox/`; polite BFS fetcher with robots.txt) + `src/tcs_bot/medallion/bronze.py` + migration `src/tcs_bot/db/schema_002_medallion.sql` (`crawl_runs`, `bronze_documents`).
  - **Silver → Elasticsearch** (structured): `src/tcs_bot/medallion/silver/` — text extraction, entities, sentiment, TCS-relevance (filters out India's "Tax Collected at Source"), 65 risk rules across 10 categories, optional gpt-4.1-mini extraction, index `silver-tcs-risk`.
  - **Gold → Qdrant vectors + graph:** `src/tcs_bot/medallion/gold/` — embeddings (OpenAI if key set, else offline TF-IDF+SVD "LSA"), Qdrant collections, cosine k-NN graph, Louvain + KMeans clusters, centrality, PCA layout, self-contained interactive HTML graph, API routes `/tcs/risks/*` (`src/tcs_bot/api/tcs_routes.py`) and MCP tools.
  - CLI: `uv run tcs-bot medallion run|bronze|silver|gold|schedule [--max-docs N --max-depth D --cron "0 */6 * * *"]`; Compose service `medallion` (profile `app`).

### 2. Verified state at handoff (measured, not assumed)
- `uv run ruff check .` clean; `uv run ruff format --check .` clean.
- `uv run pytest tests/unit crawler/tests -m "not integration"` → **363 passed**.
- Live-infra integration for the layers → **23 passed** (`-m integration -k "bronze or silver or gold or data_layer"`), earlier full run: 23 integration + 17 e2e passed for App A.
- Migration 002 applied to the sandbox MySQL (tables `crawl_runs`, `bronze_documents` exist). On a fresh machine run `uv run tcs-bot migrate`.

### 3. NOT verified / unfinished — your work
1. **End-to-end medallion run has never been executed.** Do it: put a few clearly-labelled sample files in `data/bronze/inbox/` (or use `crawler/tests/fixtures/news/`), then `uv run tcs-bot medallion run`, and confirm: rows in MySQL `bronze_documents` → docs in ES `silver-tcs-risk` → points in Qdrant `gold_tcs_risk_*` → `data/gold/{graph.json,graph.graphml,summary.json,tcs_risk_graph.html}`.
2. **The graph HTML was never visually checked** (the agent was stopped before its screenshot step). Open `data/gold/tcs_risk_graph.html`, screenshot it with Playwright, and fix visual problems (500 nodes must stay smooth; cosine slider, cluster/category colouring, click-to-inspect panel, legend).
3. **Real crawling was never run** (sandbox blocked Google News, Bing, GDELT, Reddit, HN, news sites). On the user's PC run `uv run tcs-bot medallion bronze --max-docs 50` first, check the parsers against the real responses (feed/JSON shapes may differ from the synthetic fixtures — fix parsers + add fixtures captured from real responses), then scale to 500. Respect robots.txt and rate limits; do not bypass blocks or paywalls.
4. **Quality of risk detection on real text:** review a sample of silver output for false positives/negatives; tune `silver/classify.py` rules and the relevance threshold (`MEDALLION_RELEVANCE_THRESHOLD`).
5. **Gold on real data:** check cluster labels and cosine thresholds (`GOLD_GRAPH_K`, `GOLD_GRAPH_MIN_COSINE`) give meaningful groups; with an OpenAI key use `LLM_PROVIDER=openai` and re-run `uv run tcs-bot index` + `medallion gold`.
6. **Git history cleanup (optional):** five commits named `wip(...)` hold the medallion code (untested at the time; verified now as described above). Do not rewrite pushed history unless the user asks.
7. **CI (`.github/workflows/ci.yml`) lacks a Qdrant service** — the gold integration tests need `qdrant/qdrant:v1.12.4` on 6333 and `QDRANT_URL`. Add it, and run `tcs-bot migrate` is already there. CI currently cannot start: GitHub reports "account is locked due to a billing issue" — the user must fix billing under GitHub Settings → Billing before any workflow runs.
8. **Docs:** add the medallion section to `README.md` (commands, architecture, how to view the graph) and generate `docs/BUILD_LOG.md` from `git log --reverse` grouped by phase, so the user can follow the step-by-step.
9. **Docker:** the API image's `apt-get` step (tesseract/poppler) was never built in the sandbox (Debian mirror blocked). Run `docker compose --profile app up -d --build` on the PC and fix whatever breaks. `crawler/Dockerfile` has an empty arm64 supercronic checksum (`SUPERCRONIC_SHA1SUM_ARM64`) — pin it if you build on Apple Silicon.

### 4. Hard rules
- Python 3.13, `uv`; settings only via `get_settings()`; MySQL only via `db.mysql.transaction()`; LLM output always a validated Pydantic model; temperature 0.0; models `gpt-4.1-mini` / `text-embedding-3-small`.
- Tests offline by default; anything needing live MySQL/Redis/ES/Qdrant is marked `integration`/`e2e`. Never commit `.env` or secrets. Only synthetic data may be committed; any fixture must carry a "SYNTHETIC FIXTURE" marker. Never present invented text as real TCS news.
- Commit small and often; `git push` after each verified step. Before finishing run: `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/unit crawler/tests -m "not integration"`.

### 5. Report back with
What you ran and the exact results (counts), anything that failed and why, the graph screenshot path, and the first real-crawl stats (`discovered / fetched / stored / duplicates / blocked / failed` by source and depth).
