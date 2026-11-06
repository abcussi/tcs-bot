---
name: run-stack
description: Boot and verify the full local stack (MySQL, Redis, Elasticsearch, crawler, API, MCP). Use when asked to run, demo or smoke-test the app.
---
1. `uv sync`
2. `uv run vendor-risk-ai all` (or step by step: `up`, `migrate`, `seed`, `index`, `crawl`, `api`).
   If the Docker daemon is not running: start Docker Desktop (or `sudo dockerd &` on Linux).
3. Smoke test:
   - `curl -s localhost:8000/health`
   - `curl -s localhost:8000/vendors | head`
   - `curl -s -XPOST localhost:8000/reports -H 'content-type: application/json' -d '{"vendor_id":1}'`
4. MCP: `uv run vendor-risk-mcp` (stdio) — registered in `.mcp.json`.
5. `uv run pytest -m "integration or e2e"` with the stack up.
