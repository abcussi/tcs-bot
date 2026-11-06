# 004 — Tasks

- [x] `main.py` orchestrator: `all` + up|down|status|migrate|seed|index|crawl|api|mcp, `--no-infra` (AC-004.1/.2)
- [x] `Dockerfile` (multi-stage uv, OCR deps), `crawler/Dockerfile` (supercronic), `docker-compose.yml` (AC-004.3)
- [x] `scripts/run.sh` (AC-004.4)

## Open (architect review 2026-11-06)
- [x] BLOCKER `docker-compose.yml:37`: ES healthcheck `curl ...health?wait_for_status=yellow&timeout=5s`
      is unquoted inside `sh -c`, so `&` backgrounds curl and the check always exits 0 (verified).
      ES is "healthy" before it listens -> api CMD `vendor-risk-ai index` -> `ensure_indices()` raises
      -> api container exits (no restart policy). Fix: quote the URL
      (`curl -fs 'http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=5s'`).
- [x] `docker-compose.yml` api: add `restart: on-failure` and pin in-network `MYSQL_PORT: "3306"`
      (a host-side `MYSQL_PORT` override in .env currently leaks into the container's DSN).
- [x] `Dockerfile:31`: add `--start-period=120s` (migrate+seed+index run before uvicorn listens).
- [x] `env_file: .env` is mandatory for the `app` profile (compose fails without it); either
      `env_file: [{path: .env, required: false}]` or keep the README `cp .env.example .env` step.
- [x] `all`: crawl runs before the API; with Google unreachable/blackholed it can block ~6 min
      (8 tickers x 3 attempts x 15 s timeout). Consider running the crawl after the API starts or with
      a short timeout in the orchestrator. -> done: `all` runs the first crawl in a daemon thread.
      (env_file: kept mandatory; `scripts/run.sh` and README create .env from .env.example.)
