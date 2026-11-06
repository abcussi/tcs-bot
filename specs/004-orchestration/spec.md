# 004 — Orchestration, Docker and run script

- AC-004.1 `uv run tcs-bot all` (main()) : docker compose up mysql/redis/elasticsearch -> wait healthy ->
  migrate -> seed -> ensure ES indices -> embed+index seed docs -> crawler once (non-fatal) -> start API.
- AC-004.2 Subcommands up|down|migrate|seed|index|crawl|api|mcp|status.
- AC-004.3 `Dockerfile` (api, multi-stage uv) and `crawler/Dockerfile` (supercronic cron); `docker-compose.yml` with healthchecks.
- AC-004.4 `scripts/run.sh` thin wrapper for humans/CI.
