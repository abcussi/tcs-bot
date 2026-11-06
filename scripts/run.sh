#!/usr/bin/env bash
# Thin wrapper for humans/CI. Usage: scripts/run.sh [all|up|down|migrate|seed|index|crawl|api|mcp|test|docker]
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example — set OPENAI_API_KEY"; }
uv sync --quiet

case "${1:-all}" in
  test)   uv run pytest tests/unit crawler/tests -q ;;
  docker) docker compose --profile app up -d --build ;;
  *)      uv run vendor-risk-ai "${1:-all}" ;;
esac
