# syntax=docker/dockerfile:1
# gfinance-crawler — Google Finance -> Elasticsearch, scheduled by supercronic (AC-002.5).
# Build from the repo root:  docker build -f crawler/Dockerfile .

FROM python:3.13-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.11.32
WORKDIR /app
# dependency layer: the workspace root must exist for uv to resolve the lockfile
COPY pyproject.toml uv.lock ./
COPY crawler/pyproject.toml crawler/pyproject.toml
RUN mkdir -p src/vendor_risk_ai crawler/src/gfinance_crawler \
 && touch src/vendor_risk_ai/__init__.py crawler/src/gfinance_crawler/__init__.py \
 && uv sync --frozen --no-dev --package gfinance-crawler --no-install-workspace
# application layer: only the crawler package is installed into the venv
COPY crawler ./crawler
RUN uv sync --frozen --no-dev --package gfinance-crawler

FROM python:3.13-slim AS runtime
ARG SUPERCRONIC_VERSION=v0.2.33
# sha1 of supercronic-linux-amd64 for the version above; pass an empty value to skip (e.g. arm64)
ARG SUPERCRONIC_SHA1SUM=71b0d58cc53f6bd72cf2f293e09e294b79c666d8
ARG TARGETARCH=amd64
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSLo /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${TARGETARCH}" \
 && if [ -n "${SUPERCRONIC_SHA1SUM}" ]; then \
      echo "${SUPERCRONIC_SHA1SUM}  /usr/local/bin/supercronic" | sha1sum -c -; fi \
 && chmod +x /usr/local/bin/supercronic \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/crawler /app/crawler
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
CMD ["supercronic", "/app/crawler/cron/crontab"]
