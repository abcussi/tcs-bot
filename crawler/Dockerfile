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
RUN mkdir -p src/tcs_bot crawler/src/gfinance_crawler \
 && touch src/tcs_bot/__init__.py crawler/src/gfinance_crawler/__init__.py \
 && uv sync --frozen --no-dev --package gfinance-crawler --no-install-workspace
# application layer: only the crawler package is installed into the venv
COPY crawler ./crawler
RUN uv sync --frozen --no-dev --package gfinance-crawler

FROM python:3.13-slim AS runtime
ARG SUPERCRONIC_VERSION=v0.2.33
# Per-arch SHA1 of supercronic-linux-${TARGETARCH} for the version above (BuildKit sets
# TARGETARCH; the legacy builder falls back to amd64). An empty checksum skips verification
# for that arch: the arm64 sum is not pinned yet - set it with
#   --build-arg SUPERCRONIC_SHA1SUM_ARM64=<sha1 from the supercronic release page>
# Bump every checksum together with SUPERCRONIC_VERSION.
ARG SUPERCRONIC_SHA1SUM_AMD64=71b0d58cc53f6bd72cf2f293e09e294b79c666d8
ARG SUPERCRONIC_SHA1SUM_ARM64=
ARG TARGETARCH=amd64
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) sha1="${SUPERCRONIC_SHA1SUM_AMD64}" ;; \
      arm64) sha1="${SUPERCRONIC_SHA1SUM_ARM64}" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH} (amd64|arm64)" >&2; exit 1 ;; \
    esac; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    curl -fsSLo /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${TARGETARCH}"; \
    if [ -n "${sha1}" ]; then \
      echo "${sha1}  /usr/local/bin/supercronic" | sha1sum -c -; \
    else \
      echo "WARNING: no supercronic SHA1 pinned for ${TARGETARCH}; checksum not verified" >&2; \
    fi; \
    chmod +x /usr/local/bin/supercronic; \
    apt-get purge -y curl; apt-get autoremove -y; rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/crawler /app/crawler
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
CMD ["supercronic", "/app/crawler/cron/crontab"]
