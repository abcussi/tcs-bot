# syntax=docker/dockerfile:1
# Vendor Risk AI — API image (FastAPI monolith). Multi-stage: build venv with uv, ship slim runtime.

FROM python:3.13-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.11.32
WORKDIR /app
# dependency layer (cached until the lockfile changes)
COPY pyproject.toml uv.lock ./
COPY crawler/pyproject.toml crawler/pyproject.toml
RUN mkdir -p src/tcs_bot crawler/src/gfinance_crawler \
 && touch src/tcs_bot/__init__.py crawler/src/gfinance_crawler/__init__.py \
 && uv sync --frozen --no-dev --no-install-workspace
# application layer
COPY src ./src
COPY crawler ./crawler
RUN uv sync --frozen --no-dev

FROM python:3.13-slim AS runtime
# OCR fallback for scanned PDFs (pytesseract / pdf2image)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils curl \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app data ./data
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 API_HOST=0.0.0.0 API_PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 CMD curl -fs http://localhost:8000/health || exit 1
# migrate + seed + index are idempotent, then serve
CMD ["sh", "-c", "tcs-bot migrate && tcs-bot seed && tcs-bot index && tcs-bot api"]
