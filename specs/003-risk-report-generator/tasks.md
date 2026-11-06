- [x] schemas (`schemas/api.py`: ReportRequest, DocumentIn, DocumentOut, VendorOut, HealthOut, ErrorEnvelope) — AC-003.1/.7
- [x] errors (`api/errors.py`: AppError + 404/422/502/503 subclasses, handlers incl. 422 + unhandled 500 + HTTP 404/405) — AC-003.1
- [x] ingestion (`extract.py` pypdf + OCR fallback, `normalize.py`, `chunker.py`) — AC-003.2
- [x] rag (`embeddings.py` OpenAIEmbedder / HashEmbedder / get_embedder, `retriever.py` cosine_top_k + Retriever) — AC-003.3/.4
- [x] llm (`prompts.py`, `chain.py`: structured-output chain, PydanticOutputParser fallback, fake heuristic chain, finalize_report) — AC-003.5/.6
- [x] service (`services/report_service.py`: ingest, generate w/ cache + persistence, latest, embed_all_pending, build_default_service) — AC-003.8
- [x] api (`api/app.py` create_app + lifespan, `routes.py`, `deps.py`) — AC-003.7
- [x] mcp (`mcp_server/server.py`: generate_vendor_risk_report, list_vendors, stdio | streamable-http) — AC-003.9
- [x] unit tests: `tests/unit/test_ai_*.py`, `test_api_*.py`, `test_mcp_*.py` (+ shared fakes in `tests/unit/conftest.py`)
- [ ] integration/e2e tests against the live stack (qa-engineer)

Notes / deviations (to be folded into plan.md by the architect):
- `mcp>=2` renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`; server.py imports FastMCP when available, else MCPServer.
- `create_app(service=None, health_checks=None)`: optional `health_checks` mapping (mysql/redis/elasticsearch -> callable) for tests.
- `with_structured_output(RiskReport, method="function_calling")`: OpenAI strict json_schema rejects RiskReport's defaulted fields.
- Chain input accepts optional `"data_gaps": list[str]`, which is merged into the report.
- `embed_all_pending()` returns the number of documents chunked (not the number of chunks).

## Open (architect review 2026-11-06) — owner: developer
- [ ] AC-003.8: add effective `top_k` and the chain's model id to `report_cache_key`
      (`services/report_service.py:81`, call site `:303`); today a `top_k=2` report is served for
      `top_k=10`, and fake-model reports are served from Redis after switching to openai.
- [ ] `_chunk_and_store` (`services/report_service.py:234-237`): `indexed = indexer(...) > 0`.
- [ ] Re-embed path: `embed_all_pending(reembed=True)` (or a `reembed` CLI step) that re-embeds and
      persists chunks whose `embedding_model != embedder.model`, and re-indexes them into ES.
- [ ] Cache-hit path loads every chunk embedding (`chunks_for_vendor`) just to fingerprint; add a
      light `DocumentRepository.chunk_ids_for_vendor(vendor_id)` (or fingerprint query) for the key.
- [ ] AC-003.5: wire `build_report_chain_with_parser` as `.with_fallbacks([...])` on the
      structured chain (it is currently dead code), or drop the AC wording.
- [ ] Pin constitution values: validate `openai_temperature == 0.0`, `openai_chat_model ==
      "gpt-4.1-mini"`, `openai_embedding_model == "text-embedding-3-small"` in `config.py`
      (env can currently override them; a 3072-dim model would break embeddings and ES dims).
- [ ] MCP: annotate `generate_vendor_risk_report(...) -> RiskReport` for a typed outputSchema.
- [ ] Constitution §6: `latest_quote` swallows ES errors and returns None, so ES-down reads as
      "No market quote found" — return a sentinel / raise so the service emits the explicit
      "Market data unavailable (search backend error)" gap (service branch `:340-346` is dead).
