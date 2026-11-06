# 003 — Plan / contracts
```python
schemas/report.py   RedFlag, FinancialHealth, ComplianceRecommendation, SourceRef, RiskReport, RiskLevel
schemas/api.py      ReportRequest, DocumentIn, DocumentOut, VendorOut, HealthOut, ErrorEnvelope
api/errors.py       AppError, NotFoundError(404), ValidationAppError(422), UpstreamError(502), DependencyUnavailable(503) + register_error_handlers(app)
api/app.py          def create_app(service: ReportService | None = None) -> FastAPI
ingestion/extract.py def extract_text(data: bytes, filename: str) -> str
                    def extract_document(data, filename) -> ExtractionResult(text, pages, ocr_pages, skipped_pages)
ingestion/normalize.py def normalize_text(s: str) -> str
ingestion/chunker.py def chunk_text(text: str, size: int, overlap: int) -> list[str]
rag/embeddings.py   class Embedder(Protocol): model: str; dim: int; embed(texts: list[str]) -> np.ndarray
                    OpenAIEmbedder, HashEmbedder, get_embedder()
rag/retriever.py    def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]
                    class Retriever: retrieve(query: str, chunks: list[ChunkRecord], k) -> list[tuple[ChunkRecord, float]]
llm/prompts.py      SYSTEM_PROMPT, build_prompt() -> ChatPromptTemplate
llm/chain.py        def build_report_chain(settings) -> Runnable[dict, RiskReport]  (fake when settings.use_fake_llm)
services/report_service.py
    class ReportService(vendors, documents, reports, cache, embedder, chain, quote_lookup,
                        chunk_indexer: Callable[[int, list[ChunkRecord]], int] | None = None,
                        settings: Settings | None = None)
        list_vendors(limit=100, offset=0) -> list[Vendor] ; get_vendor(vendor_id) -> Vendor
        ingest_document(vendor_id, DocumentCreate) -> DocumentOut
        generate_report(ReportRequest) -> RiskReport
        latest_report(vendor_id) -> RiskReport
        embed_all_pending() -> int ; invalidate_vendor_reports(vendor_id) -> int
    def report_cache_key(vendor_id, chunks, focus_areas, top_k, model) -> str   # AC-003.8
    def build_default_service(settings: Settings | None = None) -> ReportService
mcp_server/server.py  mcp = FastMCP("vendor-risk"); tools; def main()
```

## Implementation notes (recorded deviations)
- `mcp` >= 2.x renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`; `server.py` imports either.
- `.with_structured_output(RiskReport, method="function_calling")`: OpenAI strict json_schema mode
  rejects fields with defaults. Model/temperature unchanged (gpt-4.1-mini, 0.0).
- `create_app(service=None, health_checks=None)` — health checks injectable for tests.
- Chain input accepts optional `data_gaps: list[str]`, merged into the report.
- `embed_all_pending()` returns number of documents embedded.
- Stored embeddings with a different `embedding_model` are ignored and re-embedded (hash vs OpenAI
  vectors share dim 1536 but are not comparable).
- Lookup by `vendor_name`: exact name/external_id, then unique partial match; ambiguous -> 422 `ambiguous_vendor`.
- `AppError` and subclasses live in `api/errors.py` but are also raised by `services/` and
  `ingestion/` (and caught by `mcp_server/`). Accepted for now; if the service layer is ever reused
  without FastAPI, move them to a neutral `vendor_risk_ai/errors.py` and re-export from `api.errors`.
- Retrieval never queries ES `vendor-chunks`: it is a write-only mirror (numpy cosine over
  MySQL-stored chunks is the retrieval path, constitution §3). ES down therefore only affects quotes.
- Embedding-model switch (fake <-> openai): incompatible stored vectors are re-embedded *on the fly
  per request and not persisted*; `embed_all_pending()` only handles documents with no chunks.
  A persisted re-embed path is an open task (tasks.md).
- `DocumentOut.indexed` must mean "ES accepted the chunks" (`index_chunks(...) > 0`), not "the
  indexer did not raise" — `search.elastic.index_chunks` swallows transport errors and returns 0.
- MCP tool `generate_vendor_risk_report` should declare `-> RiskReport` so its MCP `outputSchema`
  is the RiskReport schema (today it is an untyped object).

## Architect review 2026-11-06
Open findings are tracked in `tasks.md` under "Open (architect review 2026-11-06)".
