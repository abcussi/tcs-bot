# 003 — Plan / contracts
```python
schemas/report.py   RedFlag, FinancialHealth, ComplianceRecommendation, SourceRef, RiskReport, RiskLevel
schemas/api.py      ReportRequest, DocumentIn, DocumentOut, VendorOut, HealthOut, ErrorEnvelope
errors.py           AppError, NotFoundError(404), ValidationAppError(422), UpstreamError(502), DependencyUnavailable(503)
api/errors.py       re-exports the above + register_error_handlers(app)
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
llm/chain.py        def build_report_chain(settings) -> Runnable[dict, RiskReport]  (fake when settings.use_fake_llm;
                        structured output .with_fallbacks([PydanticOutputParser path]))
                    def report_model_id(chain, settings) -> str   # model id stamped on reports (chain metadata)
services/report_service.py
    class ReportService(vendors, documents, reports, cache, embedder, chain, quote_lookup,
                        chunk_indexer: Callable[[int, list[ChunkRecord]], int] | None = None,
                        settings: Settings | None = None)
        list_vendors(limit=100, offset=0) -> list[Vendor] ; get_vendor(vendor_id) -> Vendor
        ingest_document(vendor_id, DocumentCreate) -> DocumentOut
        generate_report(ReportRequest) -> RiskReport
        latest_report(vendor_id) -> RiskReport
        embed_all_pending(reembed: bool = False) -> int ; invalidate_vendor_reports(vendor_id) -> int
        model_id -> str (property)
    def report_cache_key(vendor_id, chunks: Sequence[ChunkRecord | (document_id, chunk_index)],
                         focus_areas, top_k: int, model: str) -> str   # AC-003.8
        # "report:{vendor}:{fingerprint}:k{top_k}:{model}:{focus}"
    def build_default_service(settings: Settings | None = None) -> ReportService
mcp_server/server.py  mcp = FastMCP("vendor-risk"); generate_vendor_risk_report(...) -> RiskReport;
                      list_vendors(limit) -> list[dict]; def main()
```

## Implementation notes (recorded deviations)
- `mcp` >= 2.x renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`; `server.py` imports either.
- `.with_structured_output(RiskReport, method="function_calling")`: OpenAI strict json_schema mode
  rejects fields with defaults. Model/temperature unchanged (gpt-4.1-mini, 0.0).
- `create_app(service=None, health_checks=None)` — health checks injectable for tests.
- Chain input accepts optional `data_gaps: list[str]`, merged into the report.
- `embed_all_pending(reembed=False)` returns the number of documents processed: documents without
  chunks are chunked+embedded; documents with missing / other-model embeddings (or all, with
  `reembed=True`) are re-embedded, persisted via `save_chunks` (one transaction per document) and
  re-indexed into ES; the vendor's cached reports are invalidated.
- Stored embeddings with a different `embedding_model` are ignored and re-embedded (hash vs OpenAI
  vectors share dim 1536 but are not comparable).
- Lookup by `vendor_name`: exact name/external_id, then unique partial match; ambiguous -> 422 `ambiguous_vendor`.
- `AppError` and subclasses live in the neutral `vendor_risk_ai/errors.py` (used by `services/`,
  `ingestion/`, `mcp_server/`); `api/errors.py` re-exports them and owns the FastAPI handlers.
- Retrieval never queries ES `vendor-chunks`: it is a write-only mirror (numpy cosine over
  MySQL-stored chunks is the retrieval path, constitution §3). ES down therefore only affects quotes.
- `build_default_service` wires `search.elastic.latest_quote_strict` (raises `SearchUnavailable`),
  so ES down yields the "Market data unavailable (search backend error)" data gap while a missing
  quote yields "No market quote found" (constitution §6).
- Cache key: computed first from `DocumentRepository.chunk_ids_for_vendor` (ids only), so a cache
  hit never loads chunk texts/embeddings; on a miss it is recomputed from the loaded chunks.
  `top_k` is the effective one (`req.top_k or settings.rag_top_k`); `model` is `report_model_id`.
- `ingest_document` normalizes + chunks BEFORE inserting the document, so empty content -> 422
  `empty_document` with no `vendor_documents`/audit row left behind.
- `config.Settings` rejects embedding models that are not 1536-dim (allowlist:
  `text-embedding-3-small`) and logs a warning when `openai_temperature != 0.0` or
  `openai_chat_model != "gpt-4.1-mini"` (constitution §3; overrides stay possible for experiments).
- Embedding-model switch (fake <-> openai): incompatible stored vectors are re-embedded on the fly
  per request (not persisted) until the `index` step (`embed_all_pending`) persists them.
- `DocumentOut.indexed` means "ES accepted the chunks" (`chunk_indexer(...) > 0`), not "the
  indexer did not raise" — `search.elastic.index_chunks` swallows transport errors and returns 0.
- MCP tool `generate_vendor_risk_report` returns `RiskReport`, so its MCP `outputSchema` is the
  RiskReport schema and `structuredContent` its JSON dump (direct Python calls get the model).

## Architect review 2026-11-06
Open findings are tracked in `tasks.md` under "Open (architect review 2026-11-06)".
