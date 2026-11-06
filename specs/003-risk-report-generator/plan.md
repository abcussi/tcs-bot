# 003 — Plan / contracts
```python
schemas/report.py   RedFlag, FinancialHealth, ComplianceRecommendation, SourceRef, RiskReport, RiskLevel
schemas/api.py      ReportRequest, DocumentIn, DocumentOut, VendorOut, HealthOut, ErrorEnvelope
api/errors.py       AppError, NotFoundError(404), ValidationAppError(422), UpstreamError(502), DependencyUnavailable(503) + register_error_handlers(app)
api/app.py          def create_app(service: ReportService | None = None) -> FastAPI
ingestion/extract.py def extract_text(data: bytes, filename: str) -> str
ingestion/normalize.py def normalize_text(s: str) -> str
ingestion/chunker.py def chunk_text(text: str, size: int, overlap: int) -> list[str]
rag/embeddings.py   class Embedder(Protocol): model: str; dim: int; embed(texts: list[str]) -> np.ndarray
                    OpenAIEmbedder, HashEmbedder, get_embedder()
rag/retriever.py    def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]
                    class Retriever: retrieve(query: str, chunks: list[ChunkRecord], k) -> list[tuple[ChunkRecord, float]]
llm/prompts.py      SYSTEM_PROMPT, build_prompt() -> ChatPromptTemplate
llm/chain.py        def build_report_chain(settings) -> Runnable[dict, RiskReport]  (fake when settings.use_fake_llm)
services/report_service.py
    class ReportService(vendors, documents, reports, cache, embedder, chain, quote_lookup)
        ingest_document(vendor_id, DocumentCreate) -> DocumentOut
        generate_report(ReportRequest) -> RiskReport
        latest_report(vendor_id) -> RiskReport
    def build_default_service() -> ReportService
mcp_server/server.py  mcp = FastMCP("vendor-risk"); tools; def main()
```
