# 003 — AI vendor risk report generator (API + RAG + LCEL + MCP)

## Acceptance criteria
- AC-003.1 FastAPI backend with Pydantic request/response schemas and custom error handlers
  (`AppError` -> JSON `{"error":{"code","message","details"}}`; validation errors -> 422 same envelope).
- AC-003.2 Ingestion: accepts text or PDF (pypdf; OCR via pytesseract when a page has no text layer),
  normalizes whitespace/unicode, chunks (size/overlap from settings).
- AC-003.3 Embeddings: `text-embedding-3-small` via OpenAI; offline `HashEmbedder` (deterministic, same dim 1536) when fake provider.
- AC-003.4 Retrieval: numpy cosine similarity over the vendor's chunk matrix, top-k, returns scores.
- AC-003.5 LCEL pipeline `ChatPromptTemplate | ChatOpenAI(model="gpt-4.1-mini", temperature=0.0).with_structured_output(RiskReport)`;
  `PydanticOutputParser` fallback path; fake chain for offline mode producing a valid `RiskReport` heuristically.
- AC-003.6 `RiskReport` JSON contains: vendor_id, vendor_name, overall_risk_score (0-100), risk_level
  (low|medium|high|critical), red_flags[] (category, severity, description, evidence), financial_health
  (summary, liquidity, profitability, leverage, market_signal), compliance_recommendations[] (priority, recommendation, regulation_reference),
  risk_narrative, sources[] (chunk refs), data_gaps[], generated_at, model.
- AC-003.7 Endpoints: `GET /health`, `GET /vendors`, `GET /vendors/{id}`, `POST /vendors/{id}/documents` (JSON text),
  `POST /vendors/{id}/documents/upload` (multipart PDF/txt), `POST /reports` (`{vendor_id, focus_areas?, top_k?, use_cache?}`),
  `GET /reports/{vendor_id}/latest`.
- AC-003.8 Reports are cached (key includes vendor id + doc fingerprint + normalized focus + effective top_k
  + generating model id, so a different `top_k` or a provider switch never serves a stale report) and
  persisted to `risk_reports`.
- AC-003.9 MCP server (FastMCP, stdio + streamable-http) exposes tool `generate_vendor_risk_report(vendor_id|vendor_name, focus_areas)` and `list_vendors`.
- AC-003.10 `GET /dashboard` (and `GET /` redirecting to it) serves a self-contained page that lists vendors,
  generates a report on demand through `POST /reports`, and shows the four report sections: overall risk
  score, red flags, financial health summary, compliance recommendations. All report text is rendered as
  text (never HTML).
- AC-003.11 The dashboard has an "MCP console" tab. `GET /mcp-console/tools` lists the MCP server's tools
  with their input schemas; `POST /mcp-console/call {tool, arguments}` performs a real in-memory MCP
  `tools/call` and returns the request, `is_error`, text, `structured_content` and timing. A tool error is
  a normal 200 response with `is_error: true`; an unknown tool is a 404 error envelope. A
  `generate_vendor_risk_report` answer is rendered with the same four report sections.
