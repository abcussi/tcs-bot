# 005 — TCS risk intelligence: medallion pipeline (bronze -> silver -> gold)

## Why
Risk managers need a continuously refreshed, explainable view of risks around **Tata Consultancy
Services (TCS)** as a vendor: ethics, compliance/regulatory, legal, data privacy/cyber, labour/HR,
financial, operational/delivery, reputational, ESG and geopolitical/immigration — from news AND
from communities where employees/clients discuss TCS. There is no source DB, so we crawl.

## Scope
- **Bronze (MySQL, unstructured, immutable):** raw documents exactly as fetched (HTML/JSON/XML +
  HTTP metadata, crawl lineage: query, parent URL, depth, run id).
- **Silver (Elasticsearch, structured):** cleaned text + extracted fields + risk classification.
- **Gold / prod (Elasticsearch dense_vector kNN + analytics):** chunk & document embeddings, clusters, cosine
  similarity graph, cluster/risk summaries, interactive graph visualization.

## Acceptance criteria
- AC-005.1 Sources (level 0): Google News RSS, Bing News RSS, GDELT DOC API (news); Hacker News
  (Algolia API) and Reddit search JSON (communities); `LocalInbox` folder source for offline/manual
  docs (`data/bronze/inbox/*.{html,txt,json,pdf}`). A fixed TCS risk query set (ethics, compliance,
  lawsuit, fine, layoffs, data breach, visa fraud, investigation, controversy, outage, ...).
- AC-005.2 Crawl is breadth-first with **max depth 2** (0 = search/listing item, 1 = linked article
  or thread, 2 = TCS-relevant outbound links from depth-1 pages) and **max 500 documents** per run
  (both configurable, hard caps 2 / 500). Same-URL dedupe (canonical URL), robots.txt respected,
  per-domain delay, timeouts + retries; blocked/failed fetches are recorded, never crash the run.
- AC-005.3 Depth-2 links are followed only if anchor/URL text is TCS-relevant and the domain is in
  the allowed-domain list; off-topic pages are not stored.
- AC-005.4 Bronze rows are written to MySQL in transactions (batch), idempotent on
  `url_sha256`; each run is recorded in `crawl_runs` with params and stats.
- AC-005.5 Silver: for each unprocessed bronze doc -> main-text extraction (trafilatura, fallback
  bs4), normalization, language, word count, entities (orgs, people, money, regulators, places),
  TCS relevance score, risk signals (category, severity, evidence sentence), sentiment, overall
  doc risk score 0-100. LLM extraction (gpt-4.1-mini, T=0, structured output) when OpenAI is
  configured, deterministic rule-based classifier otherwise. Indexed into ES `silver-tcs-risk`.
- AC-005.6 Docs below the relevance threshold are marked `relevant=false` in silver (kept for audit,
  excluded from gold).
- AC-005.7 Gold: chunk relevant silver docs, embed (OpenAI text-embedding-3-small when configured,
  else offline TF-IDF+SVD "LSA" embedder), bulk-upsert into Elasticsearch indices
  `gold-tcs-risk-chunks--<model>` and `gold-tcs-risk-docs--<model>` (`dense_vector`, cosine,
  `index=true`, <= 4096 dims) with rich payloads and deterministic `_id`s (idempotent). The index
  name includes the embedding model to avoid mixing vector spaces. Qdrant is not used. ES errors
  surface as `DependencyUnavailable` (`vector_store_unavailable`, 503); the pipeline still writes
  the file artifacts and reports `vector_store: unavailable: ...`.
- AC-005.8 Analytics: numpy cosine similarity matrix over doc vectors; k-NN similarity graph
  (edges = cosine >= threshold, max k neighbours) with networkx; Louvain communities + KMeans
  clusters; centrality; 2-D projection (PCA); per-cluster top terms, dominant risk categories,
  representative docs. Persist `data/gold/graph.json`, `graph.graphml`, `summary.json` and a
  self-contained interactive `data/gold/tcs_risk_graph.html`.
- AC-005.9 API: `GET /tcs/risks/search?q=&k=` (semantic search over gold), `GET /tcs/risks/summary`,
  `GET /tcs/risks/graph`; MCP tool `search_tcs_risks(query, k)`.
- AC-005.10 CLI: `tcs-bot medallion bronze|silver|gold|run [--max-docs N --max-depth D]`; cron entry.
- AC-005.11 All layers have offline unit tests (fixtures; no network). Integration tests against
  live MySQL/ES are marked `integration`.
- AC-005.12 Typed read layer: other code reads bronze / silver / gold through `tcs_bot.medallion.read`
  (`TcsData`, join key `bronze_documents.id == silver doc_id == gold doc_id`), never raw SQL / ES.
  (a) `list_documents(filters, page)` is silver-backed; filters: category (any of), min severity,
  min risk score, source, source type, relevant (default true), published date range, gold
  cluster / community, free text on title / text; sort by risk_score | published_at |
  tcs_relevance (asc|desc); `limit` <= 100 + `offset` (window <= 10 000); returns the total.
  (b) `get_document(doc_id)` returns silver fields (risk signals + evidence), bronze lineage (url,
  depth, parent url, query, run id, fetch time, HTTP status, source) and the gold node (cluster,
  community, centrality, merged duplicates); bronze `raw_content` and silver `clean_text` only on
  explicit request (`include_raw` / `include_text`) and never logged. (c) `similar_documents`
  (ES kNN on the stored gold vector, cosine-ordered, self excluded, merged duplicates resolved to
  the kept doc), `category_breakdown`, `risk_overview` (ES aggregations by category, severity,
  source, source type, month + top documents), `list_clusters` / `get_cluster`, `search`.
  (d) API (prefix `/tcs/risks`, existing routes unchanged): `GET /documents`, `/documents/{doc_id}`,
  `/documents/{doc_id}/similar`, `/categories`, `/overview`, `/clusters`, `/clusters/{cluster_id}`
  with typed `response_model`s, validated / capped query params and the JSON error envelope.
  (e) MCP tools `list_tcs_risks`, `get_tcs_risk_document`, `similar_tcs_risks`,
  `tcs_risk_clusters` reuse the read layer. (f) Errors: gold missing -> `gold_not_built` (404),
  unknown id -> `document_not_found` / `cluster_not_found` (404), no vector ->
  `document_not_in_gold` (404), Elasticsearch down -> `silver_unavailable` /
  `vector_store_unavailable` (503), MySQL down -> `bronze_unavailable` (503). The read layer never
  writes to any layer.
