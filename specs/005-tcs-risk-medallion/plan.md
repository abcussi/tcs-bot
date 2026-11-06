# 005 — Plan / contracts

## Ownership
| Area | Path | Owner |
|---|---|---|
| Sources + BFS crawler | `crawler/src/gfinance_crawler/news/` | agent BRONZE |
| Bronze persistence + migration 002 | `src/tcs_bot/medallion/bronze.py`, `src/tcs_bot/db/schema_002_medallion.sql`, `db/migrate.py` | agent BRONZE |
| Silver | `src/tcs_bot/medallion/silver/` | agent SILVER |
| Gold + analytics + viz + API/MCP | `src/tcs_bot/medallion/gold/`, `api/tcs_routes.py`, `mcp_server/` tool | agent GOLD |
| Typed read layer (AC-005.12) | `src/tcs_bot/medallion/read/`, read routes in `api/tcs_routes.py`, read tools in `mcp_server/server.py` | agent READ |
| Shared models, CLI, compose, config | `src/tcs_bot/medallion/models.py`, `medallion/pipeline.py`, `main.py`, `config.py` | lead |

## Shared models (`tcs_bot.medallion.models`) — the contract
`RiskCategory` (Literal), `RawFetch` -> `BronzeDocument`, `RiskSignal`, `Entities`, `SilverRecord`,
`GoldHit`, `ClusterSummary`, `CrawlStats`. See the module for fields.
The crawler package defines its own `gfinance_crawler.news.models.RawDocument` with the same
fields as `BronzeDocument` minus `id` (crawler never imports tcs_bot); bronze maps 1:1.

## Signatures
```python
# crawler: gfinance_crawler.news
sources.py   class Source(Protocol): name: str; source_type: str; def search(self, query: str, limit: int) -> list[RawDocument]
             GoogleNewsRSS, BingNewsRSS, GdeltDoc, HackerNews, RedditSearch, LocalInbox(path)
fetch.py     class PoliteFetcher(client=None, user_agent, timeout, delay, robots=True): fetch(url) -> FetchResult
links.py     def extract_links(html: str, base_url: str) -> list[tuple[str, str]]   # (url, anchor text)
             def is_tcs_relevant(text: str) -> bool ; def canonical_url(url: str) -> str
bfs.py       class RiskCrawler(sources, fetcher, allowed_domains, max_docs=500, max_depth=2):
                 crawl(queries: list[str]) -> Iterator[RawDocument] ; stats -> dict
queries.py   TCS_RISK_QUERIES: list[str] ; DEFAULT_ALLOWED_DOMAINS: list[str]

# app: tcs_bot.medallion
bronze.py    class BronzeRepository: start_run(params)->int; finish_run(run_id, stats); save_batch(run_id, docs: list[BronzeDocument]) -> int
                 iter_unprocessed(layer="silver", limit=None) -> Iterator[BronzeDocument]; mark_processed(ids, layer)
                 count() -> int
             def run_bronze(max_docs=500, max_depth=2, queries=None, sources=None) -> CrawlStats
silver/      extract.py  def extract_main_text(raw: str, content_type: str, url: str) -> tuple[str, str|None]  # text, title
             classify.py def classify(text: str, title: str) -> tuple[list[RiskSignal], float]  # signals, risk_score
             entities.py def extract_entities(text: str) -> Entities
             sentiment.py def sentiment(text: str) -> float   # -1..1
             llm.py      def llm_extract(doc, text, settings, base=None, chain=None) -> SilverRecord | None  # OpenAI path
             store.py    class SilverStore(es=None): ensure_index(); upsert(records) -> int; iter_relevant() -> Iterator[SilverRecord]; get(doc_id)
             pipeline.py def build_silver_record(doc: BronzeDocument) -> SilverRecord ; def run_silver(limit=None) -> dict
                         # helpers: is_relevant(relevance, word_count, title, threshold) -> bool;
                         #          attribute_risk(risk_score, relevance, threshold) -> float
             classify.py also: title_names_tcs(title) -> bool; length_damping(words) -> float
gold/        embedder.py  class LsaEmbedder(dim=256).fit(texts)  # unigram TF-IDF + SVD rank min(dim, n-1, max(6, 0.3 n)); get_gold_embedder(settings, corpus) -> Embedder-like
             vector_store.py class GoldVectorStore(es=None, model_slug, prefix): ensure(dim, recreate=False); upsert_docs/upsert_chunks -> int (+ last_failed_ids);
                           search(vector, k, categories, min_risk, level, *, sources, source_types, cluster, community) -> list[GoldHit]  # ES knn + pre-filter, score = cosine
                           count(level); stats(); delete_docs(doc_ids); drop(); errors -> DependencyUnavailable('vector_store_unavailable') / NotFoundError('gold_not_built')
             analytics.py cosine_matrix(X); knn_graph(X, ids, k, threshold) -> nx.Graph; kmeans_clusters(X, k_range=None, seed, min_size=2) -> (labels, info); louvain_communities(G, seed, resolution=1.0, min_size=2);
                          cluster_summaries(X, labels, records, texts, S=None) -> list[ClusterSummary]; project_2d(X)
                          suggest_k(n)=clamp(round(sqrt(n/2)),2,12); graph_params(n, k, min_cosine) -> (k_eff<=round(sqrt(n)), min_cosine)
             dedupe.py    dedupe_records(records, text_of=None, S=None) -> DedupeResult(records, merged: {kept_doc_id: [dropped SilverRecord]})
                          # gold-time only: same normalised title (+ text overlap) or near-identical text; richest doc kept, silver untouched
             viz.py       def render_html(graph_json: dict, path: Path) -> Path
             pipeline.py  def run_gold() -> dict   # writes data/gold/*; dedupes, embeds embedding_text(r) (title x3 + summary + first 150 words), builds the graph
                          build_graph_json(records, X, texts, settings, model, n_chunks, merged=None); nodes carry merged_doc_ids / also_seen_in / merged_sources
```
## Read layer contracts (AC-005.12) — `tcs_bot.medallion.read`
Inputs / outputs are Pydantic models in `read/models.py`; existing `SilverRecord`, `RiskSignal`,
`GoldHit`, `ClusterSummary` are reused.
```python
models.py   DocumentFilters(category: list[RiskCategory]=[], min_severity: Severity|None, min_risk_score: 0..100|None, source: list[str]=[],
                source_type: list[SourceType]=[], relevant: bool|None=True, published_from/published_to: datetime|None, cluster: int|None,
                community: int|None, q: str|None)      # ANDed; relevant=None -> all docs
            Page(limit: 1..100 = 20, offset >= 0 = 0, sort: risk_score|published_at|tcs_relevance = risk_score, order: asc|desc = desc)  # offset+limit <= 10_000
            DocumentSummary / DocumentPage(total, limit, offset, items) / DocumentDetail(doc_id, silver: SilverView, bronze: BronzeLineage|None, gold: GoldNode|None)
            SilverView(SilverRecord, clean_text: str|None = None) ; BronzeLineage(id, run_id, url, canonical_url, parent_url, depth, source, source_type, query,
                title, content_type, http_status, published_at, fetched_at, silver_processed_at, raw_chars, raw_content: str|None = None)
            GoldNode(in_gold, merged_into, cluster, cluster_label, community, degree, betweenness, eigenvector, x, y, merged_doc_ids, also_seen_in, merged_sources, neighbors)
            SimilarDocuments(doc_id, resolved_doc_id, model, neighbours: list[GoldHit]) ; CategoryBreakdown / RiskOverview (RiskBucket(key, count, avg_risk_score,
                max_risk_score), CountBucket(key, count)) ; ClusterDetail(kind, summary: ClusterSummary, members: list[DocumentSummary])
data.py     class TcsData(silver: SilverStore|None, vector_store: GoldVectorStore|None, bronze: BronzeReader|None, settings: Settings|None)
                list_documents(filters=None, page=None) -> DocumentPage
                get_document(doc_id, *, include_raw=False, include_text=False) -> DocumentDetail
                similar_documents(doc_id, k=10) -> SimilarDocuments            # k <= 50
                search(query, k=10, filters=None) -> list[GoldHit]            # filters used: category, min_risk_score, source, source_type, cluster, community
                category_breakdown(relevant_only=True) -> CategoryBreakdown ; risk_overview(relevant_only=True, top_n=5) -> RiskOverview
                list_clusters(kind="cluster"|"community") -> list[ClusterSummary] ; get_cluster(cluster_id, kind="cluster") -> ClusterDetail
            get_tcs_data() / set_tcs_data(data|None)                           # process-wide instance (API dependency, MCP tools, tests)
bronze_reader.py  BronzeReader(Protocol): get(doc_id, include_raw=False) -> BronzeLineage|None ; MysqlBronzeReader (transaction(); body only on include_raw)
gold_graph.py     load_gold_graph(settings) -> GoldGraph (graph.json cached by mtime); raises NotFoundError('gold_not_built')
queries.py        build_query(filters, doc_ids) / build_sort(page) / overview_aggs() (pure ES DSL builders) ; guard(index, fn) -> silver_not_built / silver_unavailable
```
API (`/tcs/risks`, `api/tcs_routes.py`; existing `/search /summary /graph /graph.html` unchanged):
| Route | Query params | Response model |
|---|---|---|
| `GET /documents` | `category`* `min_severity` `min_risk_score` `source`* `source_type`* `relevant=true\|false\|all` `published_from` `published_to` `cluster` `community` `q` `sort` `order` `limit<=100` `offset<=10000` (*repeatable) | `DocumentPage` |
| `GET /documents/{doc_id}` | `include_text` `include_raw` (both default false) | `DocumentDetail` |
| `GET /documents/{doc_id}/similar` | `k` 1..50 | `SimilarDocuments` |
| `GET /categories` | `relevant_only=true` | `CategoryBreakdown` |
| `GET /overview` | `relevant_only=true` `top_n` 1..20 | `RiskOverview` |
| `GET /clusters` | `kind=cluster\|community` | `list[ClusterSummary]` |
| `GET /clusters/{cluster_id}` | `kind` | `ClusterDetail` |

Errors (envelope `{"error": {code, message, details}}`): 404 `gold_not_built`, `document_not_found`, `cluster_not_found`,
`document_not_in_gold`, `silver_not_built`; 422 `validation_error`; 503 `silver_unavailable`, `vector_store_unavailable`, `bronze_unavailable`.

MCP (`mcp_server/server.py`; `search_tcs_risks`, `tcs_risk_summary` unchanged): `list_tcs_risks(category, min_severity, min_risk_score, source,
source_type, query, published_from, published_to, cluster, community, sort, order, limit=10, offset, include_irrelevant=False) -> DocumentPage`;
`get_tcs_risk_document(doc_id, include_text=False) -> DocumentDetail` (no raw body over MCP); `similar_tcs_risks(doc_id, k=5) -> SimilarDocuments`;
`tcs_risk_clusters(cluster_id=None, kind="cluster") -> dict` (list or cluster + members). Errors become `ToolError("<code>: <message>")`.

## Storage
- MySQL `crawl_runs(id, started_at, finished_at, params JSON, stats JSON)`;
  `bronze_documents(id, run_id FK, url TEXT, url_sha256 CHAR(64) UNIQUE, canonical_url TEXT, parent_url TEXT NULL,
  depth TINYINT, source VARCHAR(32), source_type ENUM('news','community','corporate','other'), query VARCHAR(255),
  title TEXT NULL, raw_content MEDIUMTEXT, content_type VARCHAR(64), http_status SMALLINT, published_at DATETIME NULL,
  fetched_at DATETIME, content_sha256 CHAR(64), silver_processed_at DATETIME NULL)`.
- ES `silver-tcs-risk` (mapping in silver/store.py). ES `gold-tcs-risk-{docs|chunks}--{model_slug}` (prefix `GOLD_INDEX_PREFIX`): `dynamic: false`, 1 shard / 0 replicas,
  `embedding` = `dense_vector{dims, index: true, similarity: cosine, index_options: hnsw}` (ES 8.15 caps indexed
  dense_vector at 4096 dims and rejects zero-magnitude vectors under cosine), keyword `risk_categories/source/source_type/domain/url`,
  numeric `doc_id/chunk_index/cluster/community/risk_score/sentiment`, date `published_at`, text `title/text`.
  `_id` = `doc_id` (docs) / `doc_id:chunk_index` (chunks); bulk upsert then `indices.refresh`. ES `knn` score is
  `(1+cos)/2`, mapped back to cosine in `GoldHit.score`. Qdrant was dropped (no dual backend).
