# 005 — Plan / contracts

## Ownership
| Area | Path | Owner |
|---|---|---|
| Sources + BFS crawler | `crawler/src/gfinance_crawler/news/` | agent BRONZE |
| Bronze persistence + migration 002 | `src/tcs_bot/medallion/bronze.py`, `src/tcs_bot/db/schema_002_medallion.sql`, `db/migrate.py` | agent BRONZE |
| Silver | `src/tcs_bot/medallion/silver/` | agent SILVER |
| Gold + analytics + viz + API/MCP | `src/tcs_bot/medallion/gold/`, `api/tcs_routes.py`, `mcp_server/` tool | agent GOLD |
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
gold/        embedder.py  class LsaEmbedder(dim=256).fit(texts); get_gold_embedder(settings, corpus) -> Embedder-like
             vector_store.py class GoldVectorStore(url, model): ensure(dim); upsert_docs/upsert_chunks; search(vector, k, filters) -> list[GoldHit]
             analytics.py cosine_matrix(X); knn_graph(X, ids, k, threshold) -> nx.Graph; clusters(...); project_2d(X)
             viz.py       def render_html(graph_json: dict, path: Path) -> Path
             pipeline.py  def run_gold() -> dict   # writes data/gold/*
```
## Storage
- MySQL `crawl_runs(id, started_at, finished_at, params JSON, stats JSON)`;
  `bronze_documents(id, run_id FK, url TEXT, url_sha256 CHAR(64) UNIQUE, canonical_url TEXT, parent_url TEXT NULL,
  depth TINYINT, source VARCHAR(32), source_type ENUM('news','community','corporate','other'), query VARCHAR(255),
  title TEXT NULL, raw_content MEDIUMTEXT, content_type VARCHAR(64), http_status SMALLINT, published_at DATETIME NULL,
  fetched_at DATETIME, content_sha256 CHAR(64), silver_processed_at DATETIME NULL)`.
- ES `silver-tcs-risk` (mapping in silver/store.py). Qdrant `gold_tcs_risk_{docs|chunks}__{model_slug}` cosine.
