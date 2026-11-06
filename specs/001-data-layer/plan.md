# 001 — Plan / contracts

## Tables (db/schema.sql) — InnoDB, utf8mb4
- `schema_migrations(version VARCHAR(64) PK, applied_at DATETIME)`
- `vendors(id BIGINT PK AI, external_id VARCHAR(64) UNIQUE, name, sector, country, ticker NULL, criticality ENUM('low','medium','high','critical'), services TEXT, created_at, updated_at)`
- `vendor_documents(id PK AI, vendor_id FK ON DELETE CASCADE, doc_type ENUM('financial_report','news','regulatory_filing','other'), title, source, published_at DATE NULL, content MEDIUMTEXT, content_sha256 CHAR(64), created_at, UNIQUE(vendor_id, content_sha256))`
- `document_chunks(id PK AI, document_id FK ON DELETE CASCADE, chunk_index INT, text TEXT, embedding JSON NULL, embedding_model VARCHAR(64) NULL, UNIQUE(document_id, chunk_index))`
- `risk_reports(id PK AI, vendor_id FK ON DELETE CASCADE, overall_score DECIMAL(5,2), risk_level VARCHAR(16), report_json JSON, model VARCHAR(64), created_at)`
- `audit_log(id PK AI, action VARCHAR(64), entity VARCHAR(64), entity_id VARCHAR(64), payload JSON, created_at)`

## Public API
```python
# tcs_bot/db/mysql.py
def get_pool() -> MySQLConnectionPool            # lazily created singleton from settings
def reset_pool() -> None                           # tests
@contextmanager
def transaction() -> Iterator[MySQLCursorDict]    # dictionary cursor; commit/rollback; returns conn
def ping() -> bool

# tcs_bot/db/migrate.py
def migrate() -> list[str]                         # returns tables created this run

# tcs_bot/db/seed.py
BUILTIN_SEED: dict                                 # synthetic fallback
def load_seed_data(path: str | None = None) -> dict
def seed(path: str | None = None) -> dict[str, int]   # {"vendors": n, "documents": m}

# tcs_bot/db/repositories.py  (all use transaction())
class VendorRepository:
    def list(self, limit: int = 100, offset: int = 0) -> list[Vendor]
    def get(self, vendor_id: int) -> Vendor | None
    def get_by_external_id(self, external_id: str) -> Vendor | None
    def upsert(self, v: VendorCreate) -> int
class DocumentRepository:
    def add(self, vendor_id: int, doc: DocumentCreate) -> int    # dedupe by sha256
    def list_for_vendor(self, vendor_id: int) -> list[VendorDocument]
    def save_chunks(self, document_id: int, chunks: list[ChunkRecord]) -> int
    def chunks_for_vendor(self, vendor_id: int) -> list[ChunkRecord]
    def chunk_ids_for_vendor(self, vendor_id: int) -> list[tuple[int, int]]   # (document_id, chunk_index), no embeddings
class ReportRepository:
    def save(self, vendor_id: int, report: RiskReport, model: str) -> int
    def latest(self, vendor_id: int) -> StoredReport | None

# tcs_bot/cache/cache.py
class TwoTierCache:
    def __init__(self, redis_url: str | None, ttl: int, maxsize: int)
    def get(self, key: str) -> Any | None          # JSON-serializable values
    def set(self, key: str, value: Any, ttl: int | None = None) -> None
    def delete(self, key: str) -> None
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int | None = None) -> Any
def get_cache() -> TwoTierCache

# tcs_bot/search/elastic.py
def get_es() -> Elasticsearch
def ensure_indices() -> None
def index_chunks(vendor_id: int, chunks: list[ChunkRecord]) -> int
def latest_quote(ticker: str, *, raise_on_error: bool = False) -> dict | None  # newest doc by ticker
def latest_quote_strict(ticker: str) -> dict | None  # raises SearchUnavailable when ES errors
class SearchUnavailable(RuntimeError)
def ping() -> bool
```
Domain models (`Vendor`, `VendorCreate`, `DocumentCreate`, `VendorDocument`, `ChunkRecord`;
`StoredReport` is in `schemas/report.py`) live in `schemas/domain.py`.
`ChunkRecord(document_id:int|None, chunk_index:int, text:str, embedding:list[float]|None, embedding_model:str|None, doc_type:str|None, title:str|None)`.

## Implementation notes (recorded deviations, architect review 2026-11-06)
- Supersets of the contract (compatible): `TwoTierCache(..., *, key_prefix="vra:", redis_client=None)`,
  `ping()`, `clear_local()`; `reset_cache()`; `index_chunks(vendor_id, chunks, *, refresh=False)`;
  `reset_es()`, `chunk_doc_id()`; cursor-level helpers `upsert_vendor_cur`, `add_document_cur`,
  `write_audit` in `repositories.py` for multi-write transactions (used by `seed`).
- `transaction()` waits up to `POOL_ACQUIRE_TIMEOUT_S`=10 s for a pooled connection (mysql-connector
  raises `PoolError` immediately when exhausted; uvicorn's threadpool is 40 > pool 10). Never nest
  `transaction()` calls in one thread (each holds a connection) — compose with the `*_cur` helpers.
- `market-quotes` is created by whichever side runs first (`ensure_indices()` here or
  `gfinance_crawler.sink.ElasticSink.ensure_index()`). Both MUST use the identical mapping
  (`QUOTES_MAPPING`, currently equal) and identical index settings (1 shard, 0 replicas). The
  crawler cannot import this package, so parity is enforced by
  `tests/unit/test_quotes_mapping_parity.py` (imports both mappings and settings).
- `vendor-chunks.embedding` is `dense_vector` dims 1536 / cosine: vectors of other lengths are
  dropped (text kept); all-zero vectors are rejected by ES cosine (partial bulk error, logged).
