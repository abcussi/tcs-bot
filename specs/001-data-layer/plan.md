# 001 — Plan / contracts

## Tables (db/schema.sql) — InnoDB, utf8mb4
- `schema_migrations(version VARCHAR(64) PK, applied_at DATETIME)`
- `vendors(id BIGINT PK AI, external_id VARCHAR(64) UNIQUE, name, sector, country, ticker NULL, criticality ENUM('low','medium','high','critical'), services TEXT, created_at, updated_at)`
- `vendor_documents(id PK AI, vendor_id FK, doc_type ENUM('financial_report','news','regulatory_filing','other'), title, source, published_at DATE NULL, content MEDIUMTEXT, content_sha256 CHAR(64), created_at, UNIQUE(vendor_id, content_sha256))`
- `document_chunks(id PK AI, document_id FK ON DELETE CASCADE, chunk_index INT, text TEXT, embedding JSON NULL, embedding_model VARCHAR(64) NULL, UNIQUE(document_id, chunk_index))`
- `risk_reports(id PK AI, vendor_id FK, overall_score DECIMAL(5,2), risk_level VARCHAR(16), report_json JSON, model VARCHAR(64), created_at)`
- `audit_log(id PK AI, action VARCHAR(64), entity VARCHAR(64), entity_id VARCHAR(64), payload JSON, created_at)`

## Public API
```python
# vendor_risk_ai/db/mysql.py
def get_pool() -> MySQLConnectionPool            # lazily created singleton from settings
def reset_pool() -> None                           # tests
@contextmanager
def transaction() -> Iterator[MySQLCursorDict]    # dictionary cursor; commit/rollback; returns conn
def ping() -> bool

# vendor_risk_ai/db/migrate.py
def migrate() -> list[str]                         # returns tables created this run

# vendor_risk_ai/db/seed.py
BUILTIN_SEED: dict                                 # synthetic fallback
def load_seed_data(path: str | None = None) -> dict
def seed(path: str | None = None) -> dict[str, int]   # {"vendors": n, "documents": m}

# vendor_risk_ai/db/repositories.py  (all use transaction())
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
class ReportRepository:
    def save(self, vendor_id: int, report: RiskReport, model: str) -> int
    def latest(self, vendor_id: int) -> StoredReport | None

# vendor_risk_ai/cache/cache.py
class TwoTierCache:
    def __init__(self, redis_url: str | None, ttl: int, maxsize: int)
    def get(self, key: str) -> Any | None          # JSON-serializable values
    def set(self, key: str, value: Any, ttl: int | None = None) -> None
    def delete(self, key: str) -> None
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int | None = None) -> Any
def get_cache() -> TwoTierCache

# vendor_risk_ai/search/elastic.py
def get_es() -> Elasticsearch
def ensure_indices() -> None
def index_chunks(vendor_id: int, chunks: list[ChunkRecord]) -> int
def latest_quote(ticker: str) -> dict | None       # newest doc from market-quotes by ticker
def ping() -> bool
```
Domain models (`Vendor`, `VendorCreate`, `DocumentCreate`, `VendorDocument`, `ChunkRecord`,
`StoredReport`) live in `schemas/domain.py`.
`ChunkRecord(document_id:int|None, chunk_index:int, text:str, embedding:list[float]|None, embedding_model:str|None, doc_type:str|None, title:str|None)`.
