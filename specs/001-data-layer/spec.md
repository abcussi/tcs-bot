# 001 — Data layer (MySQL + Redis cache + Elasticsearch)

## Why
Risk reports need a durable system of record (vendors, documents, chunks, reports), fast reads
(cache) and searchable market/document data (Elasticsearch).

## Acceptance criteria
- AC-001.1 A MySQL connection pool (mysql-connector `MySQLConnectionPool`) is created once per process and reused.
- AC-001.2 `transaction()` commits on success and rolls back on any exception; connection always returned to pool.
- AC-001.3 `migrate()` creates all tables inside one logical transaction tracked in `schema_migrations`;
  because MySQL DDL auto-commits, a failure triggers compensating `DROP TABLE` of tables created in this run.
- AC-001.4 `migrate()` is idempotent (second run is a no-op).
- AC-001.5 `seed()` loads `SEED_FILE` (JSON). If the file is missing/unreadable it falls back to the
  built-in synthetic dataset and logs a warning. Seeding is idempotent (upsert by vendor `external_id`).
- AC-001.6 Two-tier cache: L1 in-process TTL+LRU, L2 Redis. Redis outage never raises to callers.
- AC-001.7 Elasticsearch indices `market-quotes` and `vendor-chunks` are created with explicit mappings if absent.
