-- Migration 002: TCS risk medallion bronze layer (specs/005-tcs-risk-medallion/plan.md).
-- InnoDB, utf8mb4. Applied by tcs_bot.db.migrate after 001 (schema.sql), statements in file order.
-- Every statement MUST be `CREATE TABLE IF NOT EXISTS <name>` and end with `;`.

CREATE TABLE IF NOT EXISTS crawl_runs (
    id           BIGINT   NOT NULL AUTO_INCREMENT PRIMARY KEY,
    started_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at  DATETIME NULL,
    params       JSON     NULL,
    stats        JSON     NULL,
    KEY ix_crawl_runs_started (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS bronze_documents (
    id                   BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id               BIGINT        NULL,
    url                  TEXT          NOT NULL,
    url_sha256           CHAR(64)      NOT NULL,
    canonical_url        TEXT          NOT NULL,
    parent_url           TEXT          NULL,
    depth                TINYINT       NOT NULL DEFAULT 0,
    source               VARCHAR(32)   NOT NULL,
    source_type          ENUM('news','community','corporate','other') NOT NULL DEFAULT 'news',
    query                VARCHAR(255)  NOT NULL DEFAULT '',
    title                TEXT          NULL,
    raw_content          MEDIUMTEXT    NOT NULL,
    content_type         VARCHAR(64)   NOT NULL DEFAULT 'text/html',
    http_status          SMALLINT      NOT NULL DEFAULT 200,
    published_at         DATETIME      NULL,
    fetched_at           DATETIME      NOT NULL,
    content_sha256       CHAR(64)      NOT NULL,
    silver_processed_at  DATETIME      NULL,
    UNIQUE KEY uq_bronze_documents_url_sha (url_sha256),
    KEY ix_bronze_documents_silver (silver_processed_at, id),
    KEY ix_bronze_documents_run (run_id),
    KEY ix_bronze_documents_source (source),
    CONSTRAINT fk_bronze_documents_run FOREIGN KEY (run_id)
        REFERENCES crawl_runs (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
