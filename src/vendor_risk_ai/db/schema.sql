-- Vendor Risk AI schema (specs/001-data-layer/plan.md). InnoDB, utf8mb4.
-- Statements are executed one by one by vendor_risk_ai.db.migrate in file order (FK order).
-- Every statement MUST be `CREATE TABLE IF NOT EXISTS <name>` and end with `;`.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(64) NOT NULL PRIMARY KEY,
    applied_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS vendors (
    id           BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    external_id  VARCHAR(64)  NOT NULL,
    name         VARCHAR(255) NOT NULL,
    sector       VARCHAR(128) NOT NULL DEFAULT 'unknown',
    country      VARCHAR(64)  NOT NULL DEFAULT 'unknown',
    ticker       VARCHAR(32)  NULL,
    criticality  ENUM('low','medium','high','critical') NOT NULL DEFAULT 'medium',
    services     TEXT         NULL,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_vendors_external_id (external_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS vendor_documents (
    id              BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    vendor_id       BIGINT       NOT NULL,
    doc_type        ENUM('financial_report','news','regulatory_filing','other') NOT NULL DEFAULT 'other',
    title           VARCHAR(255) NOT NULL,
    source          VARCHAR(255) NOT NULL DEFAULT 'manual',
    published_at    DATE         NULL,
    content         MEDIUMTEXT   NOT NULL,
    content_sha256  CHAR(64)     NOT NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_vendor_documents_sha (vendor_id, content_sha256),
    CONSTRAINT fk_vendor_documents_vendor FOREIGN KEY (vendor_id)
        REFERENCES vendors (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_chunks (
    id               BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
    document_id      BIGINT      NOT NULL,
    chunk_index      INT         NOT NULL,
    text             TEXT        NOT NULL,
    embedding        JSON        NULL,
    embedding_model  VARCHAR(64) NULL,
    UNIQUE KEY uq_document_chunks_idx (document_id, chunk_index),
    CONSTRAINT fk_document_chunks_document FOREIGN KEY (document_id)
        REFERENCES vendor_documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS risk_reports (
    id             BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
    vendor_id      BIGINT        NOT NULL,
    overall_score  DECIMAL(5,2)  NOT NULL,
    risk_level     VARCHAR(16)   NOT NULL,
    report_json    JSON          NOT NULL,
    model          VARCHAR(64)   NOT NULL,
    created_at     DATETIME(6)   NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY ix_risk_reports_vendor_created (vendor_id, created_at),
    CONSTRAINT fk_risk_reports_vendor FOREIGN KEY (vendor_id)
        REFERENCES vendors (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
    action      VARCHAR(64) NOT NULL,
    entity      VARCHAR(64) NOT NULL,
    entity_id   VARCHAR(64) NULL,
    payload     JSON        NULL,
    created_at  DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY ix_audit_log_entity (entity, entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
