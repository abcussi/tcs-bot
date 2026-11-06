# Bronze inbox (offline / manual ingestion)

Drop documents here to ingest them without network access (spec 005, AC-005.1). Every
`tcs-bot medallion bronze` run reads this folder through the `LocalInbox` source and stores each
file as a depth-0 bronze document (`source = local_inbox`); files already ingested are skipped
(dedupe on URL).

- Supported: `*.html`, `*.htm`, `*.txt`, `*.md`, `*.json`, `*.pdf` (text extracted with pypdf;
  scanned PDFs without a text layer are skipped).
- `*.json` may carry metadata: `{"title": ..., "url": "https://...", "published_at": "ISO-8601", ...}`.
- Inbox documents are trusted: they are not relevance-filtered at bronze (silver still scores them).
- This README and dotfiles are ignored. Only synthetic or properly licensed material belongs in
  the repository.
