# 005 — Tasks

## BRONZE (`crawler/src/gfinance_crawler/news/`, `src/tcs_bot/medallion/bronze.py`, migration 002; owner: agent BRONZE)
- [x] `news/models.py` — `RawDocument` (BronzeDocument fields minus `id`/`run_id`, plus
      crawler-internal `follow_url`, excluded from `model_dump()`), `FetchResult(url, final_url,
      status, content_type, text, error, blocked)` + `.ok`. (AC-005.1)
- [x] `news/queries.py` — 22 `TCS_RISK_QUERIES` (ethics, compliance, lawsuit, fine, layoffs, data
      breach, cyber attack, H-1B visa fraud, discrimination, investigation, SEBI, controversy, outage,
      client loss, whistleblower, bribery, tax dispute, ESG, union, trade secrets Epic/DXC/CSC),
      `COMMUNITY_SUBREDDITS`, `DEFAULT_ALLOWED_DOMAINS` (suffix-matched) + `domain_allowed()`. (AC-005.1/.3)
- [x] `news/sources.py` — `Source` protocol; `GoogleNewsRSS`, `BingNewsRSS` (apiclick unwrapped),
      `GdeltDoc`, `HackerNews` (Algolia; depth 1 = items/{story_id}), `RedditSearch` (site-wide +
      per-subreddit; depth 1 = permalink.json), `LocalInbox(path)` (html/htm/txt/md/json/pdf via
      pypdf; query-independent, not relevance-filtered). Blocked/failing endpoints raise
      `SearchError(blocked=...)`. (AC-005.1)
- [x] `news/fetch.py` — `PoliteFetcher`: robots.txt per origin through the same client (failure =>
      allow), per-domain delay, tenacity retries on transport/5xx/429 (not on proxy CONNECT
      refusals), 401/403/451/robots/proxy => `blocked`, PDFs -> text, binary refused, never raises. (AC-005.2)
- [x] `news/links.py` — `canonical_url`, `extract_links`, `extract_links_json`, `is_tcs_relevant`
      (rejects Tax-Collected-at-Source / LRS contexts). (AC-005.2/.3)
- [x] `news/bfs.py` — `RiskCrawler` breadth-first, hard caps depth 2 / 500 docs, canonical-URL
      dedupe of fetches and stored docs, relevance at every depth, depth-2 links only when
      anchor/URL is TCS-relevant and domain allow-listed, blocked source disabled without stopping
      others, `stats` (CrawlStats keys + `links_skipped`, `errors`). (AC-005.2/.3)
- [x] `db/schema_002_medallion.sql` (`crawl_runs`, `bronze_documents`; unique `url_sha256`,
      indexes on silver watermark, run_id, source; FK run_id ON DELETE SET NULL) and versioned
      `db/migrate.py` (`MIGRATIONS`, `statements_for`, `all_statements`, `LATEST_VERSION`;
      per-version compensating rollback; idempotent). (AC-005.4)
- [x] `medallion/bronze.py` — `BronzeRepository` (start_run/finish_run/get_run, save_batch ->
      new-row count, keyset-paginated `iter_unprocessed`, `mark_processed`, `count`), `to_bronze`,
      `default_sources`, `run_bronze` (batches of 25, run recorded even on failure). (AC-005.4)
- [x] `data/bronze/inbox/` (.gitkeep + README) for offline ingestion.
- [x] Tests: `crawler/tests/test_news_{links,fetch,sources,bfs}.py` + synthetic fixtures in
      `crawler/tests/fixtures/news/`; `tests/unit/test_bronze_layer.py`; migration tests updated
      for 001+002; `tests/integration/test_bronze_mysql.py` (throwaway `vendor_risk_bronzetest` DB).
- Notes / deviations: `run_bronze` adds keyword-only `repo`, `allowed_domains`,
  `per_query_limit`, `inbox_dir` and defaults caps to settings (None). `CrawlStats.stored` = NEW
  bronze rows; rows already in bronze count as `duplicates`. Depth-0 items are also
  relevance-filtered (except LocalInbox). GDELT depth-0 url is `...doc/doc?format=json&mode=artlist&item=<article url>`
  (GDELT has no item permalink). if reddit.com robots.txt disallows generic crawlers (it did as of
  2024), RedditSearch is reported as blocked in production; robots is respected, not bypassed.

- Real-internet probe (2026-11-06, local PC, robots respected): Google News RSS (`/rss/search`
  disallowed) and Reddit are blocked by robots.txt, so `default_sources` still lists them but every
  run reports them as `blocked`. GDELT intermittently answers HTTP 429 (not retried; source disabled
  for the run). Bing News RSS, Hacker News work. moneycontrol, business-standard, news18, firstpost
  answer 403 to the bot UA; the depth-0 Bing snippet is kept. HN relevance is loose: a thread counts
  if any comment mentions TCS.

## SILVER (`src/tcs_bot/medallion/silver/`, owner: agent SILVER)
- [x] `extract.py` — `extract_main_text(raw, content_type, url) -> (text, title|None)`: HTML via
      trafilatura (favor_recall, no comments) + bs4 fallback (script/style/nav/footer/header/aside
      dropped) and title from metadata/og:title/<title>; Reddit thread/listing/t3 JSON (no authors,
      `[deleted]` skipped, max 40 comments); HN Algolia item/hit (recursive, HTML stripped);
      GDELT artlist item/response; RSS/Atom items; text/markdown passthrough; content-type sniffing;
      never raises. Normalized with `ingestion.normalize.normalize_text`. (AC-005.5)
- [x] `classify.py` — 65 regex rules over all 10 `RiskCategory` values with severities;
      sentence-level evidence (<= 300 chars), TCS-proximity weighting (sentence naming TCS x1.0,
      1/2 sentences away x0.8/x0.6, further x0.25; total far-evidence contribution capped at 18
      points; long texts damped by `1/(1+0.12*log2(words/300))`), title boost (x1.5), negation ("denied", "no evidence of", "dismissed the case", "cleared of", ...) lowers
      severity one level; dedupe by (category, keyword); risk score `100*(1-prod(1-p/100))`
      (critical 40 / high 25 / medium 12 / low 5 points). `tcs_relevance()` 0..1 with penalties for
      Tax Collected at Source / LRS / remittance contexts and Tata-group-only stories;
      "TCS (Tata Consultancy Services)" counts once; when the title does not name TCS the score is
      x0.6 (recovered as TCS density rises from 1 to 3 mentions/100 words) and x0.6 more for a
      "Tata ..." headline (passing mentions in threads about something else). (AC-005.5/.6)
- [x] `entities.py` — regulators gazetteer (canonical names), money (₹/Rs/INR/$/USD/€/£ +
      crore/lakh/million/billion), places gazetteer + aliases, organizations (known list + legal
      suffix pattern), people (known execs + role/honorific patterns); deduped, max 20 each.
- [x] `sentiment.py` — finance/risk lexicon, 3-token negation window, length-damped, VADER-style
      normalization to [-1, 1].
- [x] `llm.py` — `prompt | ChatOpenAI(openai_chat_model, T=0).with_structured_output(LlmSilverOutput,
      method="function_calling")`; merged with the rules record (grounded LLM signals unioned,
      relevance/sentiment averaged, LLM summary, entity union, `extractor` = model id); any error ->
      `None` (rules fallback). Only called for docs with >= 30 words and rules relevance >=
      threshold/2 (cost guard).
- [x] `store.py` — `SilverStore(es=None, index=None)`: explicit mapping (nested `risk_signals`,
      english analyzer on `clean_text`/`summary`, `dynamic: false`), 1 shard / 0 replicas;
      `ensure_index`, `upsert(records, refresh=False) -> int` (`_id = doc_id`, per-doc failures in
      `last_failed_ids`), `get`, `iter_relevant(min_relevance=None)`, `count`, `stats`, `delete_index`.
- [x] `pipeline.py` — `build_silver_record(doc, settings=None)` (domain, 2-3 sentence summary,
      language heuristic, `relevant = is_relevant(...)` = relevance >= threshold and (word_count >= 30
      or the title names TCS); risk of docs below the threshold is scaled by relevance/threshold via
      `attribute_risk`);
      `run_silver(limit=None, repo=None, store=None)` batches of 50, bad docs skipped, bronze rows
      marked processed only after a successful ES upsert.
- [x] Synthetic fixtures `tests/fixtures/silver/` (every category, Tax-Collected-at-Source and
      Tata-Motors false positives, Reddit/HN/GDELT/RSS payloads), unit tests
      `tests/unit/test_silver_*.py`, integration `tests/integration/test_silver_es.py`
      (throwaway `silver-test-<uuid>` index, deleted after).
- Note: `llm_extract(doc, text, settings, base=None, chain=None)` takes the extracted text
  explicitly (plan listed `llm_extract(doc, settings)`).

## Verification log
- SILVER verified by lead (lint clean; 68 unit + 2 integration tests pass against live ES).
  Not yet run against real bronze rows: `bronze_documents` needs migration 002 applied.
