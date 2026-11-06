# 005 — Tasks

## SILVER (`src/tcs_bot/medallion/silver/`, owner: agent SILVER)
- [x] `extract.py` — `extract_main_text(raw, content_type, url) -> (text, title|None)`: HTML via
      trafilatura (favor_recall, no comments) + bs4 fallback (script/style/nav/footer/header/aside
      dropped) and title from metadata/og:title/<title>; Reddit thread/listing/t3 JSON (no authors,
      `[deleted]` skipped, max 40 comments); HN Algolia item/hit (recursive, HTML stripped);
      GDELT artlist item/response; RSS/Atom items; text/markdown passthrough; content-type sniffing;
      never raises. Normalized with `ingestion.normalize.normalize_text`. (AC-005.5)
- [x] `classify.py` — 65 regex rules over all 10 `RiskCategory` values with severities;
      sentence-level evidence (<= 300 chars), TCS-sentence weighting (non-TCS x0.7), title boost
      (x1.5), negation ("denied", "no evidence of", "dismissed the case", "cleared of", ...) lowers
      severity one level; dedupe by (category, keyword); risk score `100*(1-prod(1-p/100))`
      (critical 40 / high 25 / medium 12 / low 5 points). `tcs_relevance()` 0..1 with penalties for
      Tax Collected at Source / LRS / remittance contexts and Tata-group-only stories. (AC-005.5/.6)
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
      language heuristic, `relevant = relevance >= threshold and word_count >= 30`);
      `run_silver(limit=None, repo=None, store=None)` batches of 50, bad docs skipped, bronze rows
      marked processed only after a successful ES upsert.
- [x] Synthetic fixtures `tests/fixtures/silver/` (every category, Tax-Collected-at-Source and
      Tata-Motors false positives, Reddit/HN/GDELT/RSS payloads), unit tests
      `tests/unit/test_silver_*.py`, integration `tests/integration/test_silver_es.py`
      (throwaway `silver-test-<uuid>` index, deleted after).
- Note: `llm_extract(doc, text, settings, base=None, chain=None)` takes the extracted text
  explicitly (plan listed `llm_extract(doc, settings)`).
