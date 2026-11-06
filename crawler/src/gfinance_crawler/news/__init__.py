"""TCS risk news/community crawler (spec 005, bronze layer). Never imports ``tcs_bot``.

Public surface (contract: ``specs/005-tcs-risk-medallion/plan.md``)::

    sources.py  Source protocol + GoogleNewsRSS, BingNewsRSS, GdeltDoc, HackerNews, RedditSearch,
                LocalInbox
    fetch.py    PoliteFetcher(...).fetch(url) -> FetchResult      (never raises)
    links.py    canonical_url, extract_links, is_tcs_relevant
    bfs.py      RiskCrawler(...).crawl(queries) -> Iterator[RawDocument]; .stats
    queries.py  TCS_RISK_QUERIES, COMMUNITY_SUBREDDITS, DEFAULT_ALLOWED_DOMAINS
"""

from gfinance_crawler.news.models import FetchResult, RawDocument

__all__ = ["FetchResult", "RawDocument"]
