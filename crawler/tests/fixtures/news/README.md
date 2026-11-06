SYNTHETIC FIXTURES for the spec-005 news crawler tests. Every headline, article, post, comment,
id and URL path in this folder is invented for offline tests; none of it is real news or a real
statement about any company or person.

Every file here carries a "SYNTHETIC FIXTURE" marker (comment, `_comment`/`_synthetic` field or first line); they mirror the *structure* of real responses only (Bing RSS, HN Algolia hits with `_highlightResult`, GDELT plain-text throttling, robots.txt of a news aggregator), never real text.
