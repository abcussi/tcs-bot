"""Gold graph HTML (AC-005.8) — offline structural tests, no browser.

SYNTHETIC FIXTURE: every node below is invented; nothing describes a real event.
The interactive behaviour (slider, colour toggle, click-to-inspect, canvas performance) was
checked in headless Chromium by hand; these tests guard the static contract of the page.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tcs_bot.medallion.gold.viz import _TEMPLATE, embed_json, render_html

SYNTHETIC_FIXTURE = "SYNTHETIC FIXTURE"


def _node(i: int, **over) -> dict:
    n = {
        "id": i,
        "title": f"[SYNTHETIC FIXTURE] doc {i}",
        "url": f"https://synthetic.example/{i}",
        "domain": "synthetic.example",
        "source": "synthetic_news",
        "source_type": "news",
        "published_at": "2026-01-02T00:00:00+00:00",
        "risk_score": 10.0 * (i % 10),
        "sentiment": -0.2,
        "relevance": 0.8,
        "risk_categories": ["ethics"],
        "dominant_category": "ethics",
        "cluster": i % 3,
        "community": i % 2,
        "x": 0.1 * i,
        "y": -0.05 * i,
        "summary": "invented summary",
        "signals": [{"category": "ethics", "severity": "low", "keyword": "k", "evidence": "e"}],
        "entities": {"organizations": [], "regulators": [], "places": []},
        "neighbors": [],
    }
    n.update(over)
    return n


def _graph(n_nodes: int) -> dict:
    nodes = [_node(i) for i in range(n_nodes)]
    links = [
        {"source": i, "target": i + 1, "cosine": 0.5, "mutual": True} for i in range(n_nodes - 1)
    ]
    return {
        "meta": {"n_docs": n_nodes, "data_source": SYNTHETIC_FIXTURE, "synthetic": True},
        "nodes": nodes,
        "links": links,
        "clusters": [
            {"cluster": c, "label": f"c{c}", "size": 1, "dominant_categories": {"ethics": 1}}
            for c in range(3)
        ],
        "communities": [],
        "stats": {},
    }


def _page(tmp_path: Path, data: dict) -> str:
    return render_html(data, tmp_path / "g.html").read_text(encoding="utf-8")


def _embedded(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="gold-data">(.*?)</script>', html, re.S)
    assert m
    return json.loads(m.group(1))


def _app_js(html: str) -> str:
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    return scripts[0]


def test_html_has_no_network_dependencies(tmp_path: Path) -> None:
    """AC-005.8: the page must open offline: no external script/link/img/css/import/fetch."""
    html = _page(tmp_path, _graph(5))
    page = html.replace(embed_json(_graph(5)), "")
    assert not re.search(r"<(script|link|img|iframe)\b[^>]*\b(src|href)\s*=", page)
    assert not re.search(r"@import|url\(\s*['\"]?(https?:)?//", page)
    assert not re.search(r"\b(fetch|XMLHttpRequest|WebSocket|sendBeacon|importScripts)\b", page)
    assert "cdn." not in page


@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_graphs_render_and_embed(tmp_path: Path, n: int) -> None:
    """Empty and single-node graphs render; the empty state message is present."""
    html = _page(tmp_path, _graph(n))
    assert len(_embedded(html)["nodes"]) == n
    assert "No relevant silver documents yet" in html


def test_graph_with_missing_optional_fields(tmp_path: Path) -> None:
    """Nodes/meta/stats lacking optional keys still render (the JS tolerates them)."""
    data = {"nodes": [{"id": 1}, {"id": 2, "title": None, "signals": None}], "links": []}
    html = _page(tmp_path, data)
    assert _embedded(html)["nodes"][0] == {"id": 1}
    assert render_html({}, tmp_path / "empty.html").is_file()


def test_embed_json_cannot_break_out_of_script() -> None:
    nasty = {"t": "</script><!--<script>alert(1)</script>  &", "n": [1.5, None, True]}
    out = embed_json(nasty)
    for bad in ("<", ">", "&", " ", " "):
        assert bad not in out
    assert json.loads(out) == nasty


def test_script_breakout_attempt_in_node_title(tmp_path: Path) -> None:
    evil = '</script><img src=x onerror="alert(1)">'
    html = _page(tmp_path, {"nodes": [_node(1, title=evil)], "links": []})
    assert "<img src=x" not in html
    assert html.count("<script") == 2 and html.count("</script>") == 2
    assert _embedded(html)["nodes"][0]["title"] == evil


def test_page_uses_textcontent_not_innerhtml() -> None:
    """Crawled strings are only ever inserted as text."""
    html = _TEMPLATE
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html
    assert "document.write" not in html


def test_palette_has_twelve_tokens_and_other_is_defined() -> None:
    """10 risk categories need 10 distinct colours; ``tokens.other`` must exist (regression:
    it was read but never assigned, so overflow groups drew with the previous fill colour)."""
    html = _TEMPLATE
    for i in range(1, 13):
        assert html.count(f"--s{i}:") == 3  # light + dark(media) + dark(data-theme)
    assert "tokens.other = tokens['--other']" in html
    assert "MAX_COLORS = 12" in html


def test_interactive_controls_present(tmp_path: Path) -> None:
    html = _page(tmp_path, _graph(3))
    for needle in (
        'id="min-cos"',  # cosine slider
        'data-v="cluster"',
        'data-v="community"',
        'data-v="category"',  # colouring toggle
        'id="legend"',
        'id="side"',  # inspect panel
        'id="graph"',  # canvas
        "devicePixelRatio",  # high-DPI
        "prefers-color-scheme",
    ):
        assert needle in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_inline_script_is_valid_javascript(tmp_path: Path) -> None:
    js = tmp_path / "app.js"
    js.write_text(_app_js(_page(tmp_path, _graph(4))), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
