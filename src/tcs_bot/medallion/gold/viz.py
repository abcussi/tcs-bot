"""Self-contained interactive gold graph (AC-005.8): one HTML file, inline CSS + vanilla JS.

No external scripts, fonts or stylesheets, so the file opens offline (e-mail it, open it from a
share). The graph JSON is embedded in a ``<script type="application/json">`` block with ``<``,
``>`` and ``&`` escaped, and every crawled string is inserted with ``textContent``, never as HTML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_PLACEHOLDER = "__GOLD_DATA_JSON__"


def embed_json(data: Any) -> str:
    """JSON safe to inline inside a ``<script>`` element."""
    raw = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_html(graph_json: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.replace(DATA_PLACEHOLDER, embed_json(graph_json))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(path)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TCS Risk Graph</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --raised: #ffffff;
  --ink: #0b0b0b; --ink2: #52514e; --muted: #77756f;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --edge-rgb: 82,81,78; --wash: rgba(11,11,11,0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --s9: #8a5a44; --s10: #12a5b5; --s11: #8a8a1f; --s12: #5b6b8c;
  --other: #a9a7a0; --accent: #2a78d6;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --note-bg: #fff7e0; --note-ink: #5b4300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --raised: #222221;
    --ink: #f0efec; --ink2: #c3c2b7; --muted: #9a9890;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --edge-rgb: 195,194,183; --wash: rgba(255,255,255,0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
    --s9: #b57a5e; --s10: #2cc0d0; --s11: #b2b236; --s12: #8b9bc0;
    --other: #5f5e5a; --accent: #3987e5;
    --note-bg: #2a2412; --note-ink: #f1d488;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --raised: #222221;
  --ink: #f0efec; --ink2: #c3c2b7; --muted: #9a9890;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --edge-rgb: 195,194,183; --wash: rgba(255,255,255,0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  --s9: #b57a5e; --s10: #2cc0d0; --s11: #b2b236; --s12: #8b9bc0;
  --other: #5f5e5a; --accent: #3987e5;
  --note-bg: #2a2412; --note-ink: #f1d488;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
a { color: var(--accent); }
button, input, select { font: inherit; color: inherit; }
.wrap { max-width: 1680px; margin: 0 auto; padding: 16px 20px 40px; }
header.top { display: flex; flex-wrap: wrap; gap: 12px 28px; align-items: flex-end;
  justify-content: space-between; margin-bottom: 12px; }
header.top h1 { font-size: 20px; line-height: 1.2; margin: 0 0 4px; font-weight: 650; }
.sub { color: var(--ink2); font-size: 13px; }
.badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 4px; background: var(--critical); color: #fff;
  vertical-align: 3px; margin-left: 8px; }
.tiles { display: flex; flex-wrap: wrap; gap: 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 12px; min-width: 86px; }
.tile .v { font-size: 20px; font-weight: 650; line-height: 1.2; }
.tile .l { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.note { background: var(--note-bg); color: var(--note-ink); border-radius: 6px;
  padding: 6px 10px; font-size: 12.5px; margin: 0 0 12px; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; min-width: 0; }
.card h2 { font-size: 13px; font-weight: 650; margin: 0; padding: 10px 14px 6px;
  text-transform: uppercase; letter-spacing: .04em; color: var(--ink2); }
.card .body { padding: 4px 14px 14px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
  padding: 10px 12px; border-bottom: 1px solid var(--border); }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 7px; overflow: hidden; }
.seg button { border: 0; background: transparent; padding: 4px 10px; cursor: pointer; font-size: 13px; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.ctl { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ink2); }
.ctl input[type=range] { width: 130px; accent-color: var(--accent); }
.ctl output { font-variant-numeric: tabular-nums; color: var(--ink); min-width: 3.2em; }
.search { border: 1px solid var(--border); background: var(--raised); border-radius: 7px;
  padding: 4px 10px; flex: 1 1 140px; min-width: 120px; max-width: 280px; }
#edge-count { margin-left: auto; white-space: nowrap; }
.stage { position: relative; height: var(--stage-h, clamp(480px, calc(100vh - 220px), 900px)); }
.stage canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
.stage canvas.dragging { cursor: grabbing; }
.stage .empty { position: absolute; inset: 0; display: grid; place-items: center;
  color: var(--muted); text-align: center; padding: 20px; }
.legend { position: absolute; left: 10px; top: 10px; max-width: 300px; max-height: calc(100% - 50px); z-index: 2;
  overflow: auto; background: color-mix(in srgb, var(--surface) 88%, transparent);
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; font-size: 12px;
  backdrop-filter: blur(4px); }
.legend .row { display: flex; align-items: center; gap: 6px; padding: 2px 4px; border-radius: 4px;
  cursor: pointer; white-space: nowrap; }
.legend .row:hover { background: var(--wash); }
.legend .row.off { opacity: .35; }
.legend .row .lab { overflow: hidden; text-overflow: ellipsis; max-width: 220px; }
.legend .row .cnt { color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
.legend .ttl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
  padding: 0 4px 3px; }
.sw { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.sq { width: 10px; height: 10px; border-radius: 2px; flex: none; display: inline-block; }
.hint { position: absolute; right: 10px; bottom: 8px; font-size: 11.5px; color: var(--muted);
  background: color-mix(in srgb, var(--surface) 80%, transparent); padding: 2px 6px; border-radius: 4px; }
.tip { position: absolute; pointer-events: none; z-index: 5; max-width: 340px;
  background: var(--raised); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 6px 24px rgba(0,0,0,.18); padding: 8px 10px; font-size: 12.5px; display: none; }
.tip .t { font-weight: 650; margin-bottom: 3px; }
.tip .m { color: var(--ink2); }
.tip .r { font-size: 16px; font-weight: 700; }
aside.side { display: flex; flex-direction: column; min-height: 0; }
aside.side .body { overflow: auto; max-height: calc(var(--stage-h, clamp(480px, calc(100vh - 220px), 900px)) + 8px); }
.dtitle { font-size: 15px; font-weight: 650; line-height: 1.3; margin: 2px 0 4px; }
.dmeta { color: var(--ink2); font-size: 12.5px; margin-bottom: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.chip { font-size: 11.5px; border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px;
  background: var(--wash); display: inline-flex; gap: 5px; align-items: center; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 12.5px; margin: 8px 0; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
.sec { font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted);
  margin: 14px 0 6px; font-weight: 650; }
.sig { border-left: 3px solid var(--axis); padding: 2px 0 2px 8px; margin: 6px 0; font-size: 12.5px; }
.sev { font-size: 10.5px; font-weight: 700; text-transform: uppercase; padding: 0 5px; border-radius: 3px;
  color: #fff; margin-right: 4px; }
.sev.low { background: var(--good); }
.sev.medium { background: var(--warning); color: #0b0b0b; }
.sev.high { background: var(--serious); color: #0b0b0b; }
.sev.critical { background: var(--critical); }
.sig q { color: var(--ink2); display: block; margin-top: 2px; }
.nb { display: grid; grid-template-columns: 1fr 64px 50px; gap: 8px; align-items: center;
  padding: 3px 4px; border-radius: 4px; cursor: pointer; font-size: 12.5px; }
.nb:hover { background: var(--wash); }
.nb .tt { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nb .v { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink2); }
.bar { height: 6px; border-radius: 3px; background: var(--grid); overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; margin-top: 12px; }
.hbars .row { display: grid; grid-template-columns: 170px 1fr 40px; gap: 8px; align-items: center;
  padding: 3px 4px; border-radius: 4px; font-size: 12.5px; cursor: pointer; }
.hbars .row:hover { background: var(--wash); }
.hbars .row .lab { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hbars .row .v { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink2); }
.hbars .track { height: 12px; }
.hbars .track > i { display: block; height: 100%; background: var(--s1); border-radius: 0 4px 4px 0; min-width: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11.5px; padding: 4px 6px;
  border-bottom: 1px solid var(--grid); }
td { padding: 5px 6px; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--wash); }
.list { max-height: 520px; overflow: auto; }
.list .it { display: grid; grid-template-columns: 44px 1fr; gap: 8px; padding: 6px 4px;
  border-bottom: 1px solid var(--grid); cursor: pointer; font-size: 12.5px; }
.list .it:hover { background: var(--wash); }
.list .score { font-weight: 700; font-size: 15px; font-variant-numeric: tabular-nums; }
.list .it .tt { font-weight: 600; }
.list .it .mm { color: var(--muted); font-size: 12px; }
svg text { fill: var(--muted); font-size: 10.5px; font-family: inherit; }
svg .bar-rect { fill: var(--s1); }
svg .bar-rect:hover { opacity: .75; }
svg .thr { stroke: var(--ink); stroke-dasharray: 3 3; stroke-width: 1.5; }
svg .base { stroke: var(--axis); }
.muted { color: var(--muted); }
.chart-tip { position: fixed; pointer-events: none; z-index: 9; background: var(--raised);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,.15); display: none; }
@media (max-width: 1100px) {
  .layout { grid-template-columns: 1fr; }
  aside.side .body { max-height: none; }
  .stage { height: clamp(480px, 75vh, 760px); }
}
@media (max-width: 700px) {
  .stage { display: flex; flex-direction: column; }
  .stage canvas { flex: 1 1 0; min-height: 0; height: auto; }
  .legend { position: static; max-width: none; max-height: 92px; margin: 6px 8px 0; flex: none;
    display: flex; flex-wrap: wrap; gap: 0 12px; backdrop-filter: none; order: -1; }
  .legend .ttl { width: 100%; }
  .legend .row .lab { max-width: 160px; }
  .hint { display: none; }
}
@media (max-width: 640px) {
  .wrap { padding: 12px 16px 32px; }
  .panels { grid-template-columns: 1fr; }
  .hbars .row { grid-template-columns: 120px 1fr 36px; }
  .search { max-width: none; flex-basis: 100%; }
  #edge-count { margin-left: 0; }
}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1 id="h-title">TCS vendor-risk intelligence graph</h1>
      <div class="sub" id="h-meta"></div>
    </div>
    <div class="tiles" id="tiles"></div>
  </header>
  <p class="note" id="h-source"></p>
  <div class="layout">
    <section class="card" aria-label="Similarity graph">
      <div class="toolbar">
        <span class="ctl">Layout
          <span class="seg" id="seg-layout">
            <button data-v="force" aria-pressed="true">Force</button>
            <button data-v="pca" aria-pressed="false">PCA</button>
          </span>
        </span>
        <span class="ctl">Color
          <span class="seg" id="seg-color">
            <button data-v="cluster" aria-pressed="true" title="KMeans cluster">Cluster</button>
            <button data-v="community" aria-pressed="false" title="Louvain community">Community</button>
            <button data-v="category" aria-pressed="false" title="Dominant risk category">Category</button>
          </span>
        </span>
        <label class="ctl">Min cosine
          <input type="range" id="min-cos" min="0" max="1" step="0.01" value="0.3">
          <output id="min-cos-v">0.30</output>
        </label>
        <input class="search" id="q" type="search" placeholder="Search documents…" aria-label="Search documents">
        <span class="ctl muted" id="edge-count"></span>
      </div>
      <div class="stage" id="stage">
        <canvas id="graph" aria-label="Document similarity graph"></canvas>
        <div class="legend" id="legend"></div>
        <div class="tip" id="tip"></div>
        <div class="hint">Drag to pan · scroll to zoom · double-click to fit · click a node for details</div>
      </div>
    </section>
    <aside class="card side" aria-label="Details">
      <h2 id="side-h">Details</h2>
      <div class="body" id="side"></div>
    </aside>
  </div>
  <div class="panels">
    <section class="card"><h2>Documents by risk category</h2><div class="body hbars" id="p-cats"></div></section>
    <section class="card"><h2>Pairwise cosine similarity</h2><div class="body" id="p-hist"></div></section>
    <section class="card"><h2>KMeans clusters</h2><div class="body" id="p-clusters"></div></section>
    <section class="card"><h2>Riskiest documents</h2><div class="body list" id="p-risky"></div></section>
    <section class="card"><h2>Documents per month</h2><div class="body" id="p-time"></div></section>
    <section class="card"><h2>Same story, different outlets</h2><div class="body list" id="p-pairs"></div></section>
    <section class="card"><h2>Sources</h2><div class="body hbars" id="p-sources"></div></section>
  </div>
</div>
<div class="chart-tip" id="ctip"></div>
<script type="application/json" id="gold-data">__GOLD_DATA_JSON__</script>
<script>
(function () {
'use strict';
var DATA = JSON.parse(document.getElementById('gold-data').textContent);
var META = DATA.meta || {};
var STATS = DATA.stats || {};
var CAT_LABEL = {
  ethics: 'Ethics', compliance_regulatory: 'Compliance / regulatory',
  legal_litigation: 'Legal / litigation', data_privacy_cyber: 'Data privacy / cyber',
  labor_hr: 'Labour / HR', financial: 'Financial', operational_delivery: 'Operational / delivery',
  reputational: 'Reputational', esg: 'ESG', geopolitical_immigration: 'Geopolitical / immigration'
};
function catLabel(c) { return !c || c === 'none' ? 'No category' : (CAT_LABEL[c] || String(c).replace(/_/g, ' ')); }
function $(id) { return document.getElementById(id); }
function fmt(v, d) { return (v === null || v === undefined || isNaN(v)) ? '–' : Number(v).toFixed(d === undefined ? 2 : d); }
function el(tag, attrs) {
  var e = document.createElement(tag);
  if (attrs) for (var k in attrs) {
    if (attrs[k] === null || attrs[k] === undefined) continue;
    if (k === 'text') e.textContent = attrs[k];
    else if (k === 'cls') e.className = attrs[k];
    else if (k === 'style') e.style.cssText = attrs[k];
    else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  for (var i = 2; i < arguments.length; i++) {
    var c = arguments[i];
    if (c === null || c === undefined || c === false) continue;
    e.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
  }
  return e;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
function safeUrl(u) { return /^https?:\/\//i.test(u || '') ? u : null; }
function joinParts(a) { return a.filter(function (x) { return x !== null && x !== undefined && x !== ''; }).join(' · '); }
function shortDate(iso) { return iso ? String(iso).slice(0, 10) : 'undated'; }

var tokens = {};
function readTokens() {
  var cs = getComputedStyle(document.documentElement);
  ['--surface', '--ink', '--ink2', '--muted', '--edge-rgb', '--other', '--accent', '--grid']
    .forEach(function (n) { tokens[n] = cs.getPropertyValue(n).trim(); });
  tokens.other = tokens['--other'];
  tokens.series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(function (i) { return cs.getPropertyValue('--s' + i).trim(); });
}
readTokens();

// ------------------------------------------------------------------ data
var nodes = (DATA.nodes || []).map(function (n) { return Object.assign({}, n); });
var byId = new Map(nodes.map(function (n) { return [n.id, n]; }));
var links = (DATA.links || []).map(function (l) {
  return { s: byId.get(l.source), t: byId.get(l.target), cosine: l.cosine, mutual: l.mutual };
}).filter(function (l) { return l.s && l.t; });
nodes.forEach(function (n, i) {
  n.r = 3.5 + 9 * Math.sqrt(Math.max(0, Math.min(100, n.risk_score || 0)) / 100);
  // deterministic jitter so identical PCA points don't overlap
  var h = Math.sin((i + 1) * 12.9898) * 43758.5453; h = h - Math.floor(h);
  n.wx = (n.x || 0) * 260 + (h - 0.5) * 6; n.wy = (n.y || 0) * 260 + (((h * 7) % 1) - 0.5) * 6;
  n.idx = i; n.vx = 0; n.vy = 0; n.text = ((n.title || '') + ' ' + (n.domain || '')).toLowerCase();
});

var byRisk = nodes.slice().sort(function (a, b) { return (b.risk_score || 0) - (a.risk_score || 0); });
var state = {
  layout: 'force', colorBy: 'cluster',
  minCos: META.graph_min_cosine !== undefined ? META.graph_min_cosine : 0.3,
  query: '', hover: null, selected: null, group: null, matches: null
};
var activeLinks = [], adj = new Map();
function rebuildLinks() {
  activeLinks = links.filter(function (l) { return l.cosine >= state.minCos - 1e-9; });
  adj = new Map(nodes.map(function (n) { return [n, new Set()]; }));
  activeLinks.forEach(function (l) { adj.get(l.s).add(l.t); adj.get(l.t).add(l.s); });
  $('edge-count').textContent = activeLinks.length + ' of ' + links.length + ' edges';
}

// ------------------------------------------------------------------ header
(function header() {
  var km = META.kmeans || {};
  $('h-meta').textContent = joinParts([
    META.generated_at ? 'Generated ' + String(META.generated_at).replace('T', ' ').replace('+00:00', ' UTC') : null,
    'embeddings ' + (META.embedding_model || '?'),
    'KMeans k=' + (km.k || '–') + (km.silhouette !== null && km.silhouette !== undefined ? ' (silhouette ' + fmt(km.silhouette) + ')' : ''),
    'k-NN graph k=' + (META.graph_k || '–') + ', cosine ≥ ' + fmt(META.graph_min_cosine)]);
  if (META.synthetic) $('h-title').appendChild(el('span', { cls: 'badge', text: 'SYNTHETIC DATA' }));
  $('h-source').textContent = 'Data source: ' + (META.data_source || 'unknown') +
    (META.qdrant ? ' · Qdrant: ' + META.qdrant : '');
  var tiles = [
    [META.n_docs || 0, 'documents'], [META.n_chunks || 0, 'chunks'], [META.n_edges || 0, 'edges'],
    [META.n_clusters || 0, 'clusters'], [META.n_communities || 0, 'communities'],
    [fmt(STATS.mean_risk_score, 1), 'mean risk']
  ];
  tiles.forEach(function (t) {
    $('tiles').appendChild(el('div', { cls: 'tile' }, el('div', { cls: 'v', text: String(t[0]) }), el('div', { cls: 'l', text: t[1] })));
  });
})();

// ------------------------------------------------------------------ colors & legend
var MAX_COLORS = 12;
function keyOf(n, mode) {
  return String(mode === 'cluster' ? n.cluster : mode === 'community' ? n.community : (n.dominant_category || 'none'));
}
function groupsFor(mode) {
  if (mode === 'cluster') return (DATA.clusters || []).map(function (c) { return { key: String(c.cluster), label: 'C' + c.cluster + ' · ' + (c.label || ''), count: c.size || 0 }; });
  if (mode === 'community') return (DATA.communities || []).map(function (c) { return { key: String(c.cluster), label: 'G' + c.cluster + ' · ' + (c.label || ''), count: c.size || 0 }; });
  var counts = {};
  nodes.forEach(function (n) { var k = keyOf(n, 'category'); counts[k] = (counts[k] || 0) + 1; });
  return Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a] || (a < b ? -1 : 1); })
    .map(function (k) { return { key: k, label: catLabel(k), count: counts[k] }; });
}
var palettes = {};
function recolor() {
  var mode = state.colorBy, p = palettes[mode];
  nodes.forEach(function (n) {
    var k = keyOf(n, mode), g = p.byKey.get(k);
    n.gkey = k; n.col = g ? g.color : tokens.other; n.gother = !g || g.other;
  });
}
function buildPalettes() {
  ['cluster', 'community', 'category'].forEach(function (mode) {
    var gs = groupsFor(mode), map = new Map(), byKey = new Map();
    gs.forEach(function (g, i) {
      g.other = i >= MAX_COLORS; g.color = g.other ? tokens.other : tokens.series[i];
      map.set(g.key, g.color); byKey.set(g.key, g);
    });
    palettes[mode] = { groups: gs, map: map, byKey: byKey };
  });
  recolor();
}
buildPalettes();
function colorOf(n) { return n.col; }
function inGroup(n) {
  if (state.group === null) return true;
  return state.group === '__other' ? n.gother : n.gkey === state.group;
}
function toggleGroup(key) { state.group = state.group === key ? null : key; renderLegend(); draw(); }
function renderLegend() {
  var box = clear($('legend')), p = palettes[state.colorBy];
  var titles = { cluster: 'KMeans clusters (top terms)', community: 'Louvain communities', category: 'Dominant risk category' };
  box.appendChild(el('div', { cls: 'ttl', text: titles[state.colorBy] }));
  var shown = p.groups.slice(0, MAX_COLORS);
  if (p.groups.length > MAX_COLORS) {
    var rest = p.groups.slice(MAX_COLORS);
    shown = shown.concat([{ key: '__other', label: 'Other (' + rest.length + ' groups)', color: tokens.other,
      count: rest.reduce(function (a, g) { return a + g.count; }, 0) }]);
  }
  shown.forEach(function (g) {
    var row = el('div', { cls: 'row' + (state.group !== null && state.group !== g.key ? ' off' : ''), title: g.label,
      role: 'button', tabindex: '0', 'aria-pressed': String(state.group === g.key),
      onclick: function () { toggleGroup(g.key); },
      onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleGroup(g.key); } } },
      el('span', { cls: 'sw', style: 'background:' + g.color }), el('span', { cls: 'lab', text: g.label }),
      el('span', { cls: 'cnt', text: String(g.count) }));
    box.appendChild(row);
  });
  box.style.display = shown.length ? '' : 'none';
}
function setColorBy(mode, group) {
  state.colorBy = mode; state.group = group === undefined ? null : group;
  Array.prototype.forEach.call($('seg-color').children, function (b) { b.setAttribute('aria-pressed', String(b.dataset.v === mode)); });
  recolor(); renderLegend(); renderSide(); draw();
}

// ------------------------------------------------------------------ canvas & view
var canvas = $('graph'), ctx = canvas.getContext('2d'), stage = $('stage'), legendBox = $('legend');
var W = 0, H = 0, dpr = 1, view = { k: 1, x: 0, y: 0 }, userMoved = false;
function sizeStage() {
  // wide screens: the graph card fills the viewport below the header; narrow: CSS clamp()
  var root = document.documentElement.style;
  if (window.innerWidth > 1100) {
    var top = stage.getBoundingClientRect().top + window.pageYOffset;
    root.setProperty('--stage-h', Math.round(Math.max(480, Math.min(1000, window.innerHeight - top - 16))) + 'px');
  } else root.removeProperty('--stage-h');
}
function resize() {
  var r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  if (Math.abs(r.width - W) < 0.5 && Math.abs(r.height - H) < 0.5 && d === dpr) return;
  dpr = d; W = r.width; H = r.height;
  canvas.width = Math.max(1, Math.round(W * dpr)); canvas.height = Math.max(1, Math.round(H * dpr));
  if (!userMoved) fit();
  draw();
}
function legendPad() {
  var overlay = getComputedStyle(legendBox).position === 'absolute' && legendBox.style.display !== 'none';
  return overlay ? Math.min(legendBox.offsetWidth + 24, W * 0.45) : 20;
}
function fit() {
  if (!nodes.length || !W) { view = { k: 1, x: W / 2, y: H / 2 }; return; }
  var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  nodes.forEach(function (n) { x0 = Math.min(x0, n.wx - n.r); x1 = Math.max(x1, n.wx + n.r); y0 = Math.min(y0, n.wy - n.r); y1 = Math.max(y1, n.wy + n.r); });
  var padL = legendPad(), pad = 30;
  var k = Math.min((W - padL - pad) / Math.max(1, x1 - x0), (H - 2 * pad) / Math.max(1, y1 - y0));
  k = Math.max(0.05, Math.min(k, 3));
  view.k = k; view.x = padL + (W - padL - pad) / 2 - (x0 + x1) / 2 * k; view.y = H / 2 - (y0 + y1) / 2 * k;
}
function toWorld(sx, sy) { return [(sx - view.x) / view.k, (sy - view.y) / view.k]; }

// ------------------------------------------------------------------ force simulation
var alpha = 0, rafSim = 0;
function tick() {
  var N = nodes.length, i, j, a, b, dx, dy, d2, d, f;
  for (i = 0; i < N; i++) {
    a = nodes[i];
    for (j = i + 1; j < N; j++) {
      b = nodes[j]; dx = b.wx - a.wx; dy = b.wy - a.wy; d2 = dx * dx + dy * dy;
      if (d2 > 160000) continue;
      if (d2 < 0.01) { dx = 0.1 * ((i % 7) - 3) + 0.05; dy = 0.1; d2 = dx * dx + dy * dy; }
      f = 70 * alpha / d2;
      d = Math.sqrt(d2);
      var minD = a.r + b.r + 3;
      if (d < minD) f += 0.35 * (minD - d) / d;
      a.vx -= dx * f; a.vy -= dy * f; b.vx += dx * f; b.vy += dy * f;
    }
  }
  for (i = 0; i < activeLinks.length; i++) {
    var l = activeLinks[i]; a = l.s; b = l.t;
    dx = b.wx - a.wx; dy = b.wy - a.wy; d = Math.sqrt(dx * dx + dy * dy) || 0.01;
    var target = 18 + 110 * (1 - l.cosine);
    f = (d - target) / d * alpha * 0.25 * (0.25 + l.cosine);
    a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
  }
  for (i = 0; i < N; i++) {
    a = nodes[i];
    var g = adj.get(a).size ? 0.012 : 0.03;
    a.vx -= a.wx * g * alpha; a.vy -= a.wy * g * alpha;
    if (a.fixed) { a.vx = a.vy = 0; continue; }
    a.vx *= 0.6; a.vy *= 0.6; a.wx += a.vx; a.wy += a.vy;
  }
  alpha *= 0.975;
}
function runSim(a0) {
  if (state.layout !== 'force' || !nodes.length) return;
  alpha = Math.max(alpha, a0);
  if (rafSim) return;
  var loop = function () {
    var steps = nodes.length > 300 ? 1 : 2;
    for (var s = 0; s < steps; s++) tick();
    if (!userMoved) fit();
    draw();
    if (alpha > 0.004 && state.layout === 'force') rafSim = requestAnimationFrame(loop); else rafSim = 0;
  };
  rafSim = requestAnimationFrame(loop);
}
function animateTo(targets, ms) {
  var start = performance.now(), from = nodes.map(function (n) { return [n.wx, n.wy]; });
  var step = function (now) {
    var t = Math.min(1, (now - start) / ms), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    nodes.forEach(function (n, i) { n.wx = from[i][0] + (targets[i][0] - from[i][0]) * e; n.wy = from[i][1] + (targets[i][1] - from[i][1]) * e; });
    if (!userMoved) fit();
    draw();
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function setLayout(mode) {
  state.layout = mode;
  Array.prototype.forEach.call($('seg-layout').children, function (b) { b.setAttribute('aria-pressed', String(b.dataset.v === mode)); });
  if (mode === 'pca') {
    alpha = 0; if (rafSim) { cancelAnimationFrame(rafSim); rafSim = 0; }
    var S = 60 + 22 * Math.sqrt(nodes.length);
    userMoved = false;
    animateTo(nodes.map(function (n) { return [(n.x || 0) * S, (n.y || 0) * S]; }), 700);
  } else { userMoved = false; runSim(0.8); }
}

// ------------------------------------------------------------------ drawing
var drawQueued = false, TAU = 6.2832, EDGE_BUCKETS = 6;
var LABEL_FONT = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
function draw() {
  if (drawQueued) return; drawQueued = true;
  requestAnimationFrame(function () { drawQueued = false; paint(); });
}
function focusSet() {
  var f = state.hover || state.selected;
  if (!f) return null;
  var s = new Set(adj.get(f)); s.add(f); return s;
}
function zoomR() { return Math.max(0.75, Math.min(1.6, Math.sqrt(view.k))); }
function strokeLinks(list, style, width) {
  if (!list.length) return;
  ctx.strokeStyle = style; ctx.lineWidth = width; ctx.beginPath();
  for (var i = 0; i < list.length; i++) { ctx.moveTo(list[i].s.sx, list[i].s.sy); ctx.lineTo(list[i].t.sx, list[i].t.sy); }
  ctx.stroke();
}
function paint() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!nodes.length) return;
  var fs = focusSet(), focus = state.hover || state.selected, N = nodes.length, i, n, zr = zoomR(), m = 24;
  for (i = 0; i < N; i++) {
    n = nodes[i]; n.sx = n.wx * view.k + view.x; n.sy = n.wy * view.k + view.y;
    n._flt = !!((state.matches && !state.matches.has(n)) || !inGroup(n));
    n._dim = n._flt || (fs !== null && !fs.has(n));
    n._vis = n.sx > -m && n.sy > -m && n.sx < W + m && n.sy < H + m;
  }
  // edges, batched into a few alpha buckets (one path + stroke per bucket)
  var span = Math.max(0.05, 1 - state.minCos), rgb = tokens['--edge-rgb'], buckets = [], faded = [], hot = [], b;
  for (b = 0; b < EDGE_BUCKETS; b++) buckets.push([]);
  for (i = 0; i < activeLinks.length; i++) {
    var l = activeLinks[i];
    if (!l.s._vis && !l.t._vis) continue;
    if (focus && (l.s === focus || l.t === focus)) { hot.push(l); continue; }
    if (l.s._dim || l.t._dim) { faded.push(l); continue; }
    b = Math.floor((l.cosine - state.minCos) / span * EDGE_BUCKETS);
    buckets[Math.max(0, Math.min(EDGE_BUCKETS - 1, b))].push(l);
  }
  ctx.lineCap = 'round';
  strokeLinks(faded, 'rgba(' + rgb + ',0.05)', 0.6);
  buckets.forEach(function (list, bi) {
    var f = (bi + 0.5) / EDGE_BUCKETS;
    strokeLinks(list, 'rgba(' + rgb + ',' + (0.22 + 0.5 * f).toFixed(3) + ')', 0.7 + 2.0 * f);
  });
  ctx.strokeStyle = tokens['--ink'];
  hot.forEach(function (l) {
    var norm = Math.max(0, Math.min(1, (l.cosine - state.minCos) / span));
    ctx.globalAlpha = 0.4 + 0.5 * norm; ctx.lineWidth = 1.2 + 2.3 * norm;
    ctx.beginPath(); ctx.moveTo(l.s.sx, l.s.sy); ctx.lineTo(l.t.sx, l.t.sy); ctx.stroke();
  });
  ctx.globalAlpha = 1;
  // nodes: dimmed first, then active on top
  ctx.lineWidth = 1.5; ctx.strokeStyle = tokens['--surface'];
  var drawNode = function (nd) {
    ctx.fillStyle = nd.col; ctx.beginPath(); ctx.arc(nd.sx, nd.sy, nd.r * zr, 0, TAU); ctx.fill(); ctx.stroke();
  };
  ctx.globalAlpha = 0.14;
  for (i = 0; i < N; i++) { n = nodes[i]; if (n._vis && n._dim) drawNode(n); }
  ctx.globalAlpha = 1;
  for (i = 0; i < N; i++) { n = nodes[i]; if (n._vis && !n._dim) drawNode(n); }
  if (focus) drawNode(focus);
  if (state.selected) {
    ctx.strokeStyle = tokens['--ink']; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(state.selected.sx, state.selected.sy, state.selected.r * zr + 4, 0, TAU); ctx.stroke();
  }
  drawLabels(zr);
}
function shorten(t, max) { return t.length > max ? t.slice(0, max - 1) + '…' : t; }
// Labels are placed greedily by priority and skipped when they would overlap an earlier one.
function drawLabels(zr) {
  var boxes = [], sel = state.selected;
  ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
  function put(txt, n, opts) {
    ctx.font = (opts.bold ? '600 ' : '') + LABEL_FONT;
    var tw = ctx.measureText(txt).width, r = opts.center ? 0 : n.r * zr, x, y = n.sy;
    if (opts.center) x = n.sx - tw / 2; else { x = n.sx + r + 5; if (x + tw > W - 6) x = n.sx - r - 5 - tw; }
    x = Math.max(4, Math.min(x, W - 4 - tw));
    var bx = { x0: x - 3, x1: x + tw + 3, y0: y - 8, y1: y + 8 };
    if (!opts.force) for (var i = 0; i < boxes.length; i++) {
      var o = boxes[i];
      if (bx.x0 < o.x1 && bx.x1 > o.x0 && bx.y0 < o.y1 && bx.y1 > o.y0) return false;
    }
    boxes.push(bx);
    ctx.lineWidth = 4; ctx.strokeStyle = tokens['--surface']; ctx.strokeText(txt, x, y);
    ctx.fillStyle = tokens['--ink']; ctx.fillText(txt, x, y);
    return true;
  }
  // 1) the selected document and its closest linked neighbours
  if (sel && sel !== state.hover) {
    put(sel.title || ('#' + sel.id), sel, { bold: true, force: true });
    var shown = 0;
    (sel.neighbors || []).forEach(function (pr) {
      var o = byId.get(pr[0]);
      if (shown < 5 && o && o._vis && !o._flt && adj.get(sel).has(o) && put(shorten(o.title || ('#' + o.id), 40), o, {})) shown++;
    });
  }
  // 2) search hits
  if (state.matches && state.matches.size <= 80) {
    byRisk.forEach(function (n) { if (n._vis && state.matches.has(n)) put(shorten(n.title || ('#' + n.id), 40), n, {}); });
  }
  // 3) group names (clusters / communities only: categories are interleaved) unless a node is focused
  //    or so few nodes are in view that document titles are shown instead
  var inView = 0; nodes.forEach(function (n) { if (n._vis && !n._flt) inView++; });
  var titles = view.k >= 2 || inView <= 45;
  if (state.colorBy !== 'category' && !titles && !(state.hover || sel) && W >= 600) {
    // anchor each name on the member closest to the group centroid, so it never floats between blobs
    var acc = new Map();
    nodes.forEach(function (n) {
      if (!n._vis || n._flt || n.gother) return;
      var a = acc.get(n.gkey); if (!a) { a = { sx: 0, sy: 0, c: 0, best: null, bd: Infinity }; acc.set(n.gkey, a); }
      a.sx += n.sx; a.sy += n.sy; a.c++;
    });
    nodes.forEach(function (n) {
      var a = n._vis && !n._flt && !n.gother ? acc.get(n.gkey) : null;
      if (!a) return;
      var dx = n.sx - a.sx / a.c, dy = n.sy - a.sy / a.c, d = dx * dx + dy * dy;
      if (d < a.bd) { a.bd = d; a.best = n; }
    });
    palettes[state.colorBy].groups.slice(0, MAX_COLORS).forEach(function (g) {
      var a = acc.get(g.key);
      if (a && a.c >= 3 && a.best) put(shorten(g.label, 34), { sx: a.best.sx, sy: a.best.sy, r: 0 }, { center: true, bold: true });
    });
  }
  // 4) document titles when zoomed in (or when few documents are in view), riskiest first
  if (titles) {
    var cnt = 0;
    for (var i = 0; i < byRisk.length && cnt < 90; i++) {
      var n = byRisk[i];
      if (n._vis && !n._dim && n !== sel && n !== state.hover && put(shorten(n.title || ('#' + n.id), 36), n, {})) cnt++;
    }
  }
}

// ------------------------------------------------------------------ interaction
function hit(sx, sy) {
  var best = null, bd = Infinity, zr = zoomR();
  for (var i = 0; i < nodes.length; i++) {
    var n = nodes[i];
    if (n._flt) continue;
    var dx = n.wx * view.k + view.x - sx, dy = n.wy * view.k + view.y - sy, d2 = dx * dx + dy * dy;
    var lim = Math.max(12, n.r * zr + 6);
    if (d2 < lim * lim && d2 < bd) { bd = d2; best = n; }
  }
  return best;
}
var tip = $('tip');
function showTip(n, sx, sy) {
  if (!n) { tip.style.display = 'none'; return; }
  clear(tip);
  tip.appendChild(el('div', { cls: 'r', text: fmt(n.risk_score, 0) },
    el('span', { cls: 'm', style: 'font-size:12px;font-weight:400', text: ' risk score' })));
  tip.appendChild(el('div', { cls: 't', text: n.title || ('#' + n.id) }));
  tip.appendChild(el('div', { cls: 'm', text: joinParts([n.domain || n.source, n.source_type, n.published_at ? shortDate(n.published_at) : null]) }));
  tip.appendChild(el('div', { cls: 'm', text: (n.risk_categories || []).map(catLabel).join(', ') || 'No risk category' }));
  tip.appendChild(el('div', { cls: 'm', text: 'Cluster C' + n.cluster + ' · community G' + n.community + ' · ' + adj.get(n).size + ' links' }));
  tip.style.display = 'block';
  var tw = tip.offsetWidth, th = tip.offsetHeight, ox = canvas.offsetLeft, oy = canvas.offsetTop;
  var x = ox + sx + 14, y = oy + sy + 14, SW = stage.clientWidth, SH = stage.clientHeight;
  if (x + tw > SW - 6) x = ox + sx - tw - 14;
  if (y + th > SH - 6) y = oy + sy - th - 14;
  tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
}
var drag = null;
canvas.addEventListener('pointerdown', function (e) {
  var r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  var n = hit(sx, sy);
  drag = { node: n, sx: sx, sy: sy, vx: view.x, vy: view.y, moved: false };
  canvas.setPointerCapture(e.pointerId); canvas.classList.add('dragging');
});
canvas.addEventListener('pointermove', function (e) {
  var r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (drag) {
    if (Math.abs(sx - drag.sx) + Math.abs(sy - drag.sy) > 3) drag.moved = true;
    if (!drag.moved) return;
    if (drag.node) {
      var w = toWorld(sx, sy); drag.node.wx = w[0]; drag.node.wy = w[1]; drag.node.fixed = true;
      userMoved = true; if (state.layout === 'force') runSim(0.25); else draw();
    } else { view.x = drag.vx + sx - drag.sx; view.y = drag.vy + sy - drag.sy; userMoved = true; draw(); }
    showTip(null);
    return;
  }
  var n = hit(sx, sy);
  if (n !== state.hover) { state.hover = n; draw(); }
  canvas.style.cursor = n ? 'pointer' : '';
  showTip(n, sx, sy);
});
function endDrag(doSelect) {
  if (drag && doSelect && !drag.moved) { showTip(null); select(drag.node); }
  if (drag && drag.node) drag.node.fixed = false;
  drag = null; canvas.classList.remove('dragging');
}
canvas.addEventListener('pointerup', function () { endDrag(true); });
canvas.addEventListener('pointercancel', function () { endDrag(false); });
canvas.addEventListener('pointerleave', function () { if (!drag) { state.hover = null; showTip(null); draw(); } });
canvas.addEventListener('wheel', function (e) {
  e.preventDefault();
  var r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  var w = toWorld(sx, sy), k = Math.max(0.05, Math.min(12, view.k * Math.exp(-e.deltaY * 0.0015)));
  view.k = k; view.x = sx - w[0] * k; view.y = sy - w[1] * k; userMoved = true; draw();
}, { passive: false });
canvas.addEventListener('dblclick', function () { userMoved = false; fit(); draw(); });
document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { select(null); state.group = null; renderLegend(); } });

function centerOn(n) {
  var k = Math.max(view.k, 1.2), padL = legendPad(); view.k = k;
  view.x = (W > 700 ? padL + (W - padL) / 2 : W / 2) - n.wx * k; view.y = H / 2 - n.wy * k; userMoved = true;
}
function select(n, center) {
  state.selected = n || null;
  if (n && center) {
    centerOn(n);
    var r = stage.getBoundingClientRect();
    if (r.top < 0 || r.bottom > window.innerHeight) stage.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  renderSide(); draw();
}

// ------------------------------------------------------------------ side panel
function sevTag(sev) { return el('span', { cls: 'sev ' + sev, text: sev }); }
function renderOverview(box) {
  $('side-h').textContent = 'Overview';
  var km = META.kmeans || {};
  box.appendChild(el('p', { cls: 'muted', text: 'Each dot is a TCS-relevant document; size grows with its risk score. Edges join documents whose embeddings are similar (k-NN, cosine). Click a dot to see its risk evidence and nearest neighbours.' }));
  var dl = el('dl', { cls: 'kv' });
  [['Embedding model', META.embedding_model], ['KMeans k', km.k], ['Silhouette (cosine)', km.silhouette !== null && km.silhouette !== undefined ? fmt(km.silhouette) : '–'],
   ['Louvain communities', META.n_communities], ['Graph', META.graph_k !== undefined ? 'k=' + META.graph_k + ', cosine ≥ ' + fmt(META.graph_min_cosine) : '–'],
   ['Qdrant', META.qdrant || '–']].forEach(function (r) { dl.appendChild(el('dt', { text: r[0] })); dl.appendChild(el('dd', { text: String(r[1] === undefined ? '–' : r[1]) })); });
  box.appendChild(dl);
  var deg = function (n) { return adj.get(n).size; };
  var central = nodes.slice().sort(function (a, b) { return deg(b) - deg(a) || (b.betweenness || 0) - (a.betweenness || 0) || (b.risk_score || 0) - (a.risk_score || 0); }).slice(0, 10);
  if (central.length && deg(central[0]) > 0) {
    box.appendChild(el('div', { cls: 'sec', text: 'Hub documents (most similar neighbours)' }));
    central.forEach(function (n) {
      box.appendChild(el('div', { cls: 'nb', title: n.title, onclick: function () { select(n, true); } },
        el('span', { cls: 'tt' }, el('span', { cls: 'sw', style: 'display:inline-block;margin-right:6px;vertical-align:-1px;background:' + colorOf(n) }), n.title || ('#' + n.id)),
        el('span', { cls: 'bar' }, el('i', { style: 'width:' + Math.round(100 * deg(n) / deg(central[0])) + '%' })),
        el('span', { cls: 'v', text: deg(n) + ' links' })));
    });
  }
}
function renderSide() {
  var box = clear($('side')), n = state.selected;
  if (!n) { renderOverview(box); return; }
  $('side-h').textContent = 'Document #' + n.id;
  var url = safeUrl(n.url);
  box.appendChild(el('div', { cls: 'dtitle' }, url ? el('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: n.title || url }) : (n.title || '')));
  box.appendChild(el('div', { cls: 'dmeta', text: joinParts([n.domain, n.source ? n.source + (n.source_type ? ' (' + n.source_type + ')' : '') : n.source_type, shortDate(n.published_at)]) }));
  if (n.also_seen_in) {
    var seen = (n.merged_sources || []).map(function (m) { return m.domain || m.source; }).filter(Boolean);
    box.appendChild(el('div', { cls: 'dmeta', text: 'also seen in ' + n.also_seen_in + ' source' + (n.also_seen_in > 1 ? 's' : '') + (seen.length ? ': ' + seen.join(', ') : '') }));
  }
  var chips = el('div', { cls: 'chips' });
  (n.risk_categories || []).forEach(function (c) {
    chips.appendChild(el('span', { cls: 'chip' }, el('span', { cls: 'sq', style: 'background:' + (palettes.category.map.get(c) || tokens.other) }), catLabel(c)));
  });
  if (!(n.risk_categories || []).length) chips.appendChild(el('span', { cls: 'chip', text: 'No risk category' }));
  box.appendChild(chips);
  var dl = el('dl', { cls: 'kv' });
  [['Risk score', fmt(n.risk_score, 1) + ' / 100'], ['Sentiment', fmt(n.sentiment)], ['TCS relevance', fmt(n.relevance)],
   ['KMeans cluster', 'C' + n.cluster], ['Louvain community', 'G' + n.community],
   ['Degree / betweenness', fmt(n.degree, 3) + ' / ' + fmt(n.betweenness, 3)], ['Eigenvector', fmt(n.eigenvector, 3)]]
    .forEach(function (r) { dl.appendChild(el('dt', { text: r[0] })); dl.appendChild(el('dd', { text: r[1] })); });
  box.appendChild(dl);
  if (n.summary) { box.appendChild(el('div', { cls: 'sec', text: 'Summary' })); box.appendChild(el('div', { text: n.summary })); }
  box.appendChild(el('div', { cls: 'sec', text: 'Risk signals' }));
  if (!(n.signals || []).length) box.appendChild(el('div', { cls: 'muted', text: 'No risk signals extracted.' }));
  (n.signals || []).forEach(function (s) {
    box.appendChild(el('div', { cls: 'sig' }, sevTag(s.severity), el('b', { text: catLabel(s.category) }), ' · ', s.keyword,
      s.evidence ? el('q', { text: s.evidence }) : null));
  });
  var ents = n.entities || {}, entList = [].concat(ents.regulators || [], ents.organizations || [], ents.places || []);
  if (entList.length) {
    box.appendChild(el('div', { cls: 'sec', text: 'Entities' }));
    var ec = el('div', { cls: 'chips' }); entList.slice(0, 14).forEach(function (t) { ec.appendChild(el('span', { cls: 'chip', text: t })); }); box.appendChild(ec);
  }
  box.appendChild(el('div', { cls: 'sec', text: 'Nearest neighbours (cosine)' }));
  (n.neighbors || []).forEach(function (pair) {
    var o = byId.get(pair[0]); if (!o) return;
    box.appendChild(el('div', { cls: 'nb', title: o.title, onclick: function () { select(o, true); } },
      el('span', { cls: 'tt' }, el('span', { cls: 'sw', style: 'display:inline-block;margin-right:6px;vertical-align:-1px;background:' + colorOf(o) }), o.title || ('#' + o.id)),
      el('span', { cls: 'bar' }, el('i', { style: 'width:' + Math.round(100 * Math.max(0, pair[1])) + '%' })),
      el('span', { cls: 'v', text: fmt(pair[1]) })));
  });
  if (!(n.neighbors || []).length) box.appendChild(el('div', { cls: 'muted', text: 'No other documents.' }));
}

// ------------------------------------------------------------------ panels
var ctip = $('ctip');
function chartTip(target, text) {
  target.addEventListener('pointermove', function (e) { ctip.textContent = text; ctip.style.display = 'block'; ctip.style.left = (e.clientX + 12) + 'px'; ctip.style.top = (e.clientY + 12) + 'px'; });
  target.addEventListener('pointerleave', function () { ctip.style.display = 'none'; });
}
function hbars(box, rows, onclick) {
  var max = Math.max.apply(null, rows.map(function (r) { return r[1]; }).concat([1]));
  rows.forEach(function (r) {
    var row = el('div', { cls: 'row', onclick: onclick ? function () { onclick(r[0]); } : null },
      el('span', { cls: 'lab', title: r[2] || r[0], text: r[2] || r[0] }),
      el('span', { cls: 'track' }, el('i', { style: 'width:' + (100 * r[1] / max).toFixed(1) + '%' })),
      el('span', { cls: 'v', text: String(r[1]) }));
    if (!onclick) row.style.cursor = 'default';
    box.appendChild(row);
  });
}
function svgEl(tag, attrs) {
  var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (var k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
(function categories() {
  var cc = STATS.category_counts || {}, rows = Object.keys(cc).map(function (k) { return [k, cc[k], catLabel(k)]; })
    .sort(function (a, b) { return b[1] - a[1]; });
  hbars($('p-cats'), rows, function (cat) { setColorBy('category', palettes.category.map.has(cat) ? cat : null); });
  $('p-cats').appendChild(el('div', { cls: 'muted', style: 'font-size:12px;margin-top:6px', text: 'A document can carry several categories. Click a bar to highlight its documents.' }));
})();
var histThr = null, histX = null;
(function histogram() {
  var h = STATS.cosine_histogram || {}, box = $('p-hist');
  h.counts = h.counts || []; h.edges = h.edges || [];
  if (!h.counts.length || !h.counts.some(function (c) { return c > 0; })) { box.appendChild(el('div', { cls: 'muted', text: 'Needs at least two documents.' })); return; }
  var w = 460, ht = 190, ml = 34, mr = 8, mt = 10, mb = 26, iw = w - ml - mr, ih = ht - mt - mb;
  var lo = h.edges[0], hi = h.edges[h.edges.length - 1], max = Math.max.apply(null, h.counts);
  histX = function (v) { return ml + (v - lo) / (hi - lo) * iw; };
  var svg = svgEl('svg', { viewBox: '0 0 ' + w + ' ' + ht, width: '100%', role: 'img', 'aria-label': 'Histogram of pairwise cosine similarity' });
  [0, 0.5, 1].forEach(function (f) {
    var y = mt + ih - f * ih;
    svg.appendChild(svgEl('line', { x1: ml, x2: w - mr, y1: y, y2: y, stroke: tokens['--grid'] }));
    var t = svgEl('text', { x: ml - 4, y: y + 3, 'text-anchor': 'end' }); t.textContent = String(Math.round(f * max)); svg.appendChild(t);
  });
  h.counts.forEach(function (c, i) {
    var x0 = histX(h.edges[i]) + 1, x1 = histX(h.edges[i + 1]) - 1, bh = c / max * ih;
    var g = svgEl('g', {});
    g.appendChild(svgEl('rect', { x: x0 - 1, y: mt, width: x1 - x0 + 2, height: ih, fill: 'transparent' }));
    if (c > 0) g.appendChild(svgEl('rect', { class: 'bar-rect', x: x0, y: mt + ih - Math.max(bh, 1.5), width: Math.max(1, x1 - x0), height: Math.max(bh, 1.5), rx: Math.min(3, (x1 - x0) / 2) }));
    chartTip(g, 'cosine ' + fmt(h.edges[i]) + '–' + fmt(h.edges[i + 1]) + ': ' + c + ' pairs');
    svg.appendChild(g);
  });
  svg.appendChild(svgEl('line', { class: 'base', x1: ml, x2: w - mr, y1: mt + ih, y2: mt + ih }));
  var ticks = []; for (var v = Math.ceil(lo * 5) / 5; v <= hi + 1e-9; v += 0.2) ticks.push(Math.round(v * 10) / 10);
  ticks.forEach(function (v) { var t = svgEl('text', { x: histX(v), y: ht - 8, 'text-anchor': 'middle' }); t.textContent = v.toFixed(1); svg.appendChild(t); });
  histThr = svgEl('line', { class: 'thr', y1: mt - 4, y2: mt + ih });
  var lab = svgEl('text', { y: mt + 6, 'text-anchor': 'start', style: 'fill:var(--ink)' }); lab.textContent = 'min cosine';
  histThr.lab = lab; svg.appendChild(histThr); svg.appendChild(lab);
  box.appendChild(svg);
  var total = h.counts.reduce(function (a, b) { return a + b; }, 0);
  box.appendChild(el('div', { cls: 'muted', style: 'font-size:12px', text: total + ' document pairs. The dashed line is the graph’s current minimum cosine; drag the slider above the graph to move it.' }));
})();
function updateHistThr() {
  if (!histThr || !histX) return;
  var x = histX(Math.max(0, state.minCos));
  histThr.setAttribute('x1', x); histThr.setAttribute('x2', x);
  histThr.lab.setAttribute('x', x > 360 ? x - 4 : x + 4); histThr.lab.setAttribute('text-anchor', x > 360 ? 'end' : 'start');
}
(function clusterTable() {
  var box = $('p-clusters'), cl = DATA.clusters || [];
  if (!cl.length) { box.appendChild(el('div', { cls: 'muted', text: 'No clusters.' })); return; }
  var tb = el('tbody');
  cl.forEach(function (c) {
    var cats = Object.keys(c.dominant_categories || {}).slice(0, 2).map(catLabel).join(', ');
    tb.appendChild(el('tr', { onclick: function () { setColorBy('cluster', String(c.cluster)); } },
      el('td', {}, el('span', { cls: 'sq', style: 'background:' + (palettes.cluster.map.get(String(c.cluster)) || tokens.other) }), ' C' + c.cluster),
      el('td', {}, el('div', { style: 'font-weight:600', text: c.label }), el('div', { cls: 'muted', text: cats || '—' })),
      el('td', { cls: 'n', text: String(c.size) }), el('td', { cls: 'n', text: fmt(c.mean_risk_score, 1) }),
      el('td', { cls: 'n', text: fmt(c.mean_intra_cosine) })));
  });
  box.appendChild(el('table', {}, el('thead', {}, el('tr', {}, el('th', { text: '#' }), el('th', { text: 'Label (top terms) · categories' }),
    el('th', { cls: 'n', text: 'Docs' }), el('th', { cls: 'n', text: 'Mean risk' }), el('th', { cls: 'n', text: 'Intra cos' }))), tb));
})();
(function risky() {
  var box = $('p-risky'), rows = STATS.top_risky || [];
  if (!rows.length) box.appendChild(el('div', { cls: 'muted', text: 'No documents.' }));
  rows.forEach(function (r) {
    box.appendChild(el('div', { cls: 'it', onclick: function () { var n = byId.get(r.doc_id); if (n) select(n, true); } },
      el('div', { cls: 'score', text: fmt(r.risk_score, 0) }),
      el('div', {}, el('div', { cls: 'tt', text: r.title }),
        el('div', { cls: 'mm', text: (r.domain || '') + ' · ' + shortDate(r.published_at) + ' · ' + (r.risk_categories || []).map(catLabel).join(', ') }))));
  });
})();
(function timeline() {
  var box = $('p-time'), tl = STATS.timeline || [];
  if (!tl.length) { box.appendChild(el('div', { cls: 'muted', text: 'No publication dates available.' })); return; }
  var w = 460, ht = 190, ml = 30, mr = 8, mt = 10, mb = 26, iw = w - ml - mr, ih = ht - mt - mb;
  var max = Math.max.apply(null, tl.map(function (t) { return t.count; })), bw = iw / tl.length;
  var svg = svgEl('svg', { viewBox: '0 0 ' + w + ' ' + ht, width: '100%', role: 'img', 'aria-label': 'Documents per month' });
  [0, 0.5, 1].forEach(function (f) {
    var y = mt + ih - f * ih;
    svg.appendChild(svgEl('line', { x1: ml, x2: w - mr, y1: y, y2: y, stroke: tokens['--grid'] }));
    var t = svgEl('text', { x: ml - 4, y: y + 3, 'text-anchor': 'end' }); t.textContent = String(Math.round(f * max)); svg.appendChild(t);
  });
  var every = Math.max(1, Math.ceil(tl.length / 8));
  tl.forEach(function (t, i) {
    var x = ml + i * bw, bh = t.count / max * ih, g = svgEl('g', {});
    g.appendChild(svgEl('rect', { x: x, y: mt, width: bw, height: ih, fill: 'transparent' }));
    g.appendChild(svgEl('rect', { class: 'bar-rect', x: x + Math.min(2, bw * 0.1), y: mt + ih - bh, width: Math.max(1, bw - 2 * Math.min(2, bw * 0.1)), height: bh, rx: Math.min(4, bw / 4) }));
    chartTip(g, t.month + ': ' + t.count + ' docs · mean risk ' + fmt(t.mean_risk, 1));
    svg.appendChild(g);
    if (i % every === 0) { var lab = svgEl('text', { x: x + bw / 2, y: ht - 8, 'text-anchor': 'middle' }); lab.textContent = t.month; svg.appendChild(lab); }
  });
  svg.appendChild(svgEl('line', { class: 'base', x1: ml, x2: w - mr, y1: mt + ih, y2: mt + ih }));
  box.appendChild(svg);
})();
(function pairs() {
  var box = $('p-pairs'), rows = STATS.top_cross_source_pairs || [];
  if (!rows.length) box.appendChild(el('div', { cls: 'muted', text: 'No cross-outlet pairs.' }));
  rows.forEach(function (p) {
    box.appendChild(el('div', { cls: 'it', onclick: function () { var n = byId.get(p.a); if (n) select(n, true); } },
      el('div', { cls: 'score', style: 'font-size:13px', text: fmt(p.cosine) }),
      el('div', {}, el('div', { cls: 'tt', text: p.a_title }), el('div', { cls: 'mm', text: p.a_domain }),
        el('div', { cls: 'tt', style: 'margin-top:2px', text: p.b_title }), el('div', { cls: 'mm', text: p.b_domain }))));
  });
})();
(function sources() {
  var box = $('p-sources'), s = STATS.sources || {};
  [['by_source_type', 'Source type'], ['by_source', 'Source'], ['by_domain', 'Top domains']].forEach(function (part) {
    var d = s[part[0]] || {}, keys = Object.keys(d);
    if (!keys.length) return;
    box.appendChild(el('div', { cls: 'sec', text: part[1] }));
    hbars(box, keys.map(function (k) { return [k, d[k]]; }), null);
  });
})();

// ------------------------------------------------------------------ controls
$('seg-layout').addEventListener('click', function (e) { var v = e.target.dataset && e.target.dataset.v; if (v) setLayout(v); });
$('seg-color').addEventListener('click', function (e) { var v = e.target.dataset && e.target.dataset.v; if (v) setColorBy(v); });
var slider = $('min-cos');
var minLink = links.reduce(function (m, l) { return Math.min(m, l.cosine); }, 1);
slider.min = String(Math.max(-1, Math.floor(Math.min(state.minCos, minLink) * 100) / 100));
slider.value = String(state.minCos);
function onSlider() {
  state.minCos = parseFloat(slider.value); $('min-cos-v').textContent = fmt(state.minCos);
  rebuildLinks(); updateHistThr();
  if (state.layout === 'force') runSim(0.35); else draw();
  renderSide();
}
slider.addEventListener('input', onSlider);
$('q').addEventListener('input', function (e) {
  var q = e.target.value.trim().toLowerCase(); state.query = q;
  state.matches = q ? new Set(nodes.filter(function (n) { return n.text.indexOf(q) >= 0; })) : null;
  draw();
});
$('q').addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && state.matches && state.matches.size) {
    var best = Array.from(state.matches).sort(function (a, b) { return (b.risk_score || 0) - (a.risk_score || 0); })[0];
    select(best, true);
  }
});
var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
if (mq && mq.addEventListener) mq.addEventListener('change', function () { readTokens(); buildPalettes(); renderLegend(); renderSide(); draw(); });

// ------------------------------------------------------------------ boot
renderLegend(); rebuildLinks();
$('min-cos-v').textContent = fmt(state.minCos); updateHistThr(); renderSide();
if (!nodes.length) stage.appendChild(el('div', { cls: 'empty', text: 'No relevant silver documents yet — run `tcs-bot medallion run`.' }));
window.addEventListener('resize', function () { sizeStage(); resize(); });
if (window.ResizeObserver) new ResizeObserver(function () { resize(); }).observe(canvas);
sizeStage(); resize();
// settle most of the layout before the first frame so the initial view is stable
alpha = 1; var pre = nodes.length > 300 ? 60 : 140; for (var s = 0; s < pre && nodes.length; s++) tick();
fit(); runSim(alpha);
})();
</script>
</body>
</html>
"""  # noqa: E501
