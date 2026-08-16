"""Reporting: terminal, JSON, JUnit XML and a self-contained HTML dashboard.

The HTML report embeds its own CSS and hand-rendered SVG charts - no CDN, no
build step, no network access. That matters for a verification tool: the report
has to open from a CI artifact bundle or an air-gapped machine and still render.
"""

from __future__ import annotations

import html
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

from .types import EvalReport

__all__ = ["render_terminal", "write_json", "write_junit", "write_html", "ensure_parent"]

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def ensure_parent(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _sort_value(value: Any) -> str:
    """Numeric payload for client-side table sorting; NaN sorts last as -1."""
    if value is None:
        return "-1"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-1"
    return "-1" if math.isnan(number) else f"{number:.4f}"


def _pct(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Terminal
# --------------------------------------------------------------------------- #

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _c(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], color: bool = True) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [_c(line, _BOLD, color), "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def render_terminal(report: EvalReport, *, color: bool = True, top_errors: int = 5) -> str:
    """Human-readable summary for the console."""
    data = report.to_dict()
    ks = report.config.get("ks", [1, 3, 5, 10])
    lines: List[str] = []

    stats = report.corpus_stats
    lines.append(_c("RAG EVALUATION HARNESS", _BOLD, color))
    lines.append(
        f"dataset={stats.get('name')}  docs={stats.get('n_documents')}  "
        f"queries={stats.get('n_queries')}  answers={stats.get('n_answers')}  "
        f"retriever={report.config.get('retriever_name')}"
    )
    lines.append("")

    lines.append(_c("Retrieval quality (macro-averaged over queries)", _BOLD, color))
    headers = ["k", "precision", "recall", "F1", "hit rate", "MAP", "nDCG"]
    rows = []
    for k in ks:
        rows.append(
            [
                str(k),
                _fmt(report.retrieval.get(f"precision@{k}")),
                _fmt(report.retrieval.get(f"recall@{k}")),
                _fmt(report.retrieval.get(f"f1@{k}")),
                _fmt(report.retrieval.get(f"hit_rate@{k}")),
                _fmt(report.retrieval.get(f"map@{k}")),
                _fmt(report.retrieval.get(f"ndcg@{k}")),
            ]
        )
    lines.append(_table(headers, rows, color))
    mrr = report.retrieval.get("mrr")
    lat = report.retrieval.get("avg_latency_ms", 0.0)
    lines.append(f"MRR {_fmt(mrr)}   avg query latency {lat:.2f} ms")

    if report.retrieval_ci:
        lines.append("")
        lines.append(_c("95% bootstrap confidence intervals", _BOLD, color))
        for name in ("precision@5", "recall@5", "ndcg@5", "mrr"):
            if name in report.retrieval_ci:
                lo, hi = report.retrieval_ci[name]
                lines.append(
                    f"  {name:<12} {_fmt(report.retrieval.get(name))}  [{_fmt(lo)}, {_fmt(hi)}]"
                )

    if report.by_tag:
        lines.append("")
        lines.append(_c("By tag", _BOLD, color))
        tag_rows = [
            [
                tag,
                str(int(vals.get("n_queries", 0))),
                _fmt(vals.get("recall@5")),
                _fmt(vals.get("ndcg@5")),
            ]
            for tag, vals in report.by_tag.items()
        ]
        lines.append(_table(["tag", "n", "recall@5", "nDCG@5"], tag_rows, color))

    det = report.detector
    if det:
        lines.append("")
        lines.append(_c("Hallucination detection", _BOLD, color))
        lines.append(
            f"  answers={int(det.get('n_answers', 0))} "
            f"labeled={int(det.get('n_labeled', 0))} "
            f"claims={int(det.get('n_claims', 0))} "
            f"flagged claims={int(det.get('n_flagged_claims', 0))}"
        )
        lines.append(
            f"  mean faithfulness {_fmt(det.get('mean_faithfulness'))}   "
            f"mean risk {_fmt(det.get('mean_risk'))}   "
            f"flag rate {_pct(det.get('flag_rate'))}"
        )
        if "f1" in det:
            f1 = det.get("f1", 0.0)
            tone = _GREEN if f1 >= 0.8 else (_YELLOW if f1 >= 0.6 else _RED)
            lines.append(
                "  "
                + _c(
                    f"precision {_fmt(det.get('precision'))}  "
                    f"recall {_fmt(det.get('recall'))}  "
                    f"F1 {_fmt(f1)}  "
                    f"ROC-AUC {_fmt(det.get('roc_auc'))}",
                    tone,
                    color,
                )
            )
            lines.append(
                f"  confusion: TP={int(det.get('tp', 0))} FP={int(det.get('fp', 0))} "
                f"FN={int(det.get('fn', 0))} TN={int(det.get('tn', 0))}  "
                f"(threshold {_fmt(det.get('risk_threshold'), 2)}; "
                f"best F1 {_fmt(det.get('best_f1'))} at {_fmt(det.get('best_threshold'), 2)})"
            )
        if det.get("retrieval_starved_answers"):
            cond = det.get("given_correct_retrieval") or {}
            lines.append(
                f"  {int(det['retrieval_starved_answers'])} answer(s) had no gold evidence in "
                f"context; {int(det.get('flags_from_retrieval_miss', 0))} of the flags are "
                "retrieval failures, not fabrications"
            )
            if cond:
                lines.append(
                    f"  given correct retrieval (n={cond['n']}): precision {_fmt(cond['precision'])} "
                    f"recall {_fmt(cond['recall'])} F1 {_fmt(cond['f1'])}"
                )
        errors = det.get("errors") or []
        if errors:
            lines.append("")
            lines.append(_c(f"Detector errors ({len(errors)})", _BOLD, color))
            for err in errors[:top_errors]:
                cause = "  <- retrieval miss" if err.get("retrieval_starved") else ""
                lines.append(
                    f"  {err['query_id']:<10} {err['kind']:<15} "
                    f"risk={_fmt(err['risk'])} faithfulness={_fmt(err['faithfulness'])}{cause}"
                )

    worst = sorted(report.queries, key=lambda q: q.metrics.get(f"recall@{max(ks)}", 0.0))[:top_errors]
    if worst:
        lines.append("")
        lines.append(_c(f"Weakest retrieval queries (recall@{max(ks)})", _BOLD, color))
        for q in worst:
            lines.append(
                f"  {q.query_id:<10} {_fmt(q.metrics.get(f'recall@{max(ks)}'))}  "
                f"{q.question[:72]}"
            )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# JSON / JUnit
# --------------------------------------------------------------------------- #


def write_json(report: EvalReport, path: str) -> str:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def write_junit(report: EvalReport, path: str, gate: Optional[Mapping[str, Any]] = None) -> str:
    """Emit JUnit XML so any CI system renders the run as a test report.

    Each query becomes a test case (failing when nothing relevant was retrieved),
    each labelled answer becomes a test case (failing on a detector error), and
    each gate check becomes a test case.
    """
    ensure_parent(path)
    suites: List[str] = []

    cases: List[str] = []
    failures = 0
    for q in report.queries:
        hit = q.metrics.get("hit_rate@5", q.metrics.get("hit_rate@1", 0.0))
        name = xml_escape(f"{q.query_id}: {q.question[:80]}")
        if q.n_relevant and (hit is None or math.isnan(hit) or hit < 1.0):
            failures += 1
            cases.append(
                f'    <testcase classname="retrieval" name="{name}">\n'
                f'      <failure message="no relevant document in top-5">'
                f"recall@5={_fmt(q.metrics.get('recall@5'))}</failure>\n"
                f"    </testcase>"
            )
        else:
            cases.append(f'    <testcase classname="retrieval" name="{name}"/>')
    suites.append(
        f'  <testsuite name="retrieval" tests="{len(report.queries)}" failures="{failures}">\n'
        + "\n".join(cases)
        + "\n  </testsuite>"
    )

    ground_cases: List[str] = []
    ground_failures = 0
    for q in report.queries:
        g = q.grounding
        if g is None or g.gold_hallucinated is None:
            continue
        name = xml_escape(f"{q.query_id}: hallucination detection")
        if g.flagged != bool(g.gold_hallucinated):
            ground_failures += 1
            kind = "false negative" if g.gold_hallucinated else "false positive"
            ground_cases.append(
                f'    <testcase classname="grounding" name="{name}">\n'
                f'      <failure message="{kind}">risk={g.risk:.3f} '
                f"threshold={report.detector.get('risk_threshold')}</failure>\n"
                f"    </testcase>"
            )
        else:
            ground_cases.append(f'    <testcase classname="grounding" name="{name}"/>')
    if ground_cases:
        suites.append(
            f'  <testsuite name="grounding" tests="{len(ground_cases)}" '
            f'failures="{ground_failures}">\n' + "\n".join(ground_cases) + "\n  </testsuite>"
        )

    if gate:
        gate_cases = []
        gate_failures = 0
        for check in gate.get("checks", []):
            name = xml_escape(f"{check['kind']}: {check['name']}")
            if check["passed"]:
                gate_cases.append(f'    <testcase classname="gate" name="{name}"/>')
            else:
                gate_failures += 1
                gate_cases.append(
                    f'    <testcase classname="gate" name="{name}">\n'
                    f'      <failure message="{xml_escape(str(check.get("detail", "")))}">'
                    f"value={check['value']} reference={check['reference']}</failure>\n"
                    f"    </testcase>"
                )
        suites.append(
            f'  <testsuite name="gate" tests="{len(gate_cases)}" failures="{gate_failures}">\n'
            + "\n".join(gate_cases)
            + "\n  </testsuite>"
        )

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites>\n' + "\n".join(suites) + "\n</testsuites>\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)
    return path


# --------------------------------------------------------------------------- #
# SVG charting (hand-rolled, no dependencies)
# --------------------------------------------------------------------------- #


def _svg_grouped_bars(
    series: Sequence[Tuple[str, Sequence[float]]],
    labels: Sequence[str],
    *,
    width: int = 660,
    height: int = 260,
    title: str = "",
) -> str:
    pad_l, pad_r, pad_t, pad_b = 44, 12, 24, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n_groups = max(len(labels), 1)
    n_series = max(len(series), 1)
    group_w = plot_w / n_groups
    bar_w = min(group_w / (n_series + 0.8), 34)
    palette = ["#4f8ef7", "#37c99a", "#f5a524", "#e5484d", "#a06bf0"]

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i in range(5):
        y = pad_t + plot_h * i / 4
        value = 1.0 - i / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">{value:.2f}</text>')

    for g, label in enumerate(labels):
        gx = pad_l + group_w * g
        for s, (name, values) in enumerate(series):
            value = values[g] if g < len(values) else 0.0
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = 0.0
            bar_h = max(plot_h * float(value), 0.0)
            x = gx + (group_w - bar_w * n_series) / 2 + s * bar_w
            y = pad_t + plot_h - bar_h
            color = palette[s % len(palette)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" '
                f'height="{bar_h:.1f}" fill="{color}" rx="2"><title>{html.escape(name)} '
                f'{html.escape(label)}: {value:.3f}</title></rect>'
            )
        parts.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{height - pad_b + 18}" class="axis" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
        f'y2="{pad_t + plot_h}" class="axis-line"/>'
    )
    parts.append("</svg>")
    legend = " ".join(
        f'<span class="key"><i style="background:{palette[i % len(palette)]}"></i>'
        f"{html.escape(name)}</span>"
        for i, (name, _) in enumerate(series)
    )
    return f'<div class="chart">{"".join(parts)}</div><div class="legend">{legend}</div>'


def _svg_lines(
    series: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    *,
    width: int = 660,
    height: int = 260,
    x_label: str = "",
    title: str = "",
) -> str:
    pad_l, pad_r, pad_t, pad_b = 44, 12, 24, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xs = [x for _, pts in series for x, _ in pts] or [0.0, 1.0]
    x_min, x_max = min(xs), max(xs)
    span = (x_max - x_min) or 1.0
    palette = ["#4f8ef7", "#37c99a", "#f5a524", "#e5484d", "#a06bf0"]

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i in range(5):
        y = pad_t + plot_h * i / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">{1.0 - i / 4:.2f}</text>'
        )
    for i in range(5):
        x = pad_l + plot_w * i / 4
        value = x_min + span * i / 4
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 18}" class="axis" text-anchor="middle">{value:.2f}</text>'
        )

    for s, (name, points) in enumerate(series):
        if not points:
            continue
        color = palette[s % len(palette)]
        coords = []
        for x, y in points:
            if y is None or (isinstance(y, float) and math.isnan(y)):
                y = 0.0
            px = pad_l + plot_w * ((x - x_min) / span)
            py = pad_t + plot_h * (1.0 - min(max(float(y), 0.0), 1.0))
            coords.append(f"{px:.1f},{py:.1f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" class="axis-line"/>'
    )
    if x_label:
        parts.append(
            f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 4}" class="axis" '
            f'text-anchor="middle">{html.escape(x_label)}</text>'
        )
    parts.append("</svg>")
    legend = " ".join(
        f'<span class="key"><i style="background:{palette[i % len(palette)]}"></i>{html.escape(name)}</span>'
        for i, (name, _) in enumerate(series)
    )
    return f'<div class="chart">{"".join(parts)}</div><div class="legend">{legend}</div>'


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

_CSS = """
:root{--bg:#0f1419;--panel:#161c24;--panel2:#1c242e;--fg:#e6edf3;--muted:#8b98a5;
--line:#232c38;--good:#37c99a;--warn:#f5a524;--bad:#e5484d;--accent:#4f8ef7;}
@media (prefers-color-scheme: light){:root{--bg:#f6f8fa;--panel:#fff;--panel2:#f0f3f6;
--fg:#1c2128;--muted:#5b6672;--line:#d8dee4;}}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:32px 0 12px;letter-spacing:.02em}
.sub{color:var(--muted);font-size:13px;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
.card .value{font-size:26px;font-weight:600;margin-top:6px;font-variant-numeric:tabular-nums}
.card .ci{color:var(--muted);font-size:11px;margin-top:2px;font-variant-numeric:tabular-nums}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);font-size:13px}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--fg)}
tbody tr:hover{background:var(--panel2)}
.num{text-align:right}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.ok{background:rgba(55,201,154,.15);color:var(--good)}
.bad{background:rgba(229,72,77,.15);color:var(--bad)}
.warn{background:rgba(245,165,36,.15);color:var(--warn)}
.chart{background:var(--panel2);border-radius:8px;padding:6px}
.chart svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.axis-line{stroke:var(--muted);stroke-width:1}
text.axis{fill:var(--muted);font-size:10px}
.legend{margin-top:8px;color:var(--muted);font-size:12px}
.key{margin-right:14px;white-space:nowrap}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.claim{border-left:3px solid var(--line);padding:8px 12px;margin:10px 0;background:var(--panel2);border-radius:0 6px 6px 0}
.claim.contradicted{border-left-color:var(--bad)}
.claim.unsupported{border-left-color:var(--warn)}
.claim .txt{font-weight:500}
.claim .why{color:var(--muted);font-size:12px;margin-top:4px}
.claim .ev{color:var(--muted);font-size:12px;margin-top:4px;font-style:italic}
.q{color:var(--muted);font-size:12px;margin-bottom:6px}
input[type=search]{background:var(--panel2);border:1px solid var(--line);color:var(--fg);
border-radius:7px;padding:7px 11px;width:290px;font-size:13px}
.confusion{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;max-width:320px}
.cell{background:var(--panel2);border-radius:8px;padding:12px;text-align:center}
.cell .n{font-size:22px;font-weight:700}
.cell .t{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
footer{color:var(--muted);font-size:12px;margin-top:36px;border-top:1px solid var(--line);padding-top:14px}
code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:12px}
"""

_JS = """
document.querySelectorAll('table.sortable').forEach(function(table){
  table.querySelectorAll('th').forEach(function(th, idx){
    th.addEventListener('click', function(){
      var tbody = table.tBodies[0];
      var rows = Array.prototype.slice.call(tbody.rows);
      var asc = th.dataset.asc !== 'true';
      table.querySelectorAll('th').forEach(function(o){ o.dataset.asc = ''; });
      th.dataset.asc = asc ? 'true' : 'false';
      rows.sort(function(a, b){
        var x = a.cells[idx].dataset.v !== undefined ? a.cells[idx].dataset.v : a.cells[idx].innerText;
        var y = b.cells[idx].dataset.v !== undefined ? b.cells[idx].dataset.v : b.cells[idx].innerText;
        var nx = parseFloat(x), ny = parseFloat(y);
        if (!isNaN(nx) && !isNaN(ny)) { return asc ? nx - ny : ny - nx; }
        return asc ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
      });
      rows.forEach(function(r){ tbody.appendChild(r); });
    });
  });
});
var filter = document.getElementById('q-filter');
if (filter) {
  filter.addEventListener('input', function(){
    var term = filter.value.toLowerCase();
    document.querySelectorAll('#query-table tbody tr').forEach(function(row){
      row.style.display = row.innerText.toLowerCase().indexOf(term) === -1 ? 'none' : '';
    });
  });
}
"""


def _card(label: str, value: str, ci: str = "") -> str:
    ci_html = f'<div class="ci">{html.escape(ci)}</div>' if ci else ""
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>{ci_html}</div>'
    )


def write_html(report: EvalReport, path: str, *, gate: Optional[Mapping[str, Any]] = None) -> str:
    """Render the full dashboard to a single self-contained HTML file."""
    ensure_parent(path)
    ks = [int(k) for k in report.config.get("ks", [1, 3, 5, 10])]
    stats = report.corpus_stats
    det = report.detector or {}
    body: List[str] = []

    # -- header + headline cards
    body.append(f"<h1>RAG Evaluation Report</h1>")
    body.append(
        f'<div class="sub">dataset <code>{html.escape(str(stats.get("name", "")))}</code> · '
        f'{stats.get("n_documents", 0)} documents · {stats.get("n_queries", 0)} labelled queries · '
        f'{stats.get("n_answers", 0)} answers under test · retriever '
        f'<code>{html.escape(str(report.config.get("retriever_name", "")))}</code> · '
        f"generated {html.escape(report.generated_at)}</div>"
    )

    def ci_text(name: str) -> str:
        pair = report.retrieval_ci.get(name)
        if not pair:
            return ""
        return f"95% CI [{pair[0]:.3f}, {pair[1]:.3f}]"

    cards = [
        _card("Recall@5", _fmt(report.retrieval.get("recall@5")), ci_text("recall@5")),
        _card("Precision@5", _fmt(report.retrieval.get("precision@5")), ci_text("precision@5")),
        _card("nDCG@5", _fmt(report.retrieval.get("ndcg@5")), ci_text("ndcg@5")),
        _card("MRR", _fmt(report.retrieval.get("mrr")), ci_text("mrr")),
    ]
    if det:
        cards.append(_card("Detector F1", _fmt(det.get("f1")), f"ROC-AUC {_fmt(det.get('roc_auc'))}"))
        cards.append(
            _card("Mean faithfulness", _fmt(det.get("mean_faithfulness")), f"flag rate {_pct(det.get('flag_rate'))}")
        )
    body.append(f'<div class="cards">{"".join(cards)}</div>')

    if gate:
        passed = gate.get("passed", True)
        pill = '<span class="pill ok">PASS</span>' if passed else '<span class="pill bad">FAIL</span>'
        ok_pill = '<span class="pill ok">ok</span>'
        fail_pill = '<span class="pill bad">fail</span>'
        rows = "".join(
            f'<tr><td>{html.escape(c["name"])}</td><td>{html.escape(c["kind"])}</td>'
            f'<td class="num">{_fmt(c["value"])}</td><td class="num">{_fmt(c["reference"])}</td>'
            f'<td>{ok_pill if c["passed"] else fail_pill}'
            f' {html.escape(str(c.get("detail", "")))}</td></tr>'
            for c in gate.get("checks", [])
        )
        body.append(f"<h2>Quality gate {pill}</h2>")
        body.append(
            '<div class="panel"><table><thead><tr><th>metric</th><th>kind</th>'
            "<th class='num'>value</th><th class='num'>reference</th><th>result</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    # -- retrieval charts
    body.append("<h2>Retrieval quality</h2>")
    labels = [f"k={k}" for k in ks]
    series = [
        ("precision@k", [report.retrieval.get(f"precision@{k}", 0.0) for k in ks]),
        ("recall@k", [report.retrieval.get(f"recall@{k}", 0.0) for k in ks]),
        ("nDCG@k", [report.retrieval.get(f"ndcg@{k}", 0.0) for k in ks]),
        ("hit rate@k", [report.retrieval.get(f"hit_rate@{k}", 0.0) for k in ks]),
    ]
    pr_curve = [
        (
            "precision vs recall across k",
            [
                (report.retrieval.get(f"recall@{k}", 0.0), report.retrieval.get(f"precision@{k}", 0.0))
                for k in ks
            ],
        )
    ]
    body.append(
        '<div class="grid2">'
        f'<div class="panel">{_svg_grouped_bars(series, labels, title="Retrieval metrics by k")}</div>'
        f'<div class="panel">{_svg_lines(pr_curve, x_label="recall", title="Precision-recall trade-off")}</div>'
        "</div>"
    )

    metric_rows = "".join(
        "<tr><td>k={k}</td>".format(k=k)
        + "".join(
            f'<td class="num">{_fmt(report.retrieval.get(f"{m}@{k}"))}</td>'
            for m in ("precision", "recall", "f1", "hit_rate", "map", "ndcg")
        )
        + "</tr>"
        for k in ks
    )
    body.append(
        '<div class="panel" style="margin-top:16px"><table><thead><tr><th>cut-off</th>'
        "<th class='num'>precision</th><th class='num'>recall</th><th class='num'>F1</th>"
        "<th class='num'>hit rate</th><th class='num'>MAP</th><th class='num'>nDCG</th>"
        f"</tr></thead><tbody>{metric_rows}</tbody></table>"
        f'<div class="sub" style="margin:10px 0 0">MRR {_fmt(report.retrieval.get("mrr"))} · '
        f'mean query latency {report.retrieval.get("avg_latency_ms", 0.0):.2f} ms</div></div>'
    )

    # -- by tag
    if report.by_tag:
        tag_rows = "".join(
            f'<tr><td>{html.escape(tag)}</td><td class="num">{int(v.get("n_queries", 0))}</td>'
            f'<td class="num">{_fmt(v.get("precision@5"))}</td>'
            f'<td class="num">{_fmt(v.get("recall@5"))}</td>'
            f'<td class="num">{_fmt(v.get("ndcg@5"))}</td>'
            f'<td class="num">{_fmt(v.get("mrr"))}</td></tr>'
            for tag, v in report.by_tag.items()
        )
        body.append("<h2>Breakdown by domain</h2>")
        body.append(
            '<div class="panel"><table class="sortable"><thead><tr><th>tag</th>'
            "<th class='num'>queries</th><th class='num'>precision@5</th>"
            "<th class='num'>recall@5</th><th class='num'>nDCG@5</th><th class='num'>MRR</th>"
            f"</tr></thead><tbody>{tag_rows}</tbody></table></div>"
        )

    # -- detector
    if det and "f1" in det:
        sweep = det.get("sweep") or []
        sweep_series = [
            ("precision", [(row["threshold"], row["precision"]) for row in sweep]),
            ("recall", [(row["threshold"], row["recall"]) for row in sweep]),
            ("F1", [(row["threshold"], row["f1"]) for row in sweep]),
        ]
        confusion = (
            '<div class="confusion">'
            f'<div class="cell"><div class="n">{int(det.get("tp", 0))}</div><div class="t">true positive</div></div>'
            f'<div class="cell"><div class="n">{int(det.get("fp", 0))}</div><div class="t">false positive</div></div>'
            f'<div class="cell"><div class="n">{int(det.get("fn", 0))}</div><div class="t">false negative</div></div>'
            f'<div class="cell"><div class="n">{int(det.get("tn", 0))}</div><div class="t">true negative</div></div>'
            "</div>"
            f'<div class="sub" style="margin-top:12px">precision {_fmt(det.get("precision"))} · '
            f'recall {_fmt(det.get("recall"))} · F1 {_fmt(det.get("f1"))} · '
            f'ROC-AUC {_fmt(det.get("roc_auc"))} · balanced acc {_fmt(det.get("balanced_accuracy"))}<br>'
            f'operating threshold {_fmt(det.get("risk_threshold"), 2)} · '
            f'best F1 {_fmt(det.get("best_f1"))} at threshold {_fmt(det.get("best_threshold"), 2)}</div>'
        )
        cond = det.get("given_correct_retrieval")
        if cond:
            confusion += (
                f'<div class="sub" style="margin-top:10px">Root cause: '
                f'{int(det.get("retrieval_starved_answers", 0))} answer(s) received no gold '
                f'evidence, producing {int(det.get("flags_from_retrieval_miss", 0))} flag(s) that '
                f'are retrieval failures rather than fabrications. Restricted to answers whose '
                f'context did contain the gold evidence (n={cond["n"]}): precision '
                f'{_fmt(cond["precision"])}, recall {_fmt(cond["recall"])}, F1 {_fmt(cond["f1"])}.</div>'
            )
        body.append("<h2>Hallucination detection</h2>")
        body.append(
            '<div class="grid2">'
            f'<div class="panel">{confusion}</div>'
            f'<div class="panel">{_svg_lines(sweep_series, x_label="risk threshold", title="Detector threshold sweep")}</div>'
            "</div>"
        )

    # -- flagged claims
    flagged_blocks: List[str] = []
    for q in report.queries:
        g = q.grounding
        if not g or not g.flagged_claims:
            continue
        gold = g.gold_hallucinated
        tag = ""
        if gold is not None:
            correct = g.flagged == bool(gold)
            tag = (
                f'<span class="pill {"ok" if correct else "bad"}">'
                f'{"true positive" if (correct and gold) else "false positive" if not gold and g.flagged else "correct" if correct else "miss"}'
                "</span>"
            )
        claims = "".join(
            f'<div class="claim {html.escape(c.verdict)}">'
            f'<div class="txt">{html.escape(c.claim)}</div>'
            f'<div class="why">{html.escape(c.verdict)} · support {c.support:.2f} · '
            f'{html.escape("; ".join(c.reasons))}</div>'
            + (
                f'<div class="ev">nearest evidence [{html.escape(c.evidence_doc_id)}]: '
                f"{html.escape(c.evidence[:240])}</div>"
                if c.evidence
                else ""
            )
            + "</div>"
            for c in g.flagged_claims
        )
        flagged_blocks.append(
            f'<div class="panel" style="margin-bottom:14px">'
            f'<div class="q">{html.escape(q.query_id)} · {html.escape(q.question)} · '
            f"risk {g.risk:.2f} · faithfulness {g.faithfulness:.2f} {tag}</div>{claims}</div>"
        )
    if flagged_blocks:
        body.append(f"<h2>Flagged claims ({len(flagged_blocks)} answers)</h2>")
        body.extend(flagged_blocks)

    # -- per-query table
    body.append("<h2>Per-query results</h2>")
    body.append(
        '<div class="panel"><input id="q-filter" type="search" placeholder="filter queries…" '
        'aria-label="filter queries"/>'
        '<table class="sortable" id="query-table" style="margin-top:12px"><thead><tr>'
        "<th>id</th><th>question</th><th class='num'>rel</th><th class='num'>P@5</th>"
        "<th class='num'>R@5</th><th class='num'>nDCG@5</th><th class='num'>RR</th>"
        "<th class='num'>risk</th><th>flag</th></tr></thead><tbody>"
    )
    for q in report.queries:
        g = q.grounding
        risk = "" if g is None else f"{g.risk:.2f}"
        if g is None:
            flag = "-"
        elif g.gold_hallucinated is None:
            flag = '<span class="pill warn">flagged</span>' if g.flagged else "ok"
        else:
            correct = g.flagged == bool(g.gold_hallucinated)
            label = "flagged" if g.flagged else "clean"
            flag = f'<span class="pill {"ok" if correct else "bad"}">{label}</span>'
        body.append(
            f"<tr><td>{html.escape(q.query_id)}</td>"
            f"<td>{html.escape(q.question)}</td>"
            f'<td class="num" data-v="{q.n_relevant}">{q.n_relevant}</td>'
            + "".join(
                f'<td class="num" data-v="{_sort_value(q.metrics.get(m))}">{_fmt(q.metrics.get(m))}</td>'
                for m in ("precision@5", "recall@5", "ndcg@5", "mrr")
            )
            + f'<td class="num" data-v="{risk or 0}">{risk or "-"}</td><td>{flag}</td></tr>'
        )
    body.append("</tbody></table></div>")

    body.append(
        '<footer>Generated by <code>rag-eval-harness</code> v'
        f"{html.escape(report.version)} · macro-averaged over {len(report.queries)} queries · "
        "confidence intervals are 1000-sample percentile bootstraps · "
        "all metrics are deterministic and reproducible from the committed dataset.</footer>"
    )

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>RAG Evaluation Report</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        + "".join(body)
        + f"</div><script>{_JS}</script></body></html>"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return path
