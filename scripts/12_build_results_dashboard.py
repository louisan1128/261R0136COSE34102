import csv
import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import ensure_dir, read_yaml


DOCS_DIR = root / "docs"
DASHBOARD_PATH = DOCS_DIR / "results_dashboard.html"
SUMMARY_PATH = DOCS_DIR / "results_summary.md"


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data = config["data"]
    outputs = {
        "original": root / data["original_results_path"],
        "hard_cases": root / data["hard_cases_path"],
        "sample": root / config.get("llm_rewrite", {}).get("sample_path", "data/outputs/hard_cases_random_sample.jsonl"),
        "cache": root / config.get("llm_rewrite", {}).get("cache_path", "data/outputs/llm_rewrite_cache.jsonl"),
        "candidates": root / data["rewrite_candidates_path"],
        "rewrite_results": root / data["rewrite_results_path"],
        "main": root / data["main_results_path"],
        "recovery": root / data["recovery_path"],
        "failure": root / data["failure_type_analysis_path"],
        "alpha": root / data["hybrid_alpha_sweep_path"],
        "policy_summary": root / data["policy_summary_path"],
        "final": root / data.get("final_comparison_path", "data/outputs/final_policy_comparison.csv"),
    }

    rows = {name: read_csv(path) for name, path in outputs.items() if path.suffix == ".csv"}
    counts = {
        "total_questions": _question_count(rows.get("original", [])),
        "hard_cases": count_jsonl(outputs["hard_cases"]),
        "sampled_hard_cases": count_jsonl(outputs["sample"]),
        "rewrite_candidates": count_jsonl(outputs["candidates"]),
        "rewrite_results": count_jsonl(outputs["rewrite_results"]),
        "llm_cache": count_jsonl(outputs["cache"]),
    }
    insights = build_insights(rows, counts)

    ensure_dir(DOCS_DIR)
    DASHBOARD_PATH.write_text(build_dashboard(rows, counts, insights), encoding="utf-8")
    SUMMARY_PATH.write_text(build_markdown_summary(rows, counts, insights), encoding="utf-8")
    print(f"Saved dashboard to {DASHBOARD_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fin:
        return sum(1 for line in fin if line.strip())


def build_insights(rows: dict[str, list[dict]], counts: dict[str, int]) -> dict:
    original = rows.get("original", [])
    main = rows.get("main", [])
    final = rows.get("final", [])
    recovery = rows.get("recovery", [])
    alpha = rows.get("alpha", [])

    best_original = max(original, key=lambda row: fnum(row, "recall@10"), default={})
    best_strategy = {
        retriever: max(items, key=lambda row: fnum(row, "recall@10"))
        for retriever, items in group_by(main, "retriever").items()
    }
    best_final = {
        retriever: max(items, key=lambda row: fnum(row, "recall@10"))
        for retriever, items in group_by(final, "retriever").items()
    }
    best_alpha = max(alpha, key=lambda row: fnum(row, "mrr"), default={})
    retriever_failed_recovery = {
        row["retriever"]: row
        for row in recovery
        if row.get("case_subset") == "retriever_originally_failed"
    }
    return {
        "best_original": best_original,
        "best_strategy": best_strategy,
        "best_final": best_final,
        "best_alpha": best_alpha,
        "retriever_failed_recovery": retriever_failed_recovery,
        "consistency": consistency_checks(counts),
    }


def consistency_checks(counts: dict[str, int]) -> list[tuple[str, str]]:
    checks = []
    expected_rewrite_rows = counts["rewrite_candidates"] * 6 * 3
    if counts["rewrite_results"] == expected_rewrite_rows:
        checks.append(("OK", f"rewrite_results rows match candidates x 6 strategies x 3 retrievers ({expected_rewrite_rows})."))
    else:
        checks.append(("WARN", f"rewrite_results rows={counts['rewrite_results']}, expected={expected_rewrite_rows}."))
    if counts["sampled_hard_cases"] == counts["rewrite_candidates"]:
        checks.append(("OK", "sampled hard cases and rewrite candidates use the same count."))
    else:
        checks.append(("WARN", "sampled hard case count and rewrite candidate count differ."))
    if counts["llm_cache"] >= counts["rewrite_candidates"]:
        checks.append(("OK", "LLM cache has at least as many entries as current rewrite candidates."))
    else:
        checks.append(("WARN", "LLM cache has fewer entries than current rewrite candidates."))
    return checks


def build_dashboard(rows: dict[str, list[dict]], counts: dict[str, int], insights: dict) -> str:
    original = rows.get("original", [])
    main = rows.get("main", [])
    final = rows.get("final", [])
    recovery = rows.get("recovery", [])
    alpha = rows.get("alpha", [])
    failure = rows.get("failure", [])

    sections = [
        cards(counts),
        section("Current Verdict", verdict_html(insights)),
        section("Original Retrieval", bar_chart(original, "retriever", "recall@10", title="Original Recall@10") + table_html(original)),
        section("Hard Case Recovery", bar_chart(recovery, "retriever", "recovery@10", title="Recovery@10 by subset", series_key="case_subset") + table_html(recovery)),
        section("Rewrite Strategy Results", grouped_strategy_chart(main) + table_html(main)),
        section("Final Policy Comparison", grouped_final_chart(final) + table_html(final)),
        section("Hybrid Alpha Sweep", line_chart(alpha, "hybrid_alpha", ["recall@10", "mrr"], title="Hybrid alpha sweep") + table_html(alpha)),
        section("Failure-Type Analysis", top_failure_table(failure)),
    ]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KorQR-RL Results Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f7f9; }}
    header {{ padding: 28px 36px; background: #18202a; color: white; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 16px; font-size: 20px; }}
    main {{ padding: 24px 36px 48px; }}
    section {{ margin: 0 0 24px; padding: 20px; background: white; border: 1px solid #d8dee8; border-radius: 8px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card {{ padding: 16px; background: white; border: 1px solid #d8dee8; border-radius: 8px; }}
    .label {{ color: #52616f; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; align-items: start; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #e3e8ef; padding: 7px 8px; text-align: left; }}
    th {{ background: #f0f3f7; font-weight: 700; }}
    .note {{ line-height: 1.55; margin: 0; }}
    .ok {{ color: #13795b; font-weight: 700; }}
    .warn {{ color: #b42318; font-weight: 700; }}
    svg {{ width: 100%; height: auto; border: 1px solid #e3e8ef; background: #fbfcfe; }}
    .small {{ color: #52616f; font-size: 12px; }}
  </style>
</head>
<body>
<header>
  <h1>KorQR-RL Results Dashboard</h1>
  <div>One-page view of retrieval, rewriting, reward selection, and lightweight RL policy results.</div>
</header>
<main>
  {''.join(sections)}
</main>
</body>
</html>
"""


def cards(counts: dict[str, int]) -> str:
    labels = [
        ("Total QA", "total_questions"),
        ("Hard cases", "hard_cases"),
        ("Random subset", "sampled_hard_cases"),
        ("Rewrite candidates", "rewrite_candidates"),
        ("Rewrite eval rows", "rewrite_results"),
        ("LLM cache", "llm_cache"),
    ]
    body = "".join(
        f"<div class='card'><div class='label'>{escape(label)}</div><div class='value'>{counts[key]:,}</div></div>"
        for label, key in labels
    )
    return f"<div class='cards'>{body}</div>"


def verdict_html(insights: dict) -> str:
    best_original = insights["best_original"]
    best_alpha = insights["best_alpha"]
    parts = [
        f"<p class='note'><b>Best original retriever:</b> {escape(best_original.get('retriever', 'n/a'))} "
        f"(Recall@10={pct(fnum(best_original, 'recall@10'))}, MRR={fmt(fnum(best_original, 'mrr'))}).</p>",
        f"<p class='note'><b>Best hybrid alpha by MRR:</b> alpha={escape(best_alpha.get('hybrid_alpha', 'n/a'))} "
        f"(Recall@10={pct(fnum(best_alpha, 'recall@10'))}, MRR={fmt(fnum(best_alpha, 'mrr'))}).</p>",
    ]
    for retriever, row in insights["best_final"].items():
        parts.append(
            f"<p class='note'><b>Final test winner for {escape(retriever)}:</b> "
            f"{escape(row.get('comparison_method', 'n/a'))} "
            f"(Recall@10={pct(fnum(row, 'recall@10'))}, MRR={fmt(fnum(row, 'mrr'))}).</p>"
        )
    parts.append("<ul>" + "".join(
        f"<li><span class='{status.lower()}'>{status}</span> {escape(text)}</li>"
        for status, text in insights["consistency"]
    ) + "</ul>")
    return "".join(parts)


def section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def bar_chart(rows: list[dict], label_key: str, value_key: str, title: str, series_key: str | None = None) -> str:
    if not rows:
        return "<p class='small'>No data.</p>"
    chart_rows = rows
    max_value = max((fnum(row, value_key) for row in chart_rows), default=1.0) or 1.0
    width = 820
    row_h = 26
    height = 58 + row_h * len(chart_rows)
    bars = [f"<text x='20' y='24' font-size='14' font-weight='700'>{escape(title)}</text>"]
    for idx, row in enumerate(chart_rows):
        y = 44 + idx * row_h
        label = row.get(label_key, "")
        if series_key:
            label = f"{row.get(series_key, '')} / {label}"
        value = fnum(row, value_key)
        bar_w = int((value / max_value) * 520)
        color = color_for(row.get(label_key, "") + row.get(series_key, ""))
        bars.append(f"<text x='20' y='{y + 15}' font-size='11'>{escape(label)}</text>")
        bars.append(f"<rect x='260' y='{y}' width='{bar_w}' height='18' fill='{color}' rx='3'></rect>")
        bars.append(f"<text x='{265 + bar_w}' y='{y + 14}' font-size='11'>{fmt(value)}</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(bars)}</svg>"


def grouped_strategy_chart(rows: list[dict]) -> str:
    top_rows = sorted(rows, key=lambda row: (row.get("retriever", ""), -fnum(row, "recall@10")))
    return bar_chart(top_rows, "strategy", "recall@10", "Rewrite strategy Recall@10", series_key="retriever")


def grouped_final_chart(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda row: (row.get("retriever", ""), -fnum(row, "recall@10")))
    return bar_chart(ordered, "comparison_method", "recall@10", "Final test Recall@10", series_key="retriever")


def line_chart(rows: list[dict], x_key: str, y_keys: list[str], title: str) -> str:
    if not rows:
        return "<p class='small'>No data.</p>"
    points = sorted(rows, key=lambda row: fnum(row, x_key))
    width, height = 820, 300
    x0, y0, plot_w, plot_h = 60, 36, 700, 210
    xs = [fnum(row, x_key) for row in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, 1.0
    elements = [f"<text x='20' y='24' font-size='14' font-weight='700'>{escape(title)}</text>"]
    elements.append(f"<line x1='{x0}' y1='{y0 + plot_h}' x2='{x0 + plot_w}' y2='{y0 + plot_h}' stroke='#8b98a8'/>")
    elements.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0 + plot_h}' stroke='#8b98a8'/>")
    colors = ["#276ef1", "#d97706", "#13795b"]
    for key_idx, y_key in enumerate(y_keys):
        coords = []
        for row in points:
            x = scale(fnum(row, x_key), min_x, max_x, x0, x0 + plot_w)
            y = scale(fnum(row, y_key), min_y, max_y, y0 + plot_h, y0)
            coords.append(f"{x:.1f},{y:.1f}")
        color = colors[key_idx % len(colors)]
        elements.append(f"<polyline points='{' '.join(coords)}' fill='none' stroke='{color}' stroke-width='3'/>")
        elements.append(f"<text x='{x0 + key_idx * 150}' y='{height - 22}' fill='{color}' font-size='12'>{escape(y_key)}</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(elements)}</svg>"


def top_failure_table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='small'>No data.</p>"
    selected = sorted(rows, key=lambda row: (-fnum(row, "reward"), row.get("failure_type", "")))[:24]
    return table_html(selected)


def table_html(rows: list[dict], max_rows: int = 40) -> str:
    if not rows:
        return "<p class='small'>No data.</p>"
    shown = rows[:max_rows]
    headers = list(shown[0].keys())
    thead = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{escape(format_cell(row.get(header, '')))}</td>" for header in headers) + "</tr>")
    suffix = "" if len(rows) <= max_rows else f"<p class='small'>Showing {max_rows} of {len(rows)} rows.</p>"
    return f"<div style='overflow-x:auto'><table><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{suffix}"


def build_markdown_summary(rows: dict[str, list[dict]], counts: dict[str, int], insights: dict) -> str:
    lines = [
        "# Results Summary",
        "",
        f"- Total QA pairs: {counts['total_questions']:,}",
        f"- Union hard cases: {counts['hard_cases']:,}",
        f"- Random hard subset: {counts['sampled_hard_cases']:,}",
        f"- Rewrite candidates: {counts['rewrite_candidates']:,}",
        f"- Rewrite evaluation rows: {counts['rewrite_results']:,}",
        f"- LLM rewrite cache entries: {counts['llm_cache']:,}",
        "",
        "## Main Takeaways",
        "",
    ]
    best_original = insights["best_original"]
    lines.append(
        f"- Best original retriever: `{best_original.get('retriever', 'n/a')}` "
        f"with Recall@10 {pct(fnum(best_original, 'recall@10'))} and MRR {fmt(fnum(best_original, 'mrr'))}."
    )
    best_alpha = insights["best_alpha"]
    lines.append(
        f"- Best hybrid alpha by MRR: `{best_alpha.get('hybrid_alpha', 'n/a')}` "
        f"with MRR {fmt(fnum(best_alpha, 'mrr'))}."
    )
    for retriever, row in insights["best_final"].items():
        lines.append(
            f"- Final test winner for `{retriever}`: `{row.get('comparison_method', 'n/a')}` "
            f"(Recall@10 {pct(fnum(row, 'recall@10'))}, MRR {fmt(fnum(row, 'mrr'))})."
        )
    lines.extend(["", "## Consistency Checks", ""])
    for status, text in insights["consistency"]:
        lines.append(f"- {status}: {text}")
    lines.extend(["", f"HTML dashboard: `{DASHBOARD_PATH.relative_to(root)}`", ""])
    return "\n".join(lines)


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return dict(grouped)


def _question_count(rows: list[dict]) -> int:
    if not rows:
        return 0
    return int(float(rows[0].get("num_questions", 0) or 0))


def fnum(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def scale(value: float, min_value: float, max_value: float, out_min: float, out_max: float) -> float:
    if max_value == min_value:
        return (out_min + out_max) / 2
    return out_min + (value - min_value) * (out_max - out_min) / (max_value - min_value)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def format_cell(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) <= 1:
        return f"{number:.4f}"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}"


def color_for(label: str) -> str:
    palette = ["#276ef1", "#13795b", "#d97706", "#7c3aed", "#c2410c", "#0f766e", "#b91c1c"]
    return palette[sum(ord(ch) for ch in label) % len(palette)]


def escape(value: object) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()
