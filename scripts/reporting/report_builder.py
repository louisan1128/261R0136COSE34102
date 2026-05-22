import os
import sys
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.append(str(root))

from src.retrievers.factory import build_retrievers
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_csv
from src.utils.text import extract_keywords, tokenize


REPORT_DIR = root / "reports"
QUALITATIVE_CSV = REPORT_DIR / "qualitative_examples.csv"
QUALITATIVE_MD = root / "docs" / "qualitative_examples.md"
MANUAL_CHECK_CSV = REPORT_DIR / "failure_type_manual_check.csv"
MANUAL_CHECK_MD = root / "docs" / "failure_type_manual_check.md"


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])

    example_rows = build_qualitative_examples(config, hard_cases, rewrite_results)
    write_csv(example_rows, QUALITATIVE_CSV, fieldnames=_qualitative_fields())
    write_markdown_examples(example_rows, QUALITATIVE_MD)

    manual_rows = build_failure_type_manual_check(hard_cases, sample_size=100)
    write_csv(manual_rows, MANUAL_CHECK_CSV, fieldnames=_manual_check_fields())
    write_markdown_manual_check(manual_rows, MANUAL_CHECK_MD)

    print(f"Saved qualitative examples to {QUALITATIVE_CSV}")
    print(f"Saved qualitative markdown to {QUALITATIVE_MD}")
    print(f"Saved failure-type manual check sheet to {MANUAL_CHECK_CSV}")
    print(f"Saved failure-type manual check preview to {MANUAL_CHECK_MD}")


def build_qualitative_examples(config: dict, hard_cases: list[dict], rewrite_results: list[dict]) -> list[dict]:
    records_by_key = _index_rewrite_results(rewrite_results)
    hard_by_qid = {record["qid"]: record for record in hard_cases}
    candidates = []

    for hard_case in hard_cases:
        qid = hard_case["qid"]
        for retriever in _failed_retrievers(hard_case):
            strategy_records = records_by_key.get((qid, retriever), {})
            if not strategy_records:
                continue
            original = strategy_records.get("original")
            best = max(strategy_records.values(), key=lambda record: float(record.get("reward", 0.0)))
            if not original or float(original.get("recall@10", 0.0)) >= 1.0:
                continue

            category = "recovered" if float(best.get("recall@10", 0.0)) >= 1.0 else "not_recovered"
            candidates.append(
                {
                    "category": category,
                    "qid": qid,
                    "retriever": retriever,
                    "best": best,
                    "hard_case": hard_by_qid[qid],
                    "sort_key": (
                        category != "recovered",
                        retriever,
                        -float(best.get("reward", 0.0)),
                        qid,
                    ),
                }
            )

    selected = _take_diverse_examples(candidates, max_recovered=12, max_not_recovered=6)
    if not selected:
        return []

    retrievers = build_retrievers(config)

    rows = []
    for item in selected:
        hard_case = item["hard_case"]
        record = item["best"]
        retriever_name = item["retriever"]
        retrieved = retrievers[retriever_name].retrieve(record["query"], top_k=10)
        gold_rank = _rank_of_gold(retrieved, hard_case["gold_doc_id"])
        top_doc = retrieved[0] if retrieved else {}
        rows.append(
            {
                "category": item["category"],
                "qid": hard_case["qid"],
                "retriever": retriever_name,
                "failure_type": hard_case.get("failure_type", "unlabeled"),
                "failed_retrievers": "|".join(_failed_retrievers(hard_case)),
                "original_question": hard_case["question"],
                "answer": hard_case.get("answer", ""),
                "selected_strategy": record["strategy"],
                "selected_query": record["query"],
                "original_rank": _original_rank(hard_case, retriever_name),
                "selected_gold_rank": gold_rank or ">10",
                "recall@10": record.get("recall@10", 0),
                "mrr": round(float(record.get("mrr", 0.0)), 4),
                "answer_f1": round(float(record.get("answer_f1", 0.0)), 4),
                "reward": round(float(record.get("reward", 0.0)), 4),
                "top1_doc_id": top_doc.get("doc_id", ""),
                "top1_passage_snippet": _snippet(top_doc.get("text", ""), hard_case.get("answer", "")),
                "gold_passage_snippet": _snippet(hard_case.get("gold_passage", ""), hard_case.get("answer", "")),
                "interpretation": _interpret_example(item["category"], record, gold_rank),
            }
        )
    return rows


def build_failure_type_manual_check(hard_cases: list[dict], sample_size: int = 100) -> list[dict]:
    sampled = _round_robin_by_label(hard_cases, sample_size=sample_size)
    rows = []
    for hard_case in sampled:
        suggested_label, confidence, reason = _suggest_manual_label(hard_case)
        rows.append(
            {
                "qid": hard_case["qid"],
                "question": hard_case["question"],
                "answer": hard_case.get("answer", ""),
                "gold_doc_id": hard_case.get("gold_doc_id", ""),
                "current_failure_type": hard_case.get("failure_type", "unlabeled"),
                "suggested_manual_label": suggested_label,
                "confidence": confidence,
                "reason": reason,
                "failed_retrievers": "|".join(_failed_retrievers(hard_case)),
                "query_length": len(tokenize(hard_case.get("question", ""))),
                "keywords": extract_keywords(hard_case.get("question", "")),
                "bm25_original_rank": _original_rank(hard_case, "bm25"),
                "dense_original_rank": _original_rank(hard_case, "dense"),
                "hybrid_original_rank": _original_rank(hard_case, "hybrid"),
                "manual_failure_type": "",
                "is_current_label_correct": "",
                "notes": "",
            }
        )
    return rows


def write_markdown_examples(rows: list[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    lines = [
        "# Qualitative Examples",
        "",
        "These examples are selected from original-query hard cases. The selected strategy is the highest-reward rewrite action for the failed retriever in the logged reward table.",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Example {idx}: {row['category']} / {row['retriever']} / {row['selected_strategy']}",
                "",
                f"- QID: `{row['qid']}`",
                f"- Failure type: `{row['failure_type']}`",
                f"- Original question: {row['original_question']}",
                f"- Selected query: {row['selected_query']}",
                f"- Answer: {row['answer']}",
                f"- Original rank: {row['original_rank']}; selected gold rank: {row['selected_gold_rank']}",
                f"- Metrics: Recall@10={row['recall@10']}, MRR={row['mrr']}, Answer F1={row['answer_f1']}, Reward={row['reward']}",
                f"- Top-1 passage: {row['top1_passage_snippet']}",
                f"- Gold passage: {row['gold_passage_snippet']}",
                f"- Interpretation: {row['interpretation']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_manual_check(rows: list[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    lines = [
        "# Failure Type Manual Check Preview",
        "",
        f"Manual check sheet size: {len(rows)} examples.",
        "",
        "| qid | current | suggested | confidence | failed retrievers | question |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows[:30]:
        lines.append(
            "| {qid} | `{current_failure_type}` | `{suggested_manual_label}` | {confidence} | {failed_retrievers} | {question} |".format(
                **{key: _escape_md(str(value)) for key, value in row.items()}
            )
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _index_rewrite_results(rewrite_results: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    indexed = defaultdict(dict)
    for record in rewrite_results:
        indexed[(record["qid"], record["retriever"])][record["strategy"]] = record
    return indexed


def _take_diverse_examples(candidates: list[dict], max_recovered: int, max_not_recovered: int) -> list[dict]:
    selected = []
    quotas = {"recovered": max_recovered, "not_recovered": max_not_recovered}
    seen_pairs = set()
    for item in sorted(candidates, key=lambda row: row["sort_key"]):
        category = item["category"]
        if quotas.get(category, 0) <= 0:
            continue
        pair = (category, item["retriever"], item["hard_case"].get("failure_type", "unlabeled"))
        if pair in seen_pairs and _has_unseen_pair(candidates, selected, category, quotas[category]):
            continue
        selected.append(item)
        seen_pairs.add(pair)
        quotas[category] -= 1
    return selected


def _has_unseen_pair(candidates: list[dict], selected: list[dict], category: str, remaining_quota: int) -> bool:
    if remaining_quota <= 1:
        return False
    selected_pairs = {
        (item["category"], item["retriever"], item["hard_case"].get("failure_type", "unlabeled"))
        for item in selected
    }
    for item in candidates:
        pair = (item["category"], item["retriever"], item["hard_case"].get("failure_type", "unlabeled"))
        if item["category"] == category and pair not in selected_pairs:
            return True
    return False


def _round_robin_by_label(hard_cases: list[dict], sample_size: int) -> list[dict]:
    grouped = defaultdict(list)
    for hard_case in hard_cases:
        grouped[hard_case.get("failure_type", "unlabeled")].append(hard_case)

    labels = sorted(grouped)
    sampled = []
    index = 0
    while len(sampled) < sample_size:
        added = False
        for label in labels:
            if index < len(grouped[label]):
                sampled.append(grouped[label][index])
                added = True
                if len(sampled) >= sample_size:
                    break
        if not added:
            break
        index += 1
    return sampled


def _suggest_manual_label(hard_case: dict) -> tuple[str, float, str]:
    question = hard_case.get("question", "")
    tokens = tokenize(question)
    failed = set(_failed_retrievers(hard_case))

    if any(char.isdigit() for char in question):
        return "temporal_numeric", 0.9, "Question contains a date, number, or count expression."
    if any(char.isascii() and char.isalpha() for char in question):
        return "abbreviation_or_foreign_term", 0.8, "Question contains Latin alphabet terms that can cause lexical mismatch."
    if len(tokens) <= 4:
        return "underspecified_short_query", 0.7, "Question is short, so missing context or entities may hurt retrieval."
    if any(len(token) >= 7 for token in tokens):
        return "long_compound_or_entity", 0.7, "Question includes long Korean tokens that may need decomposition."
    if failed == {"dense"}:
        return "dense_only_failure", 0.7, "BM25 or hybrid found the gold passage, but dense retrieval missed it."
    if failed == {"bm25"}:
        return "lexical_only_failure", 0.7, "Dense or hybrid found the gold passage, but BM25 missed it."
    if "hybrid" in failed and len(failed) >= 2:
        return "multi_retriever_failure", 0.8, "Multiple retrievers missed the gold passage."
    return "needs_manual_review", 0.5, "No high-confidence automatic cue was detected."


def _rank_of_gold(retrieved: list[dict], gold_doc_id: str) -> int | None:
    for idx, item in enumerate(retrieved, start=1):
        if item.get("doc_id") == gold_doc_id:
            return idx
    return None


def _original_rank(hard_case: dict, retriever: str) -> int | str:
    if hard_case.get("retriever") == retriever:
        gold_rank = hard_case.get("gold_rank")
        if gold_rank not in (None, ""):
            return int(gold_rank)

    retrieved = hard_case.get("original_retrieved_by_retriever", {}).get(retriever, [])
    gold_doc_id = hard_case.get("gold_doc_id")
    if gold_doc_id in retrieved:
        return retrieved.index(gold_doc_id) + 1
    return ">10"


def _failed_retrievers(hard_case: dict) -> list[str]:
    failed = hard_case.get("failed_retrievers")
    if isinstance(failed, list):
        return [str(item) for item in failed if str(item).strip()]
    source_retriever = str(hard_case.get("retriever", "")).strip()
    return [source_retriever] if source_retriever else []


def _snippet(text: str, answer: str = "", max_chars: int = 180) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    if answer and answer in text and len(text) > max_chars:
        start = max(0, text.index(answer) - max_chars // 3)
        return _trim(text[start : start + max_chars], max_chars)
    return _trim(text, max_chars)


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _interpret_example(category: str, record: dict, gold_rank: int | None) -> str:
    strategy = record.get("strategy", "")
    if category == "recovered":
        return f"The {strategy} rewrite recovered the gold passage at rank {gold_rank} by changing the retrieval surface form."
    return f"Even the best logged action ({strategy}) did not recover the gold passage in top-10; this is a useful error case."


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _qualitative_fields() -> list[str]:
    return [
        "category",
        "qid",
        "retriever",
        "failure_type",
        "failed_retrievers",
        "original_question",
        "answer",
        "selected_strategy",
        "selected_query",
        "original_rank",
        "selected_gold_rank",
        "recall@10",
        "mrr",
        "answer_f1",
        "reward",
        "top1_doc_id",
        "top1_passage_snippet",
        "gold_passage_snippet",
        "interpretation",
    ]


def _manual_check_fields() -> list[str]:
    return [
        "qid",
        "question",
        "answer",
        "gold_doc_id",
        "current_failure_type",
        "suggested_manual_label",
        "confidence",
        "reason",
        "failed_retrievers",
        "query_length",
        "keywords",
        "bm25_original_rank",
        "dense_original_rank",
        "hybrid_original_rank",
        "manual_failure_type",
        "is_current_label_correct",
        "notes",
    ]


if __name__ == "__main__":
    main()
