import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import ensure_dir, read_jsonl, write_jsonl


DEFAULT_INPUT_PATH = Path("data/outputs/hard_cases/hard_subset_1000.jsonl")
OUTPUT_DIR = Path("data/outputs/annotation")
PREVIOUS_ANNOTATION_PATHS = [
    OUTPUT_DIR / "hard_subset_1000_annotation_final.jsonl",
]

FAILURE_TYPE_TO_LABEL = {
    "expression_mismatch": "lexical_mismatch",
    "ellipsis": "missing_key_term",
    "temporal_numeric": "numeric_temporal_mismatch",
    "abbreviation": "lexical_mismatch",
    "compound_noun": "lexical_mismatch",
    "unlabeled": "other",
}
AUTO_LABELS = {"lexical_mismatch", "missing_key_term", "numeric_temporal_mismatch"}

BASE_COLUMNS = [
    "dataset",
    "qid",
    "annotation_priority",
    "question_type",
    "failure_type",
    "suggested_failure_label",
    "failure_label",
    "secondary_failure_label",
    "annotated",
    "question",
    "answer",
    "gold_doc_id",
    "gold_rank",
    "hardness_level",
    "hit_at_10",
    "recall_at_10",
    "mrr",
    "top10_doc_ids",
    "needs_gold_check",
    "annotation_note",
    "annotator",
]

ANNOTATION_DEFAULTS = {
    "failure_label": "",
    "secondary_failure_label": "",
    "annotation_note": "",
    "needs_gold_check": False,
    "annotator": "",
    "annotated": False,
    "suggested_failure_label": "",
    "annotation_priority": "",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, assist, and auto-finalize hard-case annotations.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Hard subset JSONL path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Hard subset file not found: {input_path}")

    ensure_dir(OUTPUT_DIR)
    hard_cases = read_jsonl(input_path)
    previous_by_qid = _load_previous_annotations(PREVIOUS_ANNOTATION_PATHS)

    annotation_records = [_make_annotation_record(record) for record in hard_cases]
    assist_records = [_make_assist_record(record) for record in annotation_records]
    final_records = [_make_final_record(record, previous_by_qid) for record in assist_records]
    needs_review = [record for record in final_records if _needs_review(record)]

    paths = {
        "annotation_jsonl": OUTPUT_DIR / "hard_subset_1000_annotation.jsonl",
        "annotation_csv": OUTPUT_DIR / "hard_subset_1000_annotation.csv",
        "assist_jsonl": OUTPUT_DIR / "hard_subset_1000_annotation_assist.jsonl",
        "assist_csv": OUTPUT_DIR / "hard_subset_1000_annotation_assist.csv",
        "needs_review_jsonl": OUTPUT_DIR / "hard_subset_1000_needs_review.jsonl",
        "needs_review_csv": OUTPUT_DIR / "hard_subset_1000_needs_review.csv",
        "final_jsonl": OUTPUT_DIR / "hard_subset_1000_annotation_final.jsonl",
        "final_csv": OUTPUT_DIR / "hard_subset_1000_annotation_final.csv",
        "summary": OUTPUT_DIR / "final_annotation_summary_1000.json",
    }

    write_jsonl(annotation_records, paths["annotation_jsonl"])
    _write_csv(annotation_records, paths["annotation_csv"])
    write_jsonl(assist_records, paths["assist_jsonl"])
    _write_csv(assist_records, paths["assist_csv"])
    write_jsonl(needs_review, paths["needs_review_jsonl"])
    _write_csv(needs_review, paths["needs_review_csv"])
    write_jsonl(final_records, paths["final_jsonl"])
    _write_csv(final_records, paths["final_csv"])

    summary = _build_summary(final_records, needs_review)
    _write_json(summary, paths["summary"])
    _print_summary(summary)

    print(f"Saved annotation files for {len(final_records)} records under {OUTPUT_DIR}")


def _make_annotation_record(record: dict[str, Any]) -> dict[str, Any]:
    annotation = dict(record)
    for key, value in ANNOTATION_DEFAULTS.items():
        annotation[key] = value
    return annotation


def _make_assist_record(record: dict[str, Any]) -> dict[str, Any]:
    assist = dict(record)
    assist["suggested_failure_label"] = FAILURE_TYPE_TO_LABEL.get(str(record.get("failure_type", "unlabeled")), "other")
    return assist


def _make_final_record(record: dict[str, Any], previous_by_qid: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    final = dict(record)
    previous = (previous_by_qid or {}).get(str(final.get("qid", "")))
    if previous:
        for key in (
            "failure_label",
            "secondary_failure_label",
            "annotation_note",
            "needs_gold_check",
            "annotator",
            "annotated",
            "suggested_failure_label",
            "annotation_priority",
        ):
            if key in previous:
                final[key] = previous[key]
        return final

    suggested = str(final.get("suggested_failure_label", "")).strip()

    final["failure_label"] = suggested if suggested in AUTO_LABELS else ""
    final["secondary_failure_label"] = _secondary_failure_label(final)
    final["annotated"] = bool(final["failure_label"])
    final["annotation_priority"] = _annotation_priority(final)
    return final


def _load_previous_annotations(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records = {}
    for path in paths:
        if not path.exists():
            continue
        for record in read_jsonl(path):
            qid = str(record.get("qid", "")).strip()
            if qid and qid not in records:
                records[qid] = record
    return records


def _secondary_failure_label(record: dict[str, Any]) -> str:
    if record.get("dataset") == "korquad2" and record.get("synthetic_from_qa") is True:
        return "synthetic_chunk_issue"
    return str(record.get("secondary_failure_label", "") or "")


def _annotation_priority(record: dict[str, Any]) -> str:
    if not str(record.get("failure_label", "")).strip():
        return "high"
    if record.get("secondary_failure_label") == "synthetic_chunk_issue":
        return "medium"
    return "low"


def _needs_review(record: dict[str, Any]) -> bool:
    return (
        not str(record.get("failure_label", "")).strip()
        or record.get("annotated") is False
        or record.get("suggested_failure_label") == "other"
    )


def _build_summary(records: list[dict[str, Any]], needs_review: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_records": len(records),
        "auto_annotated_count": sum(1 for record in records if record.get("annotated") is True),
        "needs_review_count": len(needs_review),
        "dataset_counts": _counter(records, "dataset"),
        "question_type_counts": _counter(records, "question_type"),
        "failure_type_counts": _counter(records, "failure_type"),
        "suggested_failure_label_counts": _counter(records, "suggested_failure_label"),
        "failure_label_counts": _counter(records, "failure_label"),
        "secondary_failure_label_counts": _counter(records, "secondary_failure_label"),
        "annotation_priority_counts": _counter(records, "annotation_priority"),
    }


def _counter(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(field, "") or "") for record in records).items()))


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    extra_columns = sorted({key for record in records for key in record if key not in BASE_COLUMNS})
    fieldnames = BASE_COLUMNS + extra_columns
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({column: _csv_value(record.get(column, "")) for column in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def _print_summary(summary: dict[str, Any]) -> None:
    print("Final annotation summary")
    print(f"total_records: {summary['total_records']}")
    print(f"auto_annotated_count: {summary['auto_annotated_count']}")
    print(f"needs_review_count: {summary['needs_review_count']}")
    print(f"failure_label_counts: {summary['failure_label_counts']}")
    print(f"annotation_priority_counts: {summary['annotation_priority_counts']}")


if __name__ == "__main__":
    main()
