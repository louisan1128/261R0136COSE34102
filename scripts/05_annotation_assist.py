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


INPUT_PATH = Path("data/outputs/annotation/hard_subset_300_annotation.jsonl")
OUTPUT_DIR = Path("data/outputs/annotation")
ASSIST_CSV_PATH = OUTPUT_DIR / "hard_subset_300_annotation_assist.csv"
ASSIST_JSONL_PATH = OUTPUT_DIR / "hard_subset_300_annotation_assist.jsonl"

FAILURE_TYPE_TO_LABEL = {
    "expression_mismatch": "lexical_mismatch",
    "ellipsis": "missing_key_term",
    "temporal_numeric": "numeric_temporal_mismatch",
    "abbreviation": "lexical_mismatch",
    "compound_noun": "lexical_mismatch",
    "unlabeled": "other",
}

REVIEW_COLUMNS = [
    "dataset",
    "qid",
    "annotation_priority",
    "question_type",
    "failure_type",
    "suggested_failure_label",
    "suggested_failure_label_2",
    "failure_label",
    "secondary_failure_label",
    "question",
    "answer",
    "gold_doc_id",
    "gold_rank",
    "top10_doc_ids",
    "failed_retrievers",
    "needs_gold_check",
    "annotation_note",
    "annotator",
    "annotated",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Annotation file not found: {INPUT_PATH}")

    ensure_dir(OUTPUT_DIR)
    records = read_jsonl(INPUT_PATH)
    assist_records = [_make_assist_record(record) for record in records]

    write_jsonl(assist_records, ASSIST_JSONL_PATH)
    _write_assist_csv(assist_records, ASSIST_CSV_PATH)
    _print_summary(assist_records)

    print(f"Saved {len(assist_records)} assisted annotation records to {ASSIST_JSONL_PATH}")
    print(f"Saved assisted annotation CSV to {ASSIST_CSV_PATH}")


def _make_assist_record(record: dict[str, Any]) -> dict[str, Any]:
    assist_record = dict(record)
    existing_suggestion = str(record.get("suggested_failure_label", "")).strip()
    mapped_label = FAILURE_TYPE_TO_LABEL.get(str(record.get("failure_type", "unlabeled")), "other")

    assist_record["failure_label"] = ""
    assist_record["suggested_failure_label"] = mapped_label
    assist_record["suggested_failure_label_2"] = (
        "synthetic_chunk_issue"
        if record.get("dataset") == "korquad2" and existing_suggestion == "synthetic_chunk_issue"
        else ""
    )
    assist_record["annotation_priority"] = _annotation_priority(assist_record)
    return assist_record


def _annotation_priority(record: dict[str, Any]) -> str:
    if record.get("failure_type") == "unlabeled":
        return "high"
    if record.get("dataset") == "korquad2" and record.get("suggested_failure_label_2") == "synthetic_chunk_issue":
        return "medium"
    return "low"


def _write_assist_csv(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    extra_columns = sorted({key for record in records for key in record if key not in REVIEW_COLUMNS})
    fieldnames = REVIEW_COLUMNS + extra_columns
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({column: _csv_value(record.get(column, "")) for column in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _print_summary(records: list[dict[str, Any]]) -> None:
    summaries = {
        "dataset": Counter(record.get("dataset", "") for record in records),
        "question_type": Counter(record.get("question_type", "") for record in records),
        "failure_type": Counter(record.get("failure_type", "") for record in records),
        "suggested_failure_label": Counter(record.get("suggested_failure_label", "") for record in records),
        "annotation_priority": Counter(record.get("annotation_priority", "") for record in records),
    }
    print("Annotation assist summary")
    for name, counter in summaries.items():
        print(f"- {name}: {dict(sorted(counter.items()))}")


if __name__ == "__main__":
    main()
