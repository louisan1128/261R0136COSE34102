import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import ensure_dir, read_jsonl, write_jsonl


INPUT_PATH = Path("data/outputs/hard_cases/hard_subset_300.jsonl")
OUTPUT_DIR = Path("data/outputs/annotation")

ANNOTATION_CSV_PATH = OUTPUT_DIR / "hard_subset_300_annotation.csv"
ANNOTATION_JSONL_PATH = OUTPUT_DIR / "hard_subset_300_annotation.jsonl"
LABEL_SCHEMA_PATH = OUTPUT_DIR / "annotation_label_schema.json"

ANNOTATION_FIELDS = {
    "failure_label": "",
    "secondary_failure_label": "",
    "annotation_note": "",
    "needs_gold_check": False,
    "annotator": "",
    "annotated": False,
}

CSV_COLUMNS = [
    "dataset",
    "qid",
    "question_type",
    "question",
    "answer",
    "gold_doc_id",
    "gold_rank",
    "top10_doc_ids",
    "failed_retrievers",
    "failure_type",
    "failure_label",
    "secondary_failure_label",
    "needs_gold_check",
    "annotation_note",
    "annotator",
    "annotated",
]

LABEL_SCHEMA = {
    "failure_label_candidates": [
        "lexical_mismatch",
        "missing_key_term",
        "entity_ambiguity",
        "too_broad_query",
        "too_specific_query",
        "paraphrase_mismatch",
        "numeric_temporal_mismatch",
        "question_type_mismatch",
        "gold_mismatch",
        "synthetic_chunk_issue",
        "retrieval_model_failure",
        "other",
    ],
    "label_descriptions": {
        "lexical_mismatch": "질문 표현과 gold passage 표현이 달라 BM25가 실패한 경우",
        "missing_key_term": "검색에 필요한 핵심 단어가 질문에 없는 경우",
        "entity_ambiguity": "사람/장소/기관 이름이 모호한 경우",
        "too_broad_query": "질문이 너무 넓어서 여러 문서가 비슷하게 검색되는 경우",
        "too_specific_query": "질문이 너무 세부적이거나 문장 구조가 복잡한 경우",
        "paraphrase_mismatch": "의미는 같지만 표현이 달라 dense/hybrid도 못 잡은 경우",
        "numeric_temporal_mismatch": "날짜, 수치, 시간 표현 때문에 실패한 경우",
        "question_type_mismatch": "how/why/comparison 등 질문 유형 때문에 실패한 경우",
        "gold_mismatch": "gold_doc_id 자체가 잘못됐거나 정답 passage가 부적절한 경우",
        "synthetic_chunk_issue": "KorQuAD2 synthetic chunk 때문에 생긴 실패",
        "retrieval_model_failure": "라벨상 문제는 명확하지 않고 retriever 점수화 실패로 보이는 경우",
        "other": "위에 해당하지 않는 경우",
    },
    "annotation_fields": {
        "failure_label": "Primary manual label. Keep empty until annotated.",
        "secondary_failure_label": "Optional secondary manual label. Keep empty until annotated.",
        "annotation_note": "Free-form annotator note.",
        "needs_gold_check": "Set true if gold_doc_id or gold passage quality needs review.",
        "annotator": "Annotator name or id.",
        "annotated": "Set true after manual annotation is complete.",
        "suggested_failure_label": "Optional reference hint. Do not copy automatically without review.",
    },
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Hard subset file not found: {INPUT_PATH}")

    ensure_dir(OUTPUT_DIR)
    records = read_jsonl(INPUT_PATH)
    annotation_records = [_make_annotation_record(record) for record in records]

    write_jsonl(annotation_records, ANNOTATION_JSONL_PATH)
    _write_annotation_csv(annotation_records, ANNOTATION_CSV_PATH)
    _write_json(LABEL_SCHEMA, LABEL_SCHEMA_PATH)

    print(f"Saved {len(annotation_records)} annotation records to {ANNOTATION_JSONL_PATH}")
    print(f"Saved annotation CSV to {ANNOTATION_CSV_PATH}")
    print(f"Saved annotation label schema to {LABEL_SCHEMA_PATH}")


def _make_annotation_record(record: dict[str, Any]) -> dict[str, Any]:
    annotation_record = dict(record)
    annotation_record.update(ANNOTATION_FIELDS)
    annotation_record["suggested_failure_label"] = (
        "synthetic_chunk_issue" if annotation_record.get("dataset") == "korquad2" else ""
    )
    return annotation_record


def _write_annotation_csv(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: _csv_value(record.get(column, "")) for column in CSV_COLUMNS})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
