import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import ensure_dir, read_jsonl, write_jsonl


DEFAULT_INPUT = Path("data/outputs/annotation/hard_subset_1000_needs_review.jsonl")
DEFAULT_JSONL_OUTPUT = Path("data/outputs/annotation/hard_subset_1000_needs_review_labeled.jsonl")
DEFAULT_CSV_OUTPUT = Path("data/outputs/annotation/hard_subset_1000_needs_review_labeled.csv")

CORPUS_PATHS = {
    "korquad1": Path("data/processed/korquad1_corpus.jsonl"),
    "klue_mrc": Path("data/processed/klue_mrc_corpus.jsonl"),
    "korquad2": Path("data/processed/korquad2_filtered_corpus.jsonl"),
}

LABELS = {
    "lexical_mismatch",
    "missing_key_term",
    "numeric_temporal_mismatch",
    "semantic_mismatch",
    "entity_mismatch",
    "context_boundary_issue",
    "ambiguous",
}

STOPWORDS = {
    "것",
    "수",
    "등",
    "및",
    "이",
    "그",
    "저",
    "한",
    "위해",
    "대한",
    "어떤",
    "무엇",
    "누구",
    "언제",
    "어디",
    "몇",
    "얼마",
    "있는",
    "없는",
    "하는",
    "된다",
    "되었다",
    "했다",
    "무슨",
    "어느",
    "왜",
    "어떻게",
}

PRONOUN_PATTERNS = [
    "그는",
    "그가",
    "그의",
    "그녀",
    "그들",
    "이것",
    "그것",
    "해당",
    "이 작품",
    "그 작품",
    "이 영화",
    "그 영화",
    "이 사건",
    "그 사건",
    "이 앨범",
    "그 앨범",
    "이 인물",
    "그 인물",
    "이 회사",
    "그 회사",
]

NUMERIC_RE = re.compile(
    r"(\d+(?:[.,]\d+)?\s*(?:년|월|일|위|명|개|회|차|세|분|초|시간|조원|억원|원|%|퍼센트|km|m|cm|킬로미터|미터|달러|석|점|장|권|편|대|종|번|년대)?|"
    r"[일이삼사오육칠팔구십백천만억조]+(?:년|월|일|위|명|개|회|차|세|분|초|시간|조원|억원|원|위)?)"
)
NUMERIC_CUES = {
    "몇",
    "얼마",
    "언제",
    "연도",
    "년도",
    "날짜",
    "시기",
    "기간",
    "순위",
    "몇위",
    "몇 위",
    "수량",
    "나이",
    "시간",
    "초",
    "분",
    "몇 명",
    "몇개",
    "몇 개",
}
STRUCTURE_CUES = {
    "<table",
    "</td",
    "</tr",
    "<li",
    " rowspan",
    " colspan",
    "infobox",
    "sortable",
    "wikitable",
    "편집 ]",
    "[ 편집",
    "목록",
    "표 ",
    "row",
    "col",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label hard_subset_850 needs-review rows with context-aware rules.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_JSONL_OUTPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Needs-review file not found: {input_path}")

    corpus = _load_corpus()
    rows = read_jsonl(input_path)
    labeled = [_label_record(record, corpus) for record in rows]
    _validate(rows, labeled)

    jsonl_output = Path(args.output_jsonl)
    csv_output = Path(args.output_csv)
    write_jsonl(labeled, jsonl_output)
    _write_csv(labeled, csv_output)
    _print_summary(labeled, jsonl_output, csv_output)


def _load_corpus() -> dict[str, dict[str, dict[str, Any]]]:
    corpus: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, path in CORPUS_PATHS.items():
        mapping: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            for key in ("doc_id", "pid"):
                value = str(row.get(key, "") or "")
                if value:
                    mapping[value] = row
        corpus[dataset] = mapping
    return corpus


def _label_record(record: dict[str, Any], corpus: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    dataset = str(record.get("dataset", "") or "")
    dataset_corpus = corpus.get(dataset, {})
    top_ids = record.get("top10_doc_ids") or []
    if isinstance(top_ids, str):
        try:
            top_ids = json.loads(top_ids)
        except json.JSONDecodeError:
            top_ids = [part.strip() for part in top_ids.split(",") if part.strip()]

    gold_doc_id = str(record.get("gold_doc_id", "") or record.get("gold_pid", "") or "")
    gold_row = dataset_corpus.get(gold_doc_id, {})
    top_rows = [dataset_corpus[doc_id] for doc_id in top_ids if doc_id in dataset_corpus]

    question = str(record.get("question", "") or "")
    answer = str(record.get("answer", "") or "")
    gold_context = str(record.get("gold_context", "") or record.get("gold_passage", "") or gold_row.get("text", "") or "")
    gold_title = str(gold_row.get("title", "") or record.get("title", "") or "")
    top_contexts = [str(row.get("text", "") or "") for row in top_rows]
    top_titles = [str(row.get("title", "") or "") for row in top_rows]
    retrieved_context = "\n\n".join(top_contexts[:3])

    label, secondary, reason, confidence = _classify(
        record=record,
        question=question,
        answer=answer,
        gold_context=gold_context,
        gold_title=gold_title,
        top_rows=top_rows,
        top_contexts=top_contexts,
        top_titles=top_titles,
        retrieved_context=retrieved_context,
    )

    labeled = dict(record)
    labeled["failure_label"] = label
    labeled["secondary_failure_label"] = secondary
    labeled["annotation_reason"] = reason
    labeled["confidence"] = round(confidence, 2)
    labeled["manual_review_required"] = confidence < 0.8
    labeled["retrieved_context"] = retrieved_context
    labeled["top_retrieved_titles"] = top_titles[:10]
    if gold_title:
        labeled["gold_title"] = gold_title
    return labeled


def _classify(
    record: dict[str, Any],
    question: str,
    answer: str,
    gold_context: str,
    gold_title: str,
    top_rows: list[dict[str, Any]],
    top_contexts: list[str],
    top_titles: list[str],
    retrieved_context: str,
) -> tuple[str, str, str, float]:
    dataset = str(record.get("dataset", "") or "")
    question_type = str(record.get("question_type", "") or "")
    top_text = " ".join(top_contexts[:5])
    top1_title = top_titles[0] if top_titles else ""

    if not gold_context.strip() or not top_contexts:
        return (
            "ambiguous",
            "",
            "gold_context 또는 retrieved passage가 비어 있어 문맥 차이를 확정하기 어려우므로 ambiguous로 판단함.",
            0.42,
        )

    q_tokens = _keywords(question)
    gold_tokens = _keywords(gold_context)
    top_tokens = _keywords(top_text)
    title_tokens = _keywords(gold_title)
    q_gold_overlap = _overlap(q_tokens, gold_tokens)
    q_top_overlap = _overlap(q_tokens, top_tokens)
    gold_top_overlap = _overlap(gold_tokens, top_tokens)
    same_source = _same_source_or_title(gold_title, top_rows)
    answer_in_retrieved = bool(answer.strip()) and _contains_normalized(top_text, answer)
    numeric_intent = _has_numeric_intent(question, question_type)
    structural = _has_structure_clue(gold_context) or _has_structure_clue(top_text)
    missing_key = _missing_key_term(question, gold_title, q_tokens, title_tokens, q_gold_overlap)

    if _is_boundary_issue(dataset, structural, same_source, gold_top_overlap, answer_in_retrieved):
        secondary = "numeric_temporal_mismatch" if numeric_intent else ""
        return (
            "context_boundary_issue",
            secondary,
            "retrieved passage가 정답 문서나 같은 제목의 근처 chunk를 포함하지만 필요한 근거가 표/목록 구조나 chunk 경계에서 빠져 context_boundary_issue로 판단함.",
            0.84,
        )

    if numeric_intent and (question_type == "numeric" or _numeric_mismatch(question, gold_context, top_text, answer)):
        if structural or same_source:
            secondary = "context_boundary_issue"
        elif missing_key:
            secondary = "missing_key_term"
        elif _titles_differ(gold_title, top1_title):
            secondary = "entity_mismatch"
        else:
            secondary = ""
        return (
            "numeric_temporal_mismatch",
            secondary,
            "질문의 핵심 조건이 숫자·날짜·순위·수량 정보인데 retrieved passage의 수치 조건이 gold_context와 달라 numeric_temporal_mismatch로 판단함.",
            0.86,
        )

    entity_mismatch = _entity_mismatch(
        question=question,
        gold_title=gold_title,
        top1_title=top1_title,
        q_tokens=q_tokens,
        gold_tokens=gold_tokens,
        top_tokens=top_tokens,
        missing_key=missing_key,
        q_gold_overlap=q_gold_overlap,
    )
    if entity_mismatch:
        secondary = "missing_key_term" if missing_key else ""
        return (
            "entity_mismatch",
            secondary,
            "질문이 가리키는 대상은 비교적 명확하지만 retrieved passage의 제목이나 중심 entity가 gold_context와 달라 entity_mismatch로 판단함.",
            0.83 if not missing_key else 0.78,
        )

    if missing_key:
        secondary = "entity_mismatch" if _titles_differ(gold_title, top1_title) else ""
        return (
            "missing_key_term",
            secondary,
            "질문에 gold_context를 특정할 핵심 entity나 단서가 부족해 retrieved passage가 다른 문서로 이동한 것으로 보여 missing_key_term으로 판단함.",
            0.82,
        )

    if _semantic_mismatch(q_gold_overlap, q_top_overlap, gold_top_overlap, answer_in_retrieved, same_source):
        return (
            "semantic_mismatch",
            "",
            "question과 retrieved passage가 일부 단어를 공유하지만 묻는 관계나 속성이 gold_context의 정답 근거와 달라 semantic_mismatch로 판단함.",
            0.74,
        )

    if q_gold_overlap < 0.18 and not answer_in_retrieved:
        return (
            "lexical_mismatch",
            "",
            "question과 gold_context는 같은 답 근거를 향하지만 표면 표현의 겹침이 낮고 더 강한 entity·수치 오류는 보이지 않아 lexical_mismatch로 판단함.",
            0.71,
        )

    return (
        "ambiguous",
        "",
        "gold_context와 retrieved passage 차이에서 하나의 실패 원인을 충분히 확정하기 어려워 ambiguous로 판단함.",
        0.48,
    )


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", _normalize(text))
    return {token for token in tokens if len(token) >= 2 and token not in STOPWORDS}


def _normalize(text: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _contains_normalized(text: str, needle: str) -> bool:
    haystack = re.sub(r"\s+", "", _normalize(text))
    target = re.sub(r"\s+", "", _normalize(needle))
    return bool(target) and target in haystack


def _has_numeric_intent(question: str, question_type: str) -> bool:
    if question_type == "numeric":
        return True
    normalized = _normalize(question)
    return bool(NUMERIC_RE.search(normalized)) or any(cue in normalized for cue in NUMERIC_CUES)


def _numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group(0)) for match in NUMERIC_RE.finditer(_normalize(text))}


def _numeric_mismatch(question: str, gold_context: str, top_text: str, answer: str) -> bool:
    question_nums = _numbers(question)
    gold_nums = _numbers(gold_context) | _numbers(answer)
    top_nums = _numbers(top_text)
    if not top_nums:
        return True
    if question_nums and not question_nums <= top_nums:
        return True
    if gold_nums and not (gold_nums & top_nums):
        return True
    return bool(gold_nums and top_nums and not (gold_nums & top_nums))


def _has_structure_clue(text: str) -> bool:
    normalized = _normalize(text)
    if any(cue in normalized for cue in STRUCTURE_CUES):
        return True
    return normalized.count("|") >= 4 or normalized.count("[") >= 3


def _same_source_or_title(gold_title: str, top_rows: list[dict[str, Any]]) -> bool:
    normalized_gold_title = _normalize(gold_title)
    if not normalized_gold_title:
        return False
    for row in top_rows:
        title = _normalize(str(row.get("title", "") or ""))
        if title and title == normalized_gold_title:
            return True
    return False


def _is_boundary_issue(
    dataset: str,
    structural: bool,
    same_source: bool,
    gold_top_overlap: float,
    answer_in_retrieved: bool,
) -> bool:
    if answer_in_retrieved:
        return False
    if same_source and (dataset == "korquad2" or structural or gold_top_overlap >= 0.12):
        return True
    return dataset == "korquad2" and structural and gold_top_overlap >= 0.12


def _missing_key_term(
    question: str,
    gold_title: str,
    q_tokens: set[str],
    title_tokens: set[str],
    q_gold_overlap: float,
) -> bool:
    normalized_question = _normalize(question)
    has_pronoun = any(pattern in normalized_question for pattern in PRONOUN_PATTERNS)
    title_overlap = bool(q_tokens & title_tokens) if title_tokens else False
    too_short = len(q_tokens) <= 3
    no_title_anchor = bool(gold_title and title_tokens and not title_overlap)
    return (has_pronoun and no_title_anchor) or too_short or (no_title_anchor and q_gold_overlap < 0.08)


def _entity_mismatch(
    question: str,
    gold_title: str,
    top1_title: str,
    q_tokens: set[str],
    gold_tokens: set[str],
    top_tokens: set[str],
    missing_key: bool,
    q_gold_overlap: float,
) -> bool:
    if missing_key and q_gold_overlap < 0.12:
        return False
    if not _titles_differ(gold_title, top1_title):
        return False
    clear_question_anchor = q_gold_overlap >= 0.12 or bool(q_tokens & gold_tokens)
    if not clear_question_anchor:
        return False
    shared_with_top = bool(q_tokens & top_tokens)
    return shared_with_top or q_gold_overlap >= 0.18


def _titles_differ(gold_title: str, top_title: str) -> bool:
    left = _normalize(gold_title)
    right = _normalize(top_title)
    return bool(left and right and left != right)


def _semantic_mismatch(
    q_gold_overlap: float,
    q_top_overlap: float,
    gold_top_overlap: float,
    answer_in_retrieved: bool,
    same_source: bool,
) -> bool:
    if answer_in_retrieved:
        return False
    if same_source and q_top_overlap >= 0.10:
        return True
    return q_gold_overlap >= 0.12 and q_top_overlap >= 0.10 and gold_top_overlap < 0.28


def _validate(input_rows: list[dict[str, Any]], labeled_rows: list[dict[str, Any]]) -> None:
    if len(input_rows) != len(labeled_rows):
        raise ValueError(f"Output count mismatch: input={len(input_rows)} output={len(labeled_rows)}")
    for index, row in enumerate(labeled_rows, start=1):
        label = row.get("failure_label")
        if label not in LABELS:
            raise ValueError(f"Invalid or empty failure_label at row {index}: {label!r}")
        if "confidence" not in row or not isinstance(row["confidence"], (int, float)):
            raise ValueError(f"Missing confidence at row {index}")
        if not (0.0 <= float(row["confidence"]) <= 1.0):
            raise ValueError(f"Confidence out of range at row {index}: {row['confidence']}")
        if not str(row.get("annotation_reason", "")).strip():
            raise ValueError(f"Missing annotation_reason at row {index}")
        if "manual_review_required" not in row:
            raise ValueError(f"Missing manual_review_required at row {index}")


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    preferred = [
        "dataset",
        "qid",
        "failure_label",
        "secondary_failure_label",
        "confidence",
        "manual_review_required",
        "annotation_reason",
        "question_type",
        "question",
        "answer",
        "gold_title",
        "gold_doc_id",
        "gold_passage",
        "retrieved_context",
        "top_retrieved_titles",
        "top10_doc_ids",
        "annotation_priority",
        "suggested_failure_label",
    ]
    extra = sorted({key for record in records for key in record if key not in preferred})
    fieldnames = preferred + extra
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record.get(field, "")) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _print_summary(records: list[dict[str, Any]], jsonl_output: Path, csv_output: Path) -> None:
    label_counts = Counter(str(record.get("failure_label", "") or "") for record in records)
    manual_count = sum(1 for record in records if record.get("manual_review_required") is True)
    print("Needs-review labeling summary")
    print(f"total_records: {len(records)}")
    print(f"label_counts: {dict(sorted(label_counts.items()))}")
    print(f"ambiguous_count: {label_counts.get('ambiguous', 0)}")
    print(f"manual_review_required_true: {manual_count}")
    print(f"output_jsonl: {jsonl_output}")
    print(f"output_csv: {csv_output}")


if __name__ == "__main__":
    main()
