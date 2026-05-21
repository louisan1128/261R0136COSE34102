import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.rewriting.llm_client import OpenAICompatibleRewriter, load_rewrite_cache, save_rewrite_cache
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from src.utils.text import extract_keywords, tokenize


DEFAULT_INPUT_PATH = Path("data/outputs/annotation/hard_subset_850_annotation_final.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/outputs/rewrites/hard_subset_850_rewrites.jsonl")
REWRITE_TYPES = {"original", "keyword", "prompt_style", "structured", "llm"}
REWRITE_TYPE_ORDER = ["original", "keyword", "prompt_style", "structured", "llm"]
STOPWORDS = {
    "무엇",
    "누구",
    "언제",
    "어디",
    "어떤",
    "어느",
    "왜",
    "어떻게",
    "무슨",
    "인가",
    "인가요",
    "되는가",
    "되나요",
    "했는가",
    "했나요",
    "일까",
    "까",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "에서",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate five rewrite candidates per annotated hard case.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input annotation final JSONL path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output rewrite JSONL path.")
    parser.add_argument(
        "--no-external-llm",
        action="store_true",
        help="Disable OpenAI-compatible LLM calls even if an API key is available.",
    )
    parser.add_argument(
        "--llm-delay-seconds",
        type=float,
        default=None,
        help="Optional delay between external LLM API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_jsonl = Path(args.output)
    output_csv = output_jsonl.with_suffix(".csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Annotation final file not found: {input_path}")

    config = read_yaml(root / "configs" / "default.yaml")
    records = read_jsonl(input_path)
    rewriter, rewrite_cache, cache_path, llm_delay_seconds = _maybe_build_rewriter(config, args)

    output_records = []
    last_llm_request_at = 0.0
    llm_api_calls = 0
    llm_fallback_count = 0
    fallback_examples = []

    for record in records:
        question = str(record.get("question", "")).strip()
        llm_query = None
        if rewriter is not None and question:
            llm_query = rewrite_cache.get(question)
            if not llm_query:
                try:
                    last_llm_request_at = _wait_for_request_slot(last_llm_request_at, llm_delay_seconds)
                    llm_query = rewriter.rewrite(question, str(record.get("failure_type", "unlabeled")))
                    last_llm_request_at = time.monotonic()
                    rewrite_cache[question] = llm_query
                    llm_api_calls += 1
                    save_rewrite_cache(rewrite_cache, cache_path)
                except Exception as exc:
                    llm_query = None
                    llm_fallback_count += 1
                    if len(fallback_examples) < 5:
                        fallback_examples.append(f"qid={record.get('qid')}: {exc}")

        output_records.append(_make_output_record(record, llm_query))

    csv_rows = _make_csv_rows(output_records)
    warnings = _validate_outputs(records, output_records, csv_rows)

    write_jsonl(output_records, output_jsonl)
    _write_csv(csv_rows, output_csv)
    if rewriter is not None:
        save_rewrite_cache(rewrite_cache, cache_path)

    summary = _build_summary(output_records, output_jsonl, output_csv)
    _print_summary(summary, warnings, llm_api_calls, llm_fallback_count, fallback_examples)


def _maybe_build_rewriter(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[OpenAICompatibleRewriter | None, dict[str, str], str, float]:
    llm_config = dict(config.get("llm_rewrite", {}))
    api_key_env = str(llm_config.get("api_key_env", "OPENAI_API_KEY"))
    has_api_key = bool(os.environ.get(api_key_env, "").strip())
    if args.no_external_llm or not has_api_key:
        return None, {}, str(llm_config.get("cache_path", "data/outputs/llm_rewrite_cache.jsonl")), 0.0

    llm_config.setdefault("base_url", "https://api.openai.com/v1")
    llm_config.setdefault("model_env", "OPENAI_MODEL")
    if not os.environ.get(str(llm_config.get("model_env", "OPENAI_MODEL")), "").strip() and not llm_config.get("model"):
        # If the key exists but no model is configured, fallback gracefully instead of failing the pipeline.
        print("Warning: OpenAI API key found, but no OPENAI_MODEL or llm_rewrite.model is configured; using rule-based LLM fallback.")
        return None, {}, str(llm_config.get("cache_path", "data/outputs/llm_rewrite_cache.jsonl")), 0.0

    try:
        rewriter = OpenAICompatibleRewriter.from_config(llm_config)
    except Exception as exc:
        print(f"Warning: could not initialize external LLM rewriter; using fallback. Reason: {exc}")
        return None, {}, str(llm_config.get("cache_path", "data/outputs/llm_rewrite_cache.jsonl")), 0.0

    cache_path = str(llm_config.get("cache_path", "data/outputs/llm_rewrite_cache.jsonl"))
    delay = args.llm_delay_seconds if args.llm_delay_seconds is not None else float(llm_config.get("request_delay_seconds", 0.0))
    print(f"External LLM rewrite enabled with model '{rewriter.config.model}'.")
    return rewriter, load_rewrite_cache(cache_path), cache_path, delay


def _make_output_record(record: dict[str, Any], llm_query: str | None) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    gold_context = str(record.get("gold_passage", "") or record.get("gold_context", ""))
    candidates = [
        {
            "rewrite_type": "original",
            "rewrite_query": question,
            "generation_method": "copy",
        },
        {
            "rewrite_type": "keyword",
            "rewrite_query": _keyword_rewrite(question),
            "generation_method": "rule_based",
        },
        {
            "rewrite_type": "prompt_style",
            "rewrite_query": _prompt_style_rewrite(question),
            "generation_method": "rule_based",
        },
        {
            "rewrite_type": "structured",
            "rewrite_query": _structured_rewrite(question, record),
            "generation_method": "rule_based",
        },
        {
            "rewrite_type": "llm",
            "rewrite_query": _clean_query(llm_query) or _llm_fallback_rewrite(question, record),
            "generation_method": "llm_or_fallback",
        },
    ]
    return {
        "dataset": record.get("dataset", ""),
        "qid": record.get("qid", ""),
        "question": question,
        "gold_context": gold_context,
        "failure_label": record.get("failure_label", ""),
        "secondary_failure_label": record.get("secondary_failure_label", ""),
        "annotation_priority": record.get("annotation_priority", ""),
        "rewrite_candidates": candidates,
    }


def _keyword_rewrite(question: str) -> str:
    keywords = extract_keywords(question, max_keywords=10)
    tokens = [token for token in keywords.split() if token and token not in STOPWORDS]
    if not tokens:
        tokens = [
            token
            for token in tokenize(question)
            if len(token) > 1 and token not in STOPWORDS and not _is_question_ending(token)
        ][:10]
    numeric_terms = _numeric_terms(question)
    return _join_unique(tokens + numeric_terms) or _clean_query(question)


def _prompt_style_rewrite(question: str) -> str:
    question = _clean_query(question)
    if not question:
        return ""
    return f"다음 질문의 답을 찾기 위한 문서를 검색한다: {question}"


def _structured_rewrite(question: str, record: dict[str, Any]) -> str:
    terms = _keyword_rewrite(question).split()
    entity = " ".join(terms[:3]) if terms else _clean_query(question)
    intent = _intent(question, str(record.get("question_type", "")))
    constraint = _constraint(question, record)
    return f"entity: {entity} | intent: {intent} | constraint: {constraint}"


def _llm_fallback_rewrite(question: str, record: dict[str, Any]) -> str:
    keyword = _keyword_rewrite(question)
    failure_label = str(record.get("failure_label", "") or "needs_review")
    intent = _intent(question, str(record.get("question_type", "")))
    if keyword:
        return f"{keyword} {intent} 근거 문서 검색"
    return f"{question} {failure_label} 정답 근거 문서 검색"


def _intent(question: str, question_type: str) -> str:
    if question_type in {"numeric", "when"} or re.search(r"몇|언제|연도|년도|날짜|시기|몇 년|몇 월|몇 일", question):
        return "숫자·시간 정보"
    if question_type == "why" or re.search(r"왜|이유|원인|목적|계기", question):
        return "이유·원인"
    if question_type == "how" or re.search(r"어떻게|방법|방식|과정", question):
        return "방법·과정"
    if question_type == "who" or re.search(r"누구|인물|사람|이름", question):
        return "인물·주체"
    if question_type == "where" or re.search(r"어디|장소|지역|국가|도시|기관", question):
        return "장소·기관"
    if question_type == "comparison" or re.search(r"비교|차이|다른|공통", question):
        return "비교 정보"
    if question_type == "definition" or re.search(r"정의|의미|뜻", question):
        return "정의·의미"
    if question_type == "list" or re.search(r"목록|종류|나열", question):
        return "목록 정보"
    return "정답 정보"


def _constraint(question: str, record: dict[str, Any]) -> str:
    numeric = _join_unique(_numeric_terms(question))
    if numeric:
        return numeric
    failure_label = str(record.get("failure_label", "")).strip()
    if failure_label:
        return failure_label
    priority = str(record.get("annotation_priority", "")).strip()
    return priority or "gold evidence"


def _numeric_terms(question: str) -> list[str]:
    return [token for token in tokenize(question) if re.search(r"\d|몇|년|월|일|번째|번|위|%|명|개|시간|초|cm|km", token)]


def _is_question_ending(token: str) -> bool:
    return bool(re.search(r"(인가|인가요|되는가|되나요|했는가|했나요|일까|일까요|나요|까)$", token))


def _join_unique(parts: list[str]) -> str:
    values = []
    seen = set()
    for part in parts:
        for token in str(part).split():
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                values.append(token)
    return " ".join(values)


def _clean_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _make_csv_rows(output_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for record in output_records:
        for candidate in record["rewrite_candidates"]:
            rows.append(
                {
                    "dataset": record["dataset"],
                    "qid": record["qid"],
                    "question": record["question"],
                    "failure_label": record["failure_label"],
                    "rewrite_type": candidate["rewrite_type"],
                    "rewrite_query": candidate["rewrite_query"],
                }
            )
    return rows


def _validate_outputs(
    input_records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
    csv_rows: list[dict[str, str]],
) -> list[str]:
    if len(output_records) != len(input_records):
        raise ValueError(f"Output JSONL record count {len(output_records)} != input count {len(input_records)}")
    if len(csv_rows) != len(input_records) * 5:
        raise ValueError(f"Output CSV row count {len(csv_rows)} != expected {len(input_records) * 5}")

    warnings = []
    for record in output_records:
        candidates = record.get("rewrite_candidates", [])
        if len(candidates) != 5:
            raise ValueError(f"qid={record.get('qid')} has {len(candidates)} rewrite candidates, expected 5")
        rewrite_types = {candidate.get("rewrite_type") for candidate in candidates}
        if rewrite_types != REWRITE_TYPES:
            raise ValueError(f"qid={record.get('qid')} rewrite_type set is invalid: {rewrite_types}")
        ordered_types = [candidate.get("rewrite_type") for candidate in candidates]
        if ordered_types != REWRITE_TYPE_ORDER:
            raise ValueError(f"qid={record.get('qid')} rewrite_type order is invalid: {ordered_types}")
        for candidate in candidates:
            if not str(candidate.get("rewrite_query", "")).strip():
                warnings.append(f"Warning: empty rewrite_query qid={record.get('qid')} type={candidate.get('rewrite_type')}")
    return warnings


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = ["dataset", "qid", "question", "failure_label", "rewrite_type", "rewrite_query"]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(output_records: list[dict[str, Any]], output_jsonl: Path, output_csv: Path) -> dict[str, Any]:
    candidates = [candidate for record in output_records for candidate in record["rewrite_candidates"]]
    return {
        "total_records": len(output_records),
        "total_rewrite_candidates": len(candidates),
        "rewrite_type_counts": dict(Counter(candidate["rewrite_type"] for candidate in candidates)),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
    }


def _print_summary(
    summary: dict[str, Any],
    warnings: list[str],
    llm_api_calls: int,
    llm_fallback_count: int,
    fallback_examples: list[str],
) -> None:
    for warning in warnings:
        print(warning)
    if fallback_examples:
        print("LLM fallback examples:")
        for example in fallback_examples:
            print(f"- {example}")
    print("Rewrite generation summary")
    print(f"total_records: {summary['total_records']}")
    print(f"total_rewrite_candidates: {summary['total_rewrite_candidates']}")
    print(f"rewrite_type_counts: {summary['rewrite_type_counts']}")
    print(f"llm_api_calls: {llm_api_calls}")
    print(f"llm_fallback_count: {llm_fallback_count}")
    print(f"output_jsonl: {summary['output_jsonl']}")
    print(f"output_csv: {summary['output_csv']}")


def _wait_for_request_slot(last_request_at: float, delay_seconds: float) -> float:
    if delay_seconds <= 0 or last_request_at <= 0:
        return last_request_at
    elapsed = time.monotonic() - last_request_at
    remaining = delay_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)
    return last_request_at


if __name__ == "__main__":
    main()
