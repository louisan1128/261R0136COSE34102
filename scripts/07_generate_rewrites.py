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
from src.utils.io import ensure_dir, read_jsonl, read_yaml
from src.utils.text import extract_keywords, tokenize


DEFAULT_INPUT_PATH = Path("data/outputs/annotation/hard_subset_1000_annotation_final.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/outputs/rewrite_candidates")
OUTPUT_JSONL_NAME = "hard_subset_1000_rewrite_candidates.jsonl"
OUTPUT_CSV_NAME = "hard_subset_1000_rewrite_candidates.csv"
SUMMARY_NAME = "rewrite_candidate_summary_1000.json"

REWRITE_TYPE_ORDER = [
    "original",
    "keyword_rewrite",
    "semantic_rewrite",
    "structured_rewrite",
    "llm_rewrite",
]
REWRITE_TYPES = set(REWRITE_TYPE_ORDER)

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
    "일까요",
    "나요",
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
    "으로",
    "로",
    "와",
    "과",
    "하고",
    "및",
}

LABEL_HINTS = {
    "lexical_mismatch": {
        "keyword": "핵심 표현",
        "semantic": "같은 의미를 다른 표현으로 바꾸어",
        "structured": "속성",
        "llm": "검색 표현 차이를 줄여",
    },
    "missing_key_term": {
        "keyword": "대상 단서",
        "semantic": "생략된 대상과 의도를 명확히 하여",
        "structured": "요구정보",
        "llm": "검색 대상이 드러나도록",
    },
    "numeric_temporal_mismatch": {
        "keyword": "숫자 날짜 연도 순위",
        "semantic": "몇 년·언제·몇 위·몇 명 같은 조건을 명확히 하여",
        "structured": "시간조건",
        "llm": "특정 숫자나 시점을 찾도록",
    },
    "entity_mismatch": {
        "keyword": "entity 관계 역할",
        "semantic": "인물·기관·작품 혼동을 줄이도록",
        "structured": "관계",
        "llm": "entity 혼동을 피하도록",
    },
    "semantic_mismatch": {
        "keyword": "대상 질문유형",
        "semantic": "원인·결과·방법·역할 등 질문 의도를 강조하여",
        "structured": "질문유형",
        "llm": "질문 의도가 분명해지도록",
    },
    "context_boundary_issue": {
        "keyword": "표 목록 행 열 속성",
        "semantic": "표나 목록에서 필요한 정보를 찾는 형태로",
        "structured": "조건",
        "llm": "표·목록 구조의 근거를 찾도록",
    },
    "ambiguous": {
        "keyword": "핵심 명사",
        "semantic": "원래 의미를 유지하여",
        "structured": "가능범위",
        "llm": "일반적인 검색 질문으로",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate five retrieval rewrite candidates per hard case.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Final annotation JSONL path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Rewrite candidate output directory.")
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rewrite candidate files. Use this when regenerating with real LLM rewrites.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_jsonl = output_dir / OUTPUT_JSONL_NAME
    output_csv = output_dir / OUTPUT_CSV_NAME
    output_summary = output_dir / SUMMARY_NAME

    if not input_path.exists():
        raise FileNotFoundError(f"Annotation final v2 file not found: {input_path}")

    config = read_yaml(root / "configs" / "default.yaml")
    records = read_jsonl(input_path)
    if len(records) != 1000:
        raise ValueError(f"Input row count must be 1000 for hard_subset_1000, got {len(records)}")

    rewriter, rewrite_cache, cache_path, llm_delay_seconds = _maybe_build_rewriter(config, args)
    output_records = []
    llm_api_calls = 0
    llm_fallback_count = 0
    fallback_examples: list[str] = []
    last_llm_request_at = 0.0

    for record in records:
        llm_query = None
        question = _clean_query(record.get("question", ""))
        failure_label = _primary_label(record)
        secondary_label = _clean_query(record.get("secondary_failure_label", ""))
        question_type = _clean_query(record.get("question_type", ""))

        if rewriter is not None and question:
            cache_key = _cache_key(question, failure_label, secondary_label, question_type)
            llm_query = rewrite_cache.get(cache_key)
            if not llm_query:
                try:
                    last_llm_request_at = _wait_for_request_slot(last_llm_request_at, llm_delay_seconds)
                    llm_query = rewriter.rewrite(
                        question,
                        f"{failure_label}; secondary={secondary_label or 'none'}; question_type={question_type or 'unknown'}",
                    )
                    last_llm_request_at = time.monotonic()
                    rewrite_cache[cache_key] = llm_query
                    llm_api_calls += 1
                    save_rewrite_cache(rewrite_cache, cache_path)
                except Exception as exc:
                    llm_query = None
                    llm_fallback_count += 1
                    if len(fallback_examples) < 5:
                        fallback_examples.append(f"qid={record.get('qid')}: {exc}")

        output_records.append(_make_output_record(record, llm_query))

    csv_rows = _make_csv_rows(output_records)
    validation = _validate_outputs(records, output_records, csv_rows)
    summary = _build_summary(output_records, validation, output_jsonl, output_csv, llm_api_calls, llm_fallback_count)

    _write_jsonl(output_records, output_jsonl, overwrite=args.overwrite)
    _write_csv(csv_rows, output_csv, overwrite=args.overwrite)
    _write_json(summary, output_summary, overwrite=args.overwrite)
    if rewriter is not None:
        save_rewrite_cache(rewrite_cache, cache_path)

    _print_summary(summary, output_summary, llm_api_calls, llm_fallback_count, fallback_examples)


def _maybe_build_rewriter(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[OpenAICompatibleRewriter | None, dict[str, str], str, float]:
    llm_config = dict(config.get("llm_rewrite", {}))
    api_key_env = str(llm_config.get("api_key_env", "OPENAI_API_KEY"))
    has_api_key = bool(os.environ.get(api_key_env, "").strip())
    cache_path = str(llm_config.get("cache_path", "data/outputs/cache/llm_rewrite_cache.jsonl"))
    if args.no_external_llm or not has_api_key:
        return None, {}, cache_path, 0.0

    llm_config.setdefault("base_url", "https://api.openai.com/v1")
    llm_config.setdefault("model_env", "OPENAI_MODEL")
    if not os.environ.get(str(llm_config.get("model_env", "OPENAI_MODEL")), "").strip() and not llm_config.get("model"):
        print("Warning: OpenAI API key found, but no OPENAI_MODEL or llm_rewrite.model is configured; using fallback.")
        return None, {}, cache_path, 0.0

    try:
        rewriter = OpenAICompatibleRewriter.from_config(llm_config)
    except Exception as exc:
        print(f"Warning: could not initialize external LLM rewriter; using fallback. Reason: {exc}")
        return None, {}, cache_path, 0.0

    delay = args.llm_delay_seconds if args.llm_delay_seconds is not None else float(llm_config.get("request_delay_seconds", 0.0))
    print(f"External LLM rewrite enabled with model '{rewriter.config.model}'.")
    return rewriter, load_rewrite_cache(cache_path), cache_path, delay


def _make_output_record(record: dict[str, Any], llm_query: str | None) -> dict[str, Any]:
    question = _clean_query(record.get("question", ""))
    answer = _clean_query(record.get("answer", ""))
    failure_label = _primary_label(record)
    secondary_label = _clean_query(record.get("secondary_failure_label", ""))
    question_type = _clean_query(record.get("question_type", ""))

    raw_candidates = [
        ("original", question),
        ("keyword_rewrite", _keyword_rewrite(question, failure_label, question_type)),
        ("semantic_rewrite", _semantic_rewrite(question, failure_label, question_type)),
        ("structured_rewrite", _structured_rewrite(question, failure_label, question_type)),
        ("llm_rewrite", _clean_query(llm_query) or _llm_fallback_rewrite(question, failure_label, question_type)),
    ]
    candidates = []
    for rewrite_type, query in raw_candidates:
        safe_query = query if rewrite_type == "original" else _prevent_answer_leakage(query, answer, question)
        candidates.append({"rewrite_type": rewrite_type, "query": _clean_query(safe_query) or question})

    return {
        "qid": record.get("qid", ""),
        "dataset": record.get("dataset", ""),
        "question": question,
        "answer": answer,
        "failure_label": failure_label,
        "secondary_failure_label": secondary_label,
        "rewrite_candidates": candidates,
    }


def _primary_label(record: dict[str, Any]) -> str:
    label = _clean_query(record.get("failure_label", ""))
    return label if label in LABEL_HINTS else "ambiguous"


def _keyword_rewrite(question: str, failure_label: str, question_type: str) -> str:
    keywords = _question_keywords(question)
    numeric = _numeric_terms(question)
    hint = LABEL_HINTS[failure_label]["keyword"].split()
    intent = _intent_keyword(question, question_type)

    if failure_label == "numeric_temporal_mismatch":
        parts = keywords[:6] + numeric + hint + [intent]
    elif failure_label == "context_boundary_issue":
        parts = keywords[:6] + hint + [intent]
    elif failure_label == "ambiguous":
        parts = keywords[:7] + [intent]
    else:
        parts = keywords[:8] + [intent]
    return _join_unique(parts) or question


def _semantic_rewrite(question: str, failure_label: str, question_type: str) -> str:
    prefix = LABEL_HINTS[failure_label]["semantic"]
    intent = _intent_phrase(question, question_type)
    if failure_label == "numeric_temporal_mismatch":
        return f"{prefix} {intent}를 찾는 질문: {question}"
    if failure_label == "entity_mismatch":
        return f"{prefix} 대상과 관계를 분명히 한 질문: {question}"
    if failure_label == "context_boundary_issue":
        return f"{prefix} {intent}를 확인하는 질문: {question}"
    if failure_label == "missing_key_term":
        return f"{prefix} {intent}를 찾는 질문: {question}"
    return f"{prefix} {intent}를 찾는 질문: {question}"


def _structured_rewrite(question: str, failure_label: str, question_type: str) -> str:
    keywords = _question_keywords(question)
    numeric = _numeric_terms(question)
    target = " ".join(keywords[:4]) if keywords else "질문 대상"
    requirement = _intent_phrase(question, question_type)
    condition = _join_unique(numeric) or LABEL_HINTS[failure_label]["structured"]
    if failure_label == "entity_mismatch":
        condition = f"{condition} / entity 혼동 방지"
    elif failure_label == "context_boundary_issue":
        condition = f"{condition} / 표 목록 행 열 확인"
    elif failure_label == "missing_key_term":
        condition = f"{condition} / 대상 명확화"
    return f"대상: {target} / 조건: {condition} / 질문유형: {requirement}"


def _llm_fallback_rewrite(question: str, failure_label: str, question_type: str) -> str:
    intent = _intent_phrase(question, question_type)
    hint = LABEL_HINTS[failure_label]["llm"]
    keywords = _question_keywords(question)
    if keywords:
        return f"{' '.join(keywords[:6])} {hint} {intent} 근거 검색"
    return f"{hint} {intent} 근거 검색: {question}"


def _question_keywords(question: str) -> list[str]:
    extracted = extract_keywords(question, max_keywords=12)
    candidates = [token for token in extracted.split() if _valid_keyword(token)]
    if not candidates:
        candidates = [token for token in tokenize(question) if _valid_keyword(token)]
    return _unique(candidates)


def _valid_keyword(token: str) -> bool:
    token = token.strip()
    return bool(token and len(token) > 1 and token not in STOPWORDS and not _is_question_ending(token))


def _numeric_terms(question: str) -> list[str]:
    return _unique(
        [
            token
            for token in tokenize(question)
            if re.search(r"\d|몇|년|월|일|번째|번|위|%|명|개|시간|초|분|원|조|억|세|순위|기간", token)
        ]
    )


def _intent_keyword(question: str, question_type: str) -> str:
    return _intent_phrase(question, question_type).replace("·", " ")


def _intent_phrase(question: str, question_type: str) -> str:
    if question_type in {"numeric", "when"} or re.search(r"몇|언제|연도|년도|날짜|시기|몇 년|몇 월|몇 일|순위|몇 위", question):
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
    if question_type == "yes_no":
        return "참거짓 판단 정보"
    return "정답 정보"


def _prevent_answer_leakage(query: str, answer: str, fallback_question: str) -> str:
    cleaned = _clean_query(query)
    answer = _clean_query(answer)
    if not cleaned or not answer:
        return cleaned

    cleaned = _remove_answer_text(cleaned, answer)
    cleaned = _clean_query(cleaned)
    if not cleaned or _has_answer_leakage(cleaned, answer):
        fallback_keywords = [token for token in _question_keywords(fallback_question) if not _has_answer_leakage(token, answer)]
        cleaned = _join_unique(fallback_keywords[:6]) or _remove_answer_text(fallback_question, answer)
    return _clean_query(cleaned)


def _remove_answer_text(query: str, answer: str) -> str:
    variants = _answer_variants(answer)
    cleaned = query
    for variant in variants:
        if not variant:
            continue
        if _is_numeric_answer(variant):
            cleaned = re.sub(rf"(?<!\d){re.escape(variant)}(?!\d)", " ", cleaned)
        elif re.search(r"[A-Za-z]", variant):
            cleaned = re.sub(rf"\b{re.escape(variant)}\b", " ", cleaned, flags=re.I)
        else:
            cleaned = cleaned.replace(variant, " ")
    return _clean_query(cleaned)


def _has_answer_leakage(query: str, answer: str) -> bool:
    query = _clean_query(query)
    answer = _clean_query(answer)
    if not query or not answer:
        return False
    for variant in _answer_variants(answer):
        if not variant:
            continue
        if _is_numeric_answer(variant):
            if re.search(rf"(?<!\d){re.escape(variant)}(?!\d)", query):
                return True
        elif re.search(r"[A-Za-z]", variant):
            if re.search(rf"\b{re.escape(variant)}\b", query, flags=re.I):
                return True
        elif len(variant) >= 2 and variant in query:
            return True
    return False


def _answer_variants(answer: str) -> list[str]:
    answer = _clean_query(answer)
    variants = [answer]
    if "," in answer:
        variants.extend(part.strip() for part in answer.split(","))
    if "/" in answer:
        variants.extend(part.strip() for part in answer.split("/"))
    return _unique([variant for variant in variants if len(variant) >= 2])


def _is_numeric_answer(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", value.strip()))


def _make_csv_rows(output_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for record in output_records:
        for candidate in record["rewrite_candidates"]:
            rows.append(
                {
                    "qid": record["qid"],
                    "dataset": record["dataset"],
                    "question": record["question"],
                    "failure_label": record["failure_label"],
                    "secondary_failure_label": record["secondary_failure_label"],
                    "rewrite_type": candidate["rewrite_type"],
                    "query": candidate["query"],
                }
            )
    return rows


def _validate_outputs(
    input_records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
    csv_rows: list[dict[str, str]],
) -> dict[str, int]:
    invalid_candidate_count = 0
    empty_query_count = 0
    answer_leakage_count = 0

    if len(output_records) != len(input_records):
        raise ValueError(f"Output JSONL record count {len(output_records)} != input count {len(input_records)}")
    expected_csv_rows = len(input_records) * len(REWRITE_TYPE_ORDER)
    if len(csv_rows) != expected_csv_rows:
        raise ValueError(f"CSV row count must be {expected_csv_rows}, got {len(csv_rows)}")

    for input_record, output_record in zip(input_records, output_records):
        question = _clean_query(input_record.get("question", ""))
        answer = _clean_query(input_record.get("answer", ""))
        candidates = output_record.get("rewrite_candidates", [])

        if len(candidates) != 5:
            invalid_candidate_count += 1
            continue
        ordered_types = [candidate.get("rewrite_type") for candidate in candidates]
        if ordered_types != REWRITE_TYPE_ORDER or set(ordered_types) != REWRITE_TYPES:
            invalid_candidate_count += 1
        if candidates[0].get("query") != question:
            invalid_candidate_count += 1

        for candidate in candidates:
            query = _clean_query(candidate.get("query", ""))
            if not query:
                empty_query_count += 1
            if candidate.get("rewrite_type") != "original" and _has_answer_leakage(query, answer):
                answer_leakage_count += 1

    if invalid_candidate_count:
        raise ValueError(f"Invalid rewrite candidate records found: {invalid_candidate_count}")
    if empty_query_count:
        raise ValueError(f"Empty rewrite queries found: {empty_query_count}")
    if answer_leakage_count:
        raise ValueError(f"Answer leakage found in generated rewrites: {answer_leakage_count}")

    return {
        "empty_query_count": empty_query_count,
        "invalid_candidate_count": invalid_candidate_count,
        "answer_leakage_count": answer_leakage_count,
    }


def _build_summary(
    output_records: list[dict[str, Any]],
    validation: dict[str, int],
    output_jsonl: Path,
    output_csv: Path,
    llm_api_calls: int = 0,
    llm_fallback_count: int = 0,
) -> dict[str, Any]:
    candidates = [candidate for record in output_records for candidate in record["rewrite_candidates"]]
    summary = {
        "total_questions": len(output_records),
        "total_rewrite_candidates": len(candidates),
        "rewrite_type_distribution": dict(Counter(candidate["rewrite_type"] for candidate in candidates)),
        "failure_label_distribution": dict(Counter(record["failure_label"] for record in output_records)),
        "dataset_distribution": dict(Counter(record["dataset"] for record in output_records)),
        "empty_query_count": validation["empty_query_count"],
        "invalid_candidate_count": validation["invalid_candidate_count"],
        "answer_leakage_count": validation["answer_leakage_count"],
        "llm_api_calls": llm_api_calls,
        "llm_fallback_count": llm_fallback_count,
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
    }
    return summary


def _write_jsonl(records: list[dict[str, Any]], path: Path, overwrite: bool = False) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    payload = "\n".join(lines) + "\n"
    _write_text(payload, path, overwrite=overwrite)


def _write_csv(rows: list[dict[str, str]], path: Path, overwrite: bool = False) -> None:
    fieldnames = ["qid", "dataset", "question", "failure_label", "secondary_failure_label", "rewrite_type", "query"]
    ensure_dir(path.parent)
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _write_text(buffer.getvalue(), path, overwrite=overwrite)


def _write_json(payload: dict[str, Any], path: Path, overwrite: bool = False) -> None:
    _write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", path, overwrite=overwrite)


def _write_text(payload: str, path: Path, overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == payload:
            print(f"Existing identical file kept: {path}")
            return
        if _normalize_newlines(current) == _normalize_newlines(payload):
            print(f"Existing equivalent file kept: {path}")
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}; rerun with --overwrite to replace it.")
    with path.open("w", encoding="utf-8", newline="") as fout:
        fout.write(payload)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _print_summary(
    summary: dict[str, Any],
    output_summary: Path,
    llm_api_calls: int,
    llm_fallback_count: int,
    fallback_examples: list[str],
) -> None:
    if fallback_examples:
        print("LLM fallback examples:")
        for example in fallback_examples:
            print(f"- {example}")
    print("Rewrite candidate generation summary")
    print(f"total_questions: {summary['total_questions']}")
    print(f"total_rewrite_candidates: {summary['total_rewrite_candidates']}")
    print(f"rewrite_type_distribution: {summary['rewrite_type_distribution']}")
    print(f"failure_label_distribution: {summary['failure_label_distribution']}")
    print(f"dataset_distribution: {summary['dataset_distribution']}")
    print(f"empty_query_count: {summary['empty_query_count']}")
    print(f"invalid_candidate_count: {summary['invalid_candidate_count']}")
    print(f"answer_leakage_count: {summary['answer_leakage_count']}")
    print(f"llm_api_calls: {llm_api_calls}")
    print(f"llm_fallback_count: {llm_fallback_count}")
    print(f"output_jsonl: {summary['output_jsonl']}")
    print(f"output_csv: {summary['output_csv']}")
    print(f"output_summary: {output_summary}")


def _wait_for_request_slot(last_request_at: float, delay_seconds: float) -> float:
    if delay_seconds <= 0 or last_request_at <= 0:
        return last_request_at
    elapsed = time.monotonic() - last_request_at
    remaining = delay_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)
    return last_request_at


def _cache_key(question: str, failure_label: str, secondary_label: str, question_type: str) -> str:
    # Cache by question text so earlier LLM rewrites can be reused when
    # the hard-case subset is expanded or regenerated.
    return question


def _is_question_ending(token: str) -> bool:
    return bool(re.search(r"(인가|인가요|되는가|되나요|했는가|했나요|일까|일까요|나요|까|인가\\?)$", token))


def _join_unique(parts: list[str]) -> str:
    return " ".join(_unique([str(part).strip() for part in parts if str(part).strip()]))


def _unique(parts: list[str]) -> list[str]:
    values = []
    seen = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            values.append(part)
    return values


def _clean_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


if __name__ == "__main__":
    main()
