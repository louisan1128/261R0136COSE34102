import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RAW_CHUNKS_PATH = ROOT / "data" / "raw" / "korquad2_chunks.jsonl"
PROCESSED_CORPUS_PATH = ROOT / "data" / "processed" / "korquad2_filtered_corpus.jsonl"
OUTPUT_PATH = ROOT / "data" / "korquad2_chunk_diagnosis.json"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc


def text_len(row):
    return len(str(row.get("text") or "").strip())


def length_distribution(lengths):
    bins = [
        ("empty", 0, 0),
        ("1-19", 1, 19),
        ("20-49", 20, 49),
        ("50-99", 50, 99),
        ("100-199", 100, 199),
        ("200-499", 200, 499),
        ("500-999", 500, 999),
        ("1000+", 1000, None),
    ]
    counts = {}
    for label, lower, upper in bins:
        if upper is None:
            counts[label] = sum(1 for length in lengths if length >= lower)
        else:
            counts[label] = sum(1 for length in lengths if lower <= length <= upper)
    return counts


def processed_pid_to_raw_chunk_id(pid):
    if not pid:
        return None
    pid = str(pid)
    if pid.startswith("korquad2_"):
        return pid[len("korquad2_") :]
    return pid


def majority_conclusion(synthetic_true, synthetic_false, real_paragraph_like, total):
    if total == 0:
        return "KorQuAD2 chunk diagnosis could not determine retrieval context"

    synthetic_ratio = synthetic_true / total
    real_paragraph_ratio = real_paragraph_like / total

    if synthetic_ratio > 0.5:
        return "KorQuAD2 retrieval is using synthetic chunks"
    if real_paragraph_ratio > 0.5:
        return "KorQuAD2 retrieval is using real context"
    return "KorQuAD2 retrieval context is mixed or ambiguous"


def main():
    os.chdir(ROOT)
    sys.path.append(str(ROOT))

    raw_rows = list(read_jsonl(RAW_CHUNKS_PATH))
    raw_lengths = [text_len(row) for row in raw_rows]
    raw_by_chunk_id = {str(row.get("chunk_id")): row for row in raw_rows if row.get("chunk_id")}

    # Text lookup is a fallback only. Duplicate text can exist, so keep the full synthetic flag set.
    raw_synthetic_by_text = {}
    for row in raw_rows:
        text = str(row.get("text") or "").strip()
        raw_synthetic_by_text.setdefault(text, set()).add(bool(row.get("synthetic_from_qa", False)))

    synthetic_counter = Counter(bool(row.get("synthetic_from_qa", False)) for row in raw_rows)
    total_chunks = len(raw_rows)
    short_text_count = sum(1 for length in raw_lengths if length < 20)
    real_paragraph_like_count = sum(1 for length in raw_lengths if length >= 50)
    empty_text_count = sum(1 for length in raw_lengths if length == 0)

    corpus_rows = list(read_jsonl(PROCESSED_CORPUS_PATH))
    corpus_comparison = Counter()
    corpus_length_values = []
    unmatched_examples = []

    for row in corpus_rows:
        corpus_length_values.append(text_len(row))
        raw_chunk_id = processed_pid_to_raw_chunk_id(row.get("pid") or row.get("doc_id"))
        raw_row = raw_by_chunk_id.get(raw_chunk_id)

        if raw_row is not None:
            corpus_comparison["matched_by_id"] += 1
            corpus_comparison[f"synthetic_{bool(raw_row.get('synthetic_from_qa', False))}"] += 1
            continue

        text = str(row.get("text") or "").strip()
        text_flags = raw_synthetic_by_text.get(text)
        if text_flags:
            corpus_comparison["matched_by_text"] += 1
            if len(text_flags) == 1:
                synthetic_flag = next(iter(text_flags))
                corpus_comparison[f"synthetic_{synthetic_flag}"] += 1
            else:
                corpus_comparison["synthetic_ambiguous"] += 1
            continue

        corpus_comparison["unmatched"] += 1
        if len(unmatched_examples) < 5:
            unmatched_examples.append(
                {
                    "pid": row.get("pid"),
                    "doc_id": row.get("doc_id"),
                    "text_preview": text[:120],
                }
            )

    corpus_total = len(corpus_rows)
    corpus_synthetic_true = corpus_comparison["synthetic_True"]
    corpus_synthetic_false = corpus_comparison["synthetic_False"]
    corpus_real_paragraph_like = sum(1 for length in corpus_length_values if length >= 50)
    corpus_conclusion = majority_conclusion(
        corpus_synthetic_true,
        corpus_synthetic_false,
        corpus_real_paragraph_like,
        corpus_total,
    )

    diagnosis = {
        "raw_chunks_path": str(RAW_CHUNKS_PATH.relative_to(ROOT)),
        "processed_corpus_path": str(PROCESSED_CORPUS_PATH.relative_to(ROOT)),
        "total_chunks": total_chunks,
        "synthetic_true": synthetic_counter[True],
        "synthetic_false": synthetic_counter[False],
        "avg_text_length": round(mean(raw_lengths), 2) if raw_lengths else 0,
        "min_text_length": min(raw_lengths) if raw_lengths else 0,
        "max_text_length": max(raw_lengths) if raw_lengths else 0,
        "text_length_distribution": length_distribution(raw_lengths),
        "empty_text": empty_text_count,
        "title_level_text_lt20": short_text_count,
        "real_paragraph_like_gte50": real_paragraph_like_count,
        "processed_corpus_comparison": {
            "total_corpus_rows": corpus_total,
            "matched_by_id": corpus_comparison["matched_by_id"],
            "matched_by_text": corpus_comparison["matched_by_text"],
            "unmatched": corpus_comparison["unmatched"],
            "synthetic_true": corpus_synthetic_true,
            "synthetic_false": corpus_synthetic_false,
            "synthetic_ambiguous": corpus_comparison["synthetic_ambiguous"],
            "avg_text_length": round(mean(corpus_length_values), 2) if corpus_length_values else 0,
            "min_text_length": min(corpus_length_values) if corpus_length_values else 0,
            "max_text_length": max(corpus_length_values) if corpus_length_values else 0,
            "text_length_distribution": length_distribution(corpus_length_values),
            "empty_text": sum(1 for length in corpus_length_values if length == 0),
            "short_text_lt20": sum(1 for length in corpus_length_values if length < 20),
            "real_paragraph_like_gte50": corpus_real_paragraph_like,
            "conclusion": corpus_conclusion,
            "unmatched_examples": unmatched_examples,
        },
        "conclusion": corpus_conclusion,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(diagnosis, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("KorQuAD2 chunk diagnosis")
    print(f"total_chunks: {diagnosis['total_chunks']}")
    print(f"synthetic_true: {diagnosis['synthetic_true']}")
    print(f"synthetic_false: {diagnosis['synthetic_false']}")
    print(f"avg_text_length: {diagnosis['avg_text_length']}")
    print(f"short_text(<20): {diagnosis['title_level_text_lt20']}")
    print(f"real_paragraph_like(>50): {diagnosis['real_paragraph_like_gte50']}")
    print(f"conclusion: {diagnosis['conclusion']}")


if __name__ == "__main__":
    main()
