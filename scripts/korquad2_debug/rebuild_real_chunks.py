import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = ROOT / "data" / "raw" / "KorQuAD_2.0"
FILTERED_QA_PATH = ROOT / "data" / "processed" / "korquad2_filtered_qa_pairs.jsonl"
RAW_CHUNKS_PATH = ROOT / "data" / "raw" / "korquad2_chunks.jsonl"
FILTERED_CORPUS_PATH = ROOT / "data" / "processed" / "korquad2_filtered_corpus.jsonl"
SUMMARY_PATH = ROOT / "data" / "korquad2_real_chunk_rebuild_summary.json"
KORQUAD2_METADATA_QA_PATH = ROOT / "data" / "processed" / "qa_pairs_2.jsonl"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 80

TEXT_FIELDS = ("context", "html", "raw_html", "document", "text", "paragraph", "source")
TITLE_FIELDS = ("title", "page_title", "document_title")
URL_FIELDS = ("url", "wiki_url", "source_url")
DOC_ID_FIELDS = ("doc_id", "document_id", "source_doc_id", "id", "guid", "wiki_id", "page_id")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: Any) -> str:
    return " ".join(str(text or "").split())


def normalize_key(text: Any) -> str:
    return normalize_space(html.unescape(str(text or "")).replace("_", " ")).casefold()


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", normalize_key(text))


def html_to_text(value: Any) -> str:
    soup = BeautifulSoup(str(value or ""), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return normalize_space(soup.get_text(" "))


def clean_text(value: Any) -> str:
    value = html.unescape(str(value or ""))
    if "<" in value and ">" in value:
        return html_to_text(value)
    return normalize_space(value)


def clean_answer(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "answer", "value"):
            if key in value:
                return clean_answer(value[key])
        return ""
    if isinstance(value, list):
        return clean_answer(value[0]) if value else ""
    return clean_text(value)


def first_string(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return normalize_space(value)
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def discover_original_files() -> list[Path]:
    if not ORIGINAL_DIR.exists():
        return []
    return sorted(path for path in ORIGINAL_DIR.rglob("*") if path.suffix.lower() in {".json", ".jsonl"})


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
                if isinstance(row, dict):
                    yield row
        return

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    def walk(value: Any) -> Iterable[dict[str, Any]]:
        if isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, dict):
            if "question" in value and any(field in value for field in TEXT_FIELDS):
                yield value
            for child_key in ("data", "articles", "documents", "paragraphs", "items", "records"):
                child = value.get(child_key)
                if child is not None:
                    yield from walk(child)

    yield from walk(obj)


def extract_context(row: dict[str, Any]) -> str:
    parts = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(clean_text(value))
    return normalize_space(" ".join(part for part in parts if part))


def chunk_text(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalize_space(text[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def contains_answer(chunk: str, answer: str) -> bool:
    answer = clean_answer(answer)
    if not answer:
        return False
    return normalize_key(answer) in normalize_key(chunk) or compact_text(answer) in compact_text(chunk)


def load_legacy_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if RAW_CHUNKS_PATH.exists():
        for row in read_jsonl(RAW_CHUNKS_PATH):
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id:
                metadata[f"korquad2_{chunk_id}"] = {
                    "chunk_id": chunk_id,
                    "doc_id": row.get("doc_id", ""),
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "text": row.get("text", ""),
                }
    if FILTERED_CORPUS_PATH.exists():
        for row in read_jsonl(FILTERED_CORPUS_PATH):
            pid = str(row.get("pid") or "")
            if not pid:
                continue
            record = metadata.setdefault(pid, {})
            record.setdefault("chunk_id", pid.removeprefix("korquad2_"))
            record.setdefault("doc_id", row.get("source_doc_id", ""))
            record.setdefault("title", row.get("title", ""))
            record.setdefault("text", row.get("text", ""))
    return metadata


def load_original_id_metadata(filtered_qas: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Use the full KorQuAD2 QA file only as metadata to map filtered questions to original row ids."""
    if not KORQUAD2_METADATA_QA_PATH.exists():
        return {}

    wanted_questions = {normalize_key(row.get("question", "")) for row in filtered_qas}
    metadata = {}
    for row in read_jsonl(KORQUAD2_METADATA_QA_PATH):
        question_key = normalize_key(row.get("question", ""))
        if question_key not in wanted_questions:
            continue
        metadata[question_key] = {
            "source_id": str(row.get("qid") or ""),
            "doc_id": str(row.get("doc_id") or ""),
            "chunk_id": str(row.get("gold_chunk_id") or ""),
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
        }
    return metadata


def qa_metadata(
    qa: dict[str, Any],
    legacy: dict[str, dict[str, Any]],
    original_id_metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    pid = str(qa.get("gold_pid") or qa.get("gold_doc_id") or "")
    old = legacy.get(pid, {})
    original = (original_id_metadata or {}).get(normalize_key(qa.get("question", "")), {})
    return {
        "source_id": str(original.get("source_id") or ""),
        "chunk_id": str(qa.get("gold_chunk_id") or original.get("chunk_id") or old.get("chunk_id") or pid.removeprefix("korquad2_")),
        "doc_id": str(qa.get("doc_id") or original.get("doc_id") or old.get("doc_id") or ""),
        "title": str(qa.get("title") or original.get("title") or old.get("title") or ""),
        "url": str(qa.get("url") or original.get("url") or old.get("url") or ""),
        "text": str(old.get("text") or ""),
    }


def build_original_docs(
    files: list[Path],
    filtered_qas: list[dict[str, Any]],
    legacy: dict[str, dict[str, Any]],
    original_id_metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    wanted_questions = {normalize_key(row.get("question", "")) for row in filtered_qas}
    wanted_source_ids = {
        meta["source_id"]
        for meta in original_id_metadata.values()
        if meta.get("source_id")
    }

    docs = []
    indexes: dict[str, list[int]] = defaultdict(list)
    seen_contexts: dict[str, int] = {}

    for path in files:
        for row in iter_json_records(path):
            question = normalize_space(row.get("question", ""))
            question_key = normalize_key(question)
            source_id = str(row.get("id") or row.get("qid") or "")
            if source_id not in wanted_source_ids and question_key not in wanted_questions:
                continue

            raw_context = ""
            for field in TEXT_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    raw_context = value
                    break
            if not raw_context:
                continue

            context_key = compact_text(raw_context[:5000])
            doc_idx = seen_contexts.get(context_key)
            if doc_idx is None:
                context = extract_context(row)
                if not context:
                    continue
                doc_idx = len(docs)
                seen_contexts[context_key] = doc_idx
                docs.append(
                    {
                        "doc_id": first_string(row, DOC_ID_FIELDS),
                        "title": first_string(row, TITLE_FIELDS),
                        "url": first_string(row, URL_FIELDS),
                        "context": context,
                        "chunks": chunk_text(context),
                        "source_file": str(path.relative_to(ROOT)),
                    }
                )

            if question:
                indexes[f"question:{question_key}"].append(doc_idx)
            if source_id:
                indexes[f"source_id:{normalize_key(source_id)}"].append(doc_idx)
            title = first_string(row, TITLE_FIELDS)
            if title:
                indexes[f"title:{normalize_key(title)}"].append(doc_idx)
            url = first_string(row, URL_FIELDS)
            if url:
                indexes[f"url:{normalize_key(url)}"].append(doc_idx)
            doc_id = first_string(row, DOC_ID_FIELDS)
            if doc_id:
                indexes[f"doc_id:{normalize_key(doc_id)}"].append(doc_idx)

    for key, values in list(indexes.items()):
        deduped = []
        seen = set()
        for value in values:
            if value not in seen:
                deduped.append(value)
                seen.add(value)
        indexes[key] = deduped

    return docs, indexes


def candidate_docs(qa: dict[str, Any], meta: dict[str, str], indexes: dict[str, list[int]]) -> list[int]:
    keys = [
        f"source_id:{normalize_key(meta.get('source_id', ''))}",
        f"question:{normalize_key(qa.get('question', ''))}",
        f"title:{normalize_key(meta.get('title', ''))}",
        f"url:{normalize_key(meta.get('url', ''))}",
        f"doc_id:{normalize_key(meta.get('doc_id', ''))}",
    ]
    ordered = []
    seen = set()
    for key in keys:
        if key.endswith(":"):
            continue
        for idx in indexes.get(key, []):
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)
    return ordered


def best_answer_chunk(doc: dict[str, Any], answer: str) -> tuple[int, str] | None:
    matches = [(idx, chunk) for idx, chunk in enumerate(doc["chunks"]) if contains_answer(chunk, answer)]
    if not matches:
        return None
    return sorted(matches, key=lambda item: (len(item[1]), item[0]))[0]


def fallback_text(qa: dict[str, Any], meta: dict[str, str]) -> str:
    if meta["text"]:
        return normalize_space(meta["text"])
    return normalize_space(f"{meta['title'].replace('_', ' ')} {clean_answer(qa.get('answer', ''))}")


def main() -> None:
    os.chdir(ROOT)
    sys.path.append(str(ROOT))

    original_files = discover_original_files()
    if not original_files:
        raise FileNotFoundError(
            "KorQuAD2 original context file not found. "
            "Please place KorQuAD2 original json files under data/raw/KorQuAD_2.0/."
        )

    filtered_qas = read_jsonl(FILTERED_QA_PATH)
    legacy = load_legacy_metadata()
    original_id_metadata = load_original_id_metadata(filtered_qas)
    original_docs, original_indexes = build_original_docs(original_files, filtered_qas, legacy, original_id_metadata)
    if not original_docs:
        raise ValueError("KorQuAD2 original files were found, but no matching filtered QA contexts were extracted.")

    counters = Counter()
    failed_examples = []
    chosen_by_qid: dict[str, dict[str, Any]] = {}
    real_needed: dict[tuple[int, int], str] = {}
    fallback_needed: dict[str, dict[str, Any]] = {}

    for qa in filtered_qas:
        counters["total_filtered_qa"] += 1
        qid = str(qa.get("qid"))
        meta = qa_metadata(qa, legacy, original_id_metadata)
        candidates = candidate_docs(qa, meta, original_indexes)

        best = None
        for doc_idx in candidates:
            match = best_answer_chunk(original_docs[doc_idx], qa.get("answer", ""))
            if match is None:
                continue
            chunk_idx, chunk = match
            if best is None or (len(chunk), doc_idx, chunk_idx) < (len(best["text"]), best["doc_idx"], best["chunk_idx"]):
                best = {"doc_idx": doc_idx, "chunk_idx": chunk_idx, "text": chunk}

        if best is not None:
            counters["answer_found_in_real_chunk"] += 1
            chosen_by_qid[qid] = best
            real_needed[(best["doc_idx"], best["chunk_idx"])] = ""
            continue

        counters["fallback_synthetic_count"] += 1
        fallback = {
            "text": fallback_text(qa, meta),
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "url": meta["url"],
            "legacy_chunk_id": meta["chunk_id"],
        }
        fallback_needed[qid] = fallback
        chosen_by_qid[qid] = {"fallback": True, **fallback}
        if len(failed_examples) < 20:
            failed_examples.append(
                {
                    "qid": qid,
                    "question": qa.get("question", ""),
                    "answer": clean_answer(qa.get("answer", "")),
                    "title": meta["title"],
                    "url": meta["url"],
                    "doc_id": meta["doc_id"],
                    "reason": "answer_not_found_in_candidate_context" if candidates else "candidate_doc_not_found",
                }
            )

    chunk_id_by_real_key = {}
    raw_chunks = []
    chunk_counter = 1
    for doc_idx, chunk_idx in sorted(real_needed):
        doc = original_docs[doc_idx]
        chunk = doc["chunks"][chunk_idx]
        chunk_id = f"chunk_{chunk_counter:09d}"
        chunk_counter += 1
        chunk_id_by_real_key[(doc_idx, chunk_idx)] = chunk_id
        raw_chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"] or f"korquad2_doc_{doc_idx:07d}",
                "text": chunk,
                "title": doc["title"],
                "url": doc["url"],
                "synthetic_from_qa": False,
                "source": "korquad2_original",
                "length": len(chunk),
            }
        )

    fallback_id_by_qid = {}
    fallback_seen = {}
    used_chunk_ids = {row["chunk_id"] for row in raw_chunks}
    for qid, fallback in fallback_needed.items():
        key = (fallback["doc_id"], fallback["title"], fallback["url"], fallback["text"])
        chunk_id = fallback_seen.get(key)
        if chunk_id is None:
            chunk_id = fallback["legacy_chunk_id"] or f"chunk_{chunk_counter:09d}"
            while chunk_id in used_chunk_ids:
                chunk_id = f"chunk_{chunk_counter:09d}"
                chunk_counter += 1
            fallback_seen[key] = chunk_id
            used_chunk_ids.add(chunk_id)
            raw_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": fallback["doc_id"],
                    "text": fallback["text"],
                    "title": fallback["title"],
                    "url": fallback["url"],
                    "synthetic_from_qa": True,
                    "source": "legacy_synthetic_fallback",
                    "length": len(fallback["text"]),
                }
            )
        fallback_id_by_qid[qid] = chunk_id

    raw_by_id = {row["chunk_id"]: row for row in raw_chunks}
    corpus_rows = [
        {
            "pid": f"korquad2_{row['chunk_id']}",
            "doc_id": f"korquad2_{row['chunk_id']}",
            "dataset": "korquad2",
            "text": row["text"],
            "title": row["title"],
            "source_doc_id": row["doc_id"],
            "synthetic_from_qa": row["synthetic_from_qa"],
        }
        for row in raw_chunks
    ]

    updated_qas = []
    for qa in filtered_qas:
        qid = str(qa.get("qid"))
        chosen = chosen_by_qid[qid]
        updated = dict(qa)
        if chosen.get("fallback"):
            chunk_id = fallback_id_by_qid[qid]
            synthetic = True
        else:
            chunk_id = chunk_id_by_real_key[(chosen["doc_idx"], chosen["chunk_idx"])]
            synthetic = False
        chunk = raw_by_id[chunk_id]
        updated["gold_pid"] = f"korquad2_{chunk_id}"
        updated["gold_doc_id"] = f"korquad2_{chunk_id}"
        updated["gold_passage"] = chunk["text"]
        updated["synthetic_from_qa"] = synthetic
        updated_qas.append(updated)

    lengths = [row["length"] for row in raw_chunks]
    total = counters["total_filtered_qa"]
    synthetic_ratio = counters["fallback_synthetic_count"] / total if total else 0.0
    summary = {
        "source_files": [str(path.relative_to(ROOT)) for path in original_files],
        "total_filtered_qa": total,
        "answer_found_in_real_chunk": counters["answer_found_in_real_chunk"],
        "fallback_synthetic_count": counters["fallback_synthetic_count"],
        "synthetic_ratio": round(synthetic_ratio, 6),
        "total_chunks": len(raw_chunks),
        "avg_chunk_length": round(mean(lengths), 2) if lengths else 0,
        "failed_examples": failed_examples,
    }

    write_jsonl(raw_chunks, RAW_CHUNKS_PATH)
    write_jsonl(corpus_rows, FILTERED_CORPUS_PATH)
    write_jsonl(updated_qas, FILTERED_QA_PATH)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("KorQuAD2 real chunk rebuild")
    print(f"total_filtered_qa: {summary['total_filtered_qa']}")
    print(f"answer_found_in_real_chunk: {summary['answer_found_in_real_chunk']}")
    print(f"fallback_synthetic_count: {summary['fallback_synthetic_count']}")
    print(f"synthetic_ratio: {summary['synthetic_ratio']}")
    print(f"total_chunks: {summary['total_chunks']}")
    print(f"avg_chunk_length: {summary['avg_chunk_length']}")
    if summary["synthetic_ratio"] <= 0.05:
        print("synthetic_ratio is low")
    else:
        print("synthetic_ratio is still high")


if __name__ == "__main__":
    main()
