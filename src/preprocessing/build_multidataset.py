from collections import Counter
import html
from pathlib import Path
import random
import re
from typing import Any

from src.utils.io import ensure_dir, read_json, read_jsonl, write_jsonl


KORQUAD2_MISSING_CHUNKS_MESSAGE = (
    "KorQuAD 2.0 QA file exists but chunk corpus is missing. Please place chunk corpus at "
    "data/raw/korquad2_chunks.jsonl with fields chunk_id, doc_id, text."
)


def build_multidataset(data_config: dict[str, Any]) -> None:
    corpus_path = Path(data_config["corpus_path"])
    qa_path = Path(data_config["qa_path"])
    ensure_dir(corpus_path.parent)
    ensure_dir(qa_path.parent)

    datasets = data_config.get("datasets") or ["korquad1"]
    limit = data_config.get("limit")

    corpus: list[dict[str, str]] = []
    qa_pairs: list[dict[str, str]] = []
    context_to_pid: dict[tuple[str, str], str] = {}
    summary = {
        "corpus": Counter(),
        "qa": Counter(),
    }

    for dataset_name in datasets:
        if dataset_name == "korquad1":
            _add_korquad1(data_config, corpus, qa_pairs, context_to_pid, summary, limit)
        elif dataset_name == "klue_mrc":
            _add_klue_mrc(data_config, corpus, qa_pairs, context_to_pid, summary, limit)
        elif dataset_name == "korquad2":
            _add_korquad2(data_config, corpus, qa_pairs, context_to_pid, summary, limit)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not corpus or not qa_pairs:
        raise ValueError("No usable corpus documents or QA pairs were created.")

    missing_gold = validate_gold_pids(corpus, qa_pairs)
    _print_summary(summary, len(corpus), len(qa_pairs), missing_gold)
    if missing_gold:
        raise ValueError(f"Found {missing_gold} QA pairs whose gold_pid is missing from corpus.")

    write_jsonl(corpus, corpus_path)
    write_jsonl(qa_pairs, qa_path)
    if data_config.get("write_dataset_files", True):
        _write_dataset_files(datasets, corpus, qa_pairs, corpus_path.parent)
    print(f"Saved {len(corpus)} corpus docs to {corpus_path}")
    print(f"Saved {len(qa_pairs)} QA pairs to {qa_path}")


def validate_gold_pids(corpus: list[dict[str, str]], qa_pairs: list[dict[str, str]]) -> int:
    corpus_pids = {record["pid"] for record in corpus}
    return sum(1 for qa in qa_pairs if qa.get("gold_pid") not in corpus_pids)


def _add_korquad1(
    data_config: dict[str, Any],
    corpus: list[dict[str, str]],
    qa_pairs: list[dict[str, str]],
    context_to_pid: dict[tuple[str, str], str],
    summary: dict[str, Counter],
    limit: int | None,
) -> None:
    dataset = "korquad1"
    rng = random.Random(int(data_config.get("korquad1_train_sample_seed", 7)))
    train_sample_size = data_config.get("korquad1_train_sample_size")

    for raw_path in _korquad1_raw_paths(data_config):
        if not raw_path.exists():
            raise FileNotFoundError(f"KorQuAD 1.0 raw file not found: {raw_path}")

        split = _infer_split(raw_path, data_config.get("korquad1_split") if "train" not in raw_path.name.lower() else None)
        prefix = f"{dataset}_{split}"
        corpus_counter = 0
        qa_counter = 0

        qa_items = _load_korquad1_qa_items(raw_path, split)
        if split == "train" and train_sample_size:
            sample_size = min(int(train_sample_size), len(qa_items))
            qa_items = sorted(rng.sample(qa_items, sample_size), key=lambda item: item["source_index"])

        for item in qa_items:
            pid = _get_or_add_corpus(
                corpus=corpus,
                context_to_pid=context_to_pid,
                dataset=dataset,
                text=item["context"],
                pid=f"{prefix}_{corpus_counter:06d}",
                title=item["title"],
                source_doc_id=f"{prefix}_{corpus_counter:06d}",
            )
            if pid == f"{prefix}_{corpus_counter:06d}":
                corpus_counter += 1
                summary["corpus"][dataset] += 1

            qa_pairs.append(
                _make_qa_record(
                    qid=f"{prefix}_q_{qa_counter:06d}",
                    dataset=dataset,
                    question=item["question"],
                    answer=item["answer"],
                    gold_pid=pid,
                    gold_passage=item["context"],
                    extra_fields={
                        "source_split": split,
                        "source_qid": item["source_qid"],
                    },
                )
            )
            qa_counter += 1
            summary["qa"][dataset] += 1
            if limit and summary["qa"][dataset] >= limit:
                return


def _korquad1_raw_paths(data_config: dict[str, Any]) -> list[Path]:
    paths = [Path(data_config.get("korquad1_path") or data_config.get("raw_path", ""))]
    train_path = data_config.get("korquad1_train_path")
    if train_path:
        paths.append(Path(train_path))
    return paths


def _load_korquad1_qa_items(raw_path: Path, split: str) -> list[dict[str, str | int]]:
    raw_data = read_json(raw_path)
    articles = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
    if not isinstance(articles, list):
        raise ValueError("Unsupported KorQuAD 1.0 format. Expected a JSON object with a data list.")

    items = []
    source_index = 0
    for article in articles:
        title = str(article.get("title", "")).strip() if isinstance(article, dict) else ""
        for paragraph in _iter_paragraphs(article):
            context = _get_context(paragraph)
            if not context:
                continue
            for qa in _iter_qas(paragraph):
                question = _get_question(qa)
                if not question:
                    continue
                items.append(
                    {
                        "split": split,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answer": _extract_answer_text(qa.get("answers") or qa.get("answer") or []),
                        "source_qid": str(qa.get("id") or qa.get("guid") or ""),
                        "source_index": source_index,
                    }
                )
                source_index += 1
    return items


def _add_klue_mrc(
    data_config: dict[str, Any],
    corpus: list[dict[str, str]],
    qa_pairs: list[dict[str, str]],
    context_to_pid: dict[tuple[str, str], str],
    summary: dict[str, Counter],
    limit: int | None,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("KLUE-MRC requires the `datasets` package. Install requirements.txt first.") from exc

    requested_split = data_config.get("klue_mrc_split", "validation")
    split_candidates = _unique([requested_split, "validation", "dev", "train"])
    last_error = None
    dataset_rows = None
    split = requested_split
    for candidate in split_candidates:
        try:
            dataset_rows = load_dataset("klue", "mrc", split=candidate)
            split = candidate
            break
        except Exception as exc:  # HuggingFace raises several exception types for missing splits/cache.
            last_error = exc

    if dataset_rows is None:
        raise RuntimeError(f"Could not load KLUE-MRC from HuggingFace datasets: {last_error}") from last_error

    dataset = "klue_mrc"
    prefix = f"{dataset}_{split}"
    corpus_counter = 0
    qa_counter = 0

    for row in dataset_rows:
        context = _get_context(row)
        question = _get_question(row)
        if not context or not question:
            continue

        pid = _get_or_add_corpus(
            corpus=corpus,
            context_to_pid=context_to_pid,
            dataset=dataset,
            text=context,
            pid=f"{prefix}_{corpus_counter:06d}",
            title=str(row.get("title", "")).strip(),
            source_doc_id=str(row.get("guid") or row.get("id") or f"{prefix}_{corpus_counter:06d}"),
        )
        if pid == f"{prefix}_{corpus_counter:06d}":
            corpus_counter += 1
            summary["corpus"][dataset] += 1

        qa_pairs.append(
            _make_qa_record(
                qid=f"{prefix}_q_{qa_counter:06d}",
                dataset=dataset,
                question=question,
                answer=_extract_answer_text(row.get("answers") or row.get("answer") or []),
                gold_pid=pid,
                gold_passage=context,
            )
        )
        qa_counter += 1
        summary["qa"][dataset] += 1
        if limit and summary["qa"][dataset] >= limit:
            return


def _add_korquad2(
    data_config: dict[str, Any],
    corpus: list[dict[str, str]],
    qa_pairs: list[dict[str, str]],
    context_to_pid: dict[tuple[str, str], str],
    summary: dict[str, Counter],
    limit: int | None,
) -> None:
    qa_path = _resolve_korquad2_qa_path(data_config)
    chunk_path = Path(data_config.get("korquad2_chunk_path", "data/raw/korquad2_chunks.jsonl"))
    if not qa_path.exists():
        print(f"Skipping KorQuAD 2.0 because QA file was not found: {qa_path}")
        return
    if not chunk_path.exists():
        if data_config.get("korquad2_build_chunks_from_qa", True):
            _build_korquad2_chunks_from_qa(qa_path, chunk_path)
        else:
            raise FileNotFoundError(KORQUAD2_MISSING_CHUNKS_MESSAGE)

    dataset = "korquad2"
    chunk_by_id: dict[str, dict[str, str]] = {}
    for chunk in read_jsonl(chunk_path):
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        text = _get_context(chunk)
        if not chunk_id or not text:
            continue
        chunk_by_id[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "title": str(chunk.get("title", "")).strip(),
            "doc_id": str(chunk.get("doc_id", "")).strip(),
        }

    if not chunk_by_id:
        raise ValueError(f"No usable chunks found in {chunk_path}. Expected fields chunk_id, doc_id, text.")

    reference_type_counts = _question_type_counts(qa_pairs)
    candidates = []
    for qa in read_jsonl(qa_path):
        question = _get_question(qa)
        gold_chunk_id = str(qa.get("gold_chunk_id", "")).strip()
        if not question or gold_chunk_id not in chunk_by_id:
            continue
        candidates.append(
            {
                "question": question,
                "answer": str(qa.get("answer", "")).strip(),
                "gold_chunk_id": gold_chunk_id,
                "question_type": _classify_question_type(question),
            }
        )

    selected_candidates = _select_korquad2_candidates(candidates, reference_type_counts, data_config)
    selected_chunk_ids = {candidate["gold_chunk_id"] for candidate in selected_candidates}
    chunk_id_to_pid: dict[str, str] = {}
    for chunk_id in sorted(selected_chunk_ids):
        chunk = chunk_by_id[chunk_id]
        pid = _get_or_add_corpus(
            corpus=corpus,
            context_to_pid=context_to_pid,
            dataset=dataset,
            text=chunk["text"],
            pid=f"{dataset}_{_safe_id(chunk_id)}",
            title=chunk["title"],
            source_doc_id=chunk["doc_id"],
        )
        chunk_id_to_pid[chunk_id] = pid
        if pid == f"{dataset}_{_safe_id(chunk_id)}":
            summary["corpus"][dataset] += 1

    selected_type_counts = Counter(candidate["question_type"] for candidate in selected_candidates)
    if selected_type_counts:
        print(f"KorQuAD 2.0 selected question types: {dict(sorted(selected_type_counts.items()))}")

    qa_counter = 0
    for candidate in selected_candidates:
        qa_pairs.append(
            _make_qa_record(
                qid=f"{dataset}_q_{qa_counter:06d}",
                dataset=dataset,
                question=candidate["question"],
                answer=candidate["answer"],
                gold_pid=chunk_id_to_pid[candidate["gold_chunk_id"]],
                gold_passage="",
            )
        )
        qa_counter += 1
        summary["qa"][dataset] += 1
        if limit and summary["qa"][dataset] >= limit:
            return


def _get_or_add_corpus(
    corpus: list[dict[str, str]],
    context_to_pid: dict[tuple[str, str], str],
    dataset: str,
    text: str,
    pid: str,
    title: str = "",
    source_doc_id: str = "",
) -> str:
    key = (dataset, _normalize_text(text))
    if key in context_to_pid:
        return context_to_pid[key]

    context_to_pid[key] = pid
    record = {
        "pid": pid,
        "doc_id": pid,
        "dataset": dataset,
        "text": text,
    }
    if title:
        record["title"] = title
    if source_doc_id:
        record["source_doc_id"] = source_doc_id
    corpus.append(record)
    return pid


def _make_qa_record(
    qid: str,
    dataset: str,
    question: str,
    answer: str,
    gold_pid: str,
    gold_passage: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, str]:
    record = {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "question_type": _classify_question_type(question),
        "answer": answer,
        "gold_pid": gold_pid,
        "gold_doc_id": gold_pid,
    }
    if gold_passage:
        record["gold_passage"] = gold_passage
    if extra_fields:
        record.update(extra_fields)
    return record


def _iter_paragraphs(article: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(article, dict):
        return []
    paragraphs = article.get("paragraphs") or article.get("paragraph") or []
    if isinstance(paragraphs, dict):
        return [paragraphs]
    if isinstance(paragraphs, list):
        return [paragraph for paragraph in paragraphs if isinstance(paragraph, dict)]
    return []


def _iter_qas(paragraph: dict[str, Any]) -> list[dict[str, Any]]:
    qas = paragraph.get("qas") or paragraph.get("questions") or []
    if isinstance(qas, dict):
        return [qas]
    if isinstance(qas, list):
        return [qa for qa in qas if isinstance(qa, dict)]
    return []


def _get_context(record: dict[str, Any]) -> str:
    context = record.get("context") or record.get("text") or record.get("passage")
    return context.strip() if isinstance(context, str) else ""


def _get_question(record: dict[str, Any]) -> str:
    question = record.get("question") or record.get("query")
    return question.strip() if isinstance(question, str) else ""


def _extract_answer_text(answers: Any) -> str:
    if isinstance(answers, str):
        return answers.strip()
    if isinstance(answers, dict):
        text = answers.get("text") or answers.get("answer") or answers.get("value")
        if isinstance(text, list):
            return _extract_answer_text(text)
        return text.strip() if isinstance(text, str) else ""
    if isinstance(answers, list) and answers:
        return _extract_answer_text(answers[0])
    return ""


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _classify_question_type(question: str) -> str:
    question = _normalize_text(question)
    if any(token in question for token in ["왜", "이유", "원인", "어째서"]):
        return "why"
    if any(token in question for token in ["어떻게", "방법", "과정", "방식", "절차"]):
        return "how"
    if any(token in question for token in ["비교", "차이", "다른", "공통", "유사"]):
        return "comparison"
    if any(token in question for token in ["몇", "얼마", "수", "횟수", "비율", "퍼센트"]):
        return "numeric"
    if any(token in question for token in ["언제", "몇 년", "몇월", "몇 월", "날짜", "시기"]):
        return "when"
    if any(token in question for token in ["어디", "장소", "지역", "국가", "도시"]):
        return "where"
    if any(token in question for token in ["누구", "인물", "사람"]):
        return "who"
    if any(token in question for token in ["무엇", "뭐", "어떤", "무슨"]):
        return "what"
    if any(token in question for token in ["정의", "의미", "뜻"]):
        return "definition"
    if any(token in question for token in ["나열", "목록", "종류", "예시"]):
        return "list"
    if question.endswith(("인가?", "인가", "입니까?", "입니까", "니?", "나요?", "나요")):
        return "yes_no"
    return "other"


def _question_type_counts(qa_pairs: list[dict[str, str]]) -> Counter:
    return Counter(
        qa.get("question_type") or _classify_question_type(qa.get("question", ""))
        for qa in qa_pairs
        if qa.get("dataset") != "korquad2"
    )


def _select_korquad2_candidates(
    candidates: list[dict[str, str]],
    reference_type_counts: Counter,
    data_config: dict[str, Any],
) -> list[dict[str, str]]:
    if not data_config.get("korquad2_focus_missing_question_types", True):
        return candidates

    candidate_types = {candidate["question_type"] for candidate in candidates}
    missing_types = sorted(candidate_types - set(reference_type_counts))
    if missing_types:
        selected_types = set(missing_types)
    else:
        top_k = int(data_config.get("korquad2_focus_top_k_question_types", 6))
        selected_types = {
            question_type
            for question_type, _ in sorted(
                ((question_type, reference_type_counts.get(question_type, 0)) for question_type in candidate_types),
                key=lambda item: (item[1], item[0]),
            )[:top_k]
        }

    selected = [candidate for candidate in candidates if candidate["question_type"] in selected_types]
    return selected or candidates


def _resolve_korquad2_qa_path(data_config: dict[str, Any]) -> Path:
    candidates = [
        data_config.get("korquad2_qa_path"),
        data_config.get("korquad2_qa_fallback_path"),
        "data/processed/qa_pairs_2.jsonl",
        "data/raw/qa_pairs_2(1).jsonl",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(data_config.get("korquad2_qa_path", "data/raw/qa_pairs_2(1).jsonl"))


def _build_korquad2_chunks_from_qa(qa_path: Path, chunk_path: Path) -> None:
    chunk_records: dict[str, dict[str, Any]] = {}
    for qa in read_jsonl(qa_path):
        chunk_id = str(qa.get("gold_chunk_id", "")).strip()
        if not chunk_id:
            continue
        record = chunk_records.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "doc_id": str(qa.get("doc_id", "")).strip(),
                "title": str(qa.get("title", "")).strip(),
                "url": str(qa.get("url", "")).strip(),
                "answers": [],
            },
        )
        answer = _clean_korquad2_answer(str(qa.get("answer", "")).strip())
        if answer and answer not in record["answers"]:
            record["answers"].append(answer)

    chunks = []
    for chunk_id, record in sorted(chunk_records.items()):
        text_parts = []
        if record["title"]:
            text_parts.append(record["title"].replace("_", " "))
        text_parts.extend(record["answers"])
        text = _normalize_text(" ".join(text_parts))
        if not text:
            continue
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": record["doc_id"],
                "text": text,
                "title": record["title"],
                "url": record["url"],
                "synthetic_from_qa": True,
            }
        )

    if not chunks:
        raise ValueError(f"Could not synthesize KorQuAD 2.0 chunks from {qa_path}.")
    write_jsonl(chunks, chunk_path)
    print(
        f"Created {len(chunks)} synthetic KorQuAD 2.0 chunks at {chunk_path} "
        f"from {qa_path}. These chunks use QA answers because original context text was not available."
    )


def _clean_korquad2_answer(answer: str) -> str:
    answer = re.sub(r"<[^>]+>", " ", answer)
    return _normalize_text(html.unescape(answer))


def _write_dataset_files(
    datasets: list[str],
    corpus: list[dict[str, str]],
    qa_pairs: list[dict[str, str]],
    output_dir: Path,
) -> None:
    dataset_to_slug = {
        "korquad1": "korquad1",
        "klue_mrc": "klue_mrc",
        "korquad2": "korquad2_filtered",
    }
    for dataset in datasets:
        slug = dataset_to_slug.get(dataset, dataset)
        dataset_corpus = [record for record in corpus if record.get("dataset") == dataset]
        dataset_qa = [record for record in qa_pairs if record.get("dataset") == dataset]
        write_jsonl(dataset_corpus, output_dir / f"{slug}_corpus.jsonl")
        write_jsonl(dataset_qa, output_dir / f"{slug}_qa_pairs.jsonl")
        if dataset == "korquad2":
            write_jsonl(dataset_qa, output_dir / "korquad2_filtered_qa.jsonl")
        print(
            f"Saved {dataset} split files to {output_dir / f'{slug}_corpus.jsonl'} "
            f"and {output_dir / f'{slug}_qa_pairs.jsonl'}"
        )


def _infer_split(path: Path, configured_split: str | None = None) -> str:
    if configured_split:
        return configured_split
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "valid" in name:
        return "validation"
    return "data"


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _print_summary(summary: dict[str, Counter], total_corpus: int, total_qa: int, missing_gold: int) -> None:
    print("Dataset build summary")
    for dataset in sorted(set(summary["corpus"]) | set(summary["qa"])):
        print(f"- {dataset}: corpus={summary['corpus'][dataset]}, qa={summary['qa'][dataset]}")
    print(f"- total corpus: {total_corpus}")
    print(f"- total qa: {total_qa}")
    print(f"- missing gold: {missing_gold}")
