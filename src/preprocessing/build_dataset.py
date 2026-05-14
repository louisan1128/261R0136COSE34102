import argparse
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_json, write_jsonl


def load_qa_dataset(raw_path: str | Path) -> list[dict[str, Any]]:
    """Load KorQuAD or KLUE-MRC style JSON into a list of article records."""

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    raw_data = read_json(raw_path)
    if isinstance(raw_data, dict) and isinstance(raw_data.get("data"), list):
        return raw_data["data"]
    if isinstance(raw_data, list):
        return raw_data
    raise ValueError("Unsupported raw dataset format. Expected KorQuAD/KLUE-MRC style JSON.")


def build_dataset(raw_path: str, corpus_out: str, qa_out: str, limit: int | None = None) -> None:
    """Build corpus.jsonl and qa_pairs.jsonl.

    Each QA record contains qid, question, answer, gold_doc_id, and gold_passage.
    If the raw data path is missing, a small Korean dummy dataset is written so
    the full pipeline can run as a smoke test.
    """

    raw_path_obj = Path(raw_path)
    corpus_path = Path(corpus_out)
    qa_path = Path(qa_out)
    ensure_dir(corpus_path.parent)
    ensure_dir(qa_path.parent)

    if not raw_path_obj.exists():
        print(f"Raw path not found: {raw_path_obj}. Creating dummy sample dataset.")
        _build_dummy_dataset(corpus_path, qa_path)
        return

    articles = load_qa_dataset(raw_path_obj)
    corpus: list[dict[str, str]] = []
    qa_records: list[dict[str, str]] = []
    context_to_doc_id: dict[str, str] = {}
    doc_counter = 1
    q_counter = 1

    for article in articles:
        paragraphs = _iter_paragraphs(article)
        for paragraph in paragraphs:
            context = _get_context(paragraph)
            if not context:
                continue

            if context not in context_to_doc_id:
                doc_id = f"doc_{doc_counter:06d}"
                context_to_doc_id[context] = doc_id
                corpus.append({"doc_id": doc_id, "text": context})
                doc_counter += 1
            else:
                doc_id = context_to_doc_id[context]

            for qa in _iter_qas(paragraph):
                question = _get_question(qa)
                if not question:
                    continue
                answer = _extract_answer_text(qa.get("answers") or qa.get("answer") or [])
                qid = str(qa.get("id") or qa.get("guid") or f"q_{q_counter:06d}")
                qa_records.append(
                    {
                        "qid": qid,
                        "question": question,
                        "answer": answer,
                        "gold_doc_id": doc_id,
                        "gold_passage": answer or context,
                    }
                )
                q_counter += 1
                if limit and len(qa_records) >= limit:
                    break
            if limit and len(qa_records) >= limit:
                break
        if limit and len(qa_records) >= limit:
            break

    if not corpus or not qa_records:
        raise ValueError("No usable corpus documents or QA pairs were created from the raw dataset.")

    write_jsonl(corpus, corpus_path)
    write_jsonl(qa_records, qa_path)
    print(f"Saved {len(corpus)} corpus docs to {corpus_path}")
    print(f"Saved {len(qa_records)} QA pairs to {qa_path}")


def _iter_paragraphs(article: dict[str, Any]) -> list[dict[str, Any]]:
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


def _get_context(paragraph: dict[str, Any]) -> str:
    context = paragraph.get("context") or paragraph.get("text") or paragraph.get("passage")
    return context.strip() if isinstance(context, str) else ""


def _get_question(qa: dict[str, Any]) -> str:
    question = qa.get("question") or qa.get("query")
    return question.strip() if isinstance(question, str) else ""


def _extract_answer_text(answers: Any) -> str:
    if isinstance(answers, str):
        return answers.strip()
    if isinstance(answers, dict):
        text = answers.get("text") or answers.get("answer") or answers.get("value")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(answers, list) and answers:
        return _extract_answer_text(answers[0])
    return ""


def _build_dummy_dataset(corpus_out: Path, qa_out: Path) -> None:
    # Small fallback data for smoke tests when the real KorQuAD/KLUE file is absent.
    corpus = [
        {"doc_id": "doc_000001", "text": "세종대왕은 조선의 네 번째 왕이며 훈민정음을 창제했다."},
        {"doc_id": "doc_000002", "text": "훈민정음은 백성을 가르치는 바른 소리라는 뜻을 가진 문자 체계이다."},
        {"doc_id": "doc_000003", "text": "경복궁은 조선 시대의 대표 궁궐로 서울에 위치한다."},
        {"doc_id": "doc_000004", "text": "이순신은 임진왜란에서 조선 수군을 이끈 장군이다."},
        {"doc_id": "doc_000005", "text": "김치는 배추와 양념을 발효시켜 만드는 한국의 전통 음식이다."},
        {"doc_id": "doc_000006", "text": "한라산은 제주도 중앙에 있는 대한민국의 높은 산이다."},
    ]
    qa_pairs = [
        {
            "qid": "dummy_001",
            "question": "훈민정음을 만든 왕은 누구인가?",
            "answer": "세종대왕",
            "gold_doc_id": "doc_000001",
            "gold_passage": "세종대왕은 조선의 네 번째 왕이며 훈민정음을 창제했다.",
        },
        {
            "qid": "dummy_002",
            "question": "조선 수군을 이끈 장군은 누구인가?",
            "answer": "이순신",
            "gold_doc_id": "doc_000004",
            "gold_passage": "이순신은 임진왜란에서 조선 수군을 이끈 장군이다.",
        },
        {
            "qid": "dummy_003",
            "question": "제주도 중앙에 있는 산은 무엇인가?",
            "answer": "한라산",
            "gold_doc_id": "doc_000006",
            "gold_passage": "한라산은 제주도 중앙에 있는 대한민국의 높은 산이다.",
        },
    ]
    write_jsonl(corpus, corpus_out)
    write_jsonl(qa_pairs, qa_out)
    print(f"Created dummy sample corpus and QA pairs at {corpus_out} / {qa_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed dataset for KorQR-RL.")
    parser.add_argument("--raw_path", default="data/raw/KorQuAD.json")
    parser.add_argument("--corpus_out", default="data/processed/corpus.jsonl")
    parser.add_argument("--qa_out", default="data/processed/qa_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_dataset(args.raw_path, args.corpus_out, args.qa_out, limit=args.limit)
