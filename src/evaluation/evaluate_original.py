import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import answer_f1, mrr, recall_at_k
from src.rewriting.policy import infer_failure_type
from src.utils.io import ensure_dir, read_jsonl, write_jsonl


def load_qa_pairs(path: str) -> list[dict[str, Any]]:
    return read_jsonl(path)


def evaluate_retriever(retriever, qa_pairs: list[dict[str, Any]], top_k: int = 10) -> dict[str, float]:
    counts = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "answer_f1": 0.0}
    for qa in qa_pairs:
        gold_doc_id = qa["gold_doc_id"]
        retrieved = retriever.retrieve(qa["question"], top_k=top_k)
        counts["recall@1"] += recall_at_k(retrieved, gold_doc_id, 1)
        counts["recall@5"] += recall_at_k(retrieved, gold_doc_id, 5)
        counts["recall@10"] += recall_at_k(retrieved, gold_doc_id, 10)
        counts["mrr"] += mrr(retrieved, gold_doc_id)
        counts["answer_f1"] += answer_f1(retrieved, qa.get("answer", ""), k=top_k)

    total = len(qa_pairs) or 1
    return {
        "recall@1": counts["recall@1"] / total,
        "recall@5": counts["recall@5"] / total,
        "recall@10": counts["recall@10"] / total,
        "mrr": counts["mrr"] / total,
        "answer_f1": counts["answer_f1"] / total,
        "num_questions": total,
    }


def evaluate_named_retrievers(retrievers: dict[str, Any], qa_pairs: list[dict[str, Any]], top_k: int = 10) -> list[dict]:
    results = []
    for retriever_name, retriever in retrievers.items():
        metrics = evaluate_retriever(retriever, qa_pairs, top_k=top_k)
        results.append({"retriever": retriever_name, **metrics})
    return results


def save_hard_cases(retriever, qa_pairs: list[dict[str, Any]], out_path: str, top_k: int = 10) -> list[dict[str, Any]]:
    if isinstance(retriever, dict):
        return save_hard_cases_by_retriever(retriever, qa_pairs, out_path, top_k=top_k)

    ensure_dir(Path(out_path).parent)
    hard_cases = []
    with Path(out_path).open("w", encoding="utf-8") as fout:
        for qa in qa_pairs:
            retrieved = retriever.retrieve(qa["question"], top_k=top_k)
            retrieved_ids = [item["doc_id"] for item in retrieved]
            if qa["gold_doc_id"] not in retrieved_ids:
                failure_type = infer_failure_type(qa["question"])
                hard_case = {
                    "qid": qa["qid"],
                    "question": qa["question"],
                    "answer": qa.get("answer", ""),
                    "gold_doc_id": qa["gold_doc_id"],
                    "gold_passage": qa.get("gold_passage", ""),
                    "original_retrieved_top10": retrieved_ids,
                    "failure_type": failure_type,
                    "failed_retrievers": ["bm25"],
                    "original_retrieved_by_retriever": {"bm25": retrieved_ids},
                }
                fout.write(json.dumps(hard_case, ensure_ascii=False) + "\n")
                hard_cases.append(hard_case)
    return hard_cases


def save_hard_cases_by_retriever(
    retrievers: dict[str, Any],
    qa_pairs: list[dict[str, Any]],
    out_path: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Save the union of original-query failures across retrievers."""

    ensure_dir(Path(out_path).parent)
    hard_cases_by_qid = {}

    for qa in qa_pairs:
        retrieved_by_retriever = {}
        failed_retrievers = []
        for retriever_name, retriever in retrievers.items():
            retrieved = retriever.retrieve(qa["question"], top_k=top_k)
            retrieved_ids = [item["doc_id"] for item in retrieved]
            retrieved_by_retriever[retriever_name] = retrieved_ids
            if qa["gold_doc_id"] not in retrieved_ids:
                failed_retrievers.append(retriever_name)

        if not failed_retrievers:
            continue

        primary = failed_retrievers[0]
        failure_type = infer_failure_type(qa["question"])
        hard_cases_by_qid[qa["qid"]] = {
            "qid": qa["qid"],
            "question": qa["question"],
            "answer": qa.get("answer", ""),
            "gold_doc_id": qa["gold_doc_id"],
            "gold_passage": qa.get("gold_passage", ""),
            "original_retrieved_top10": retrieved_by_retriever[primary],
            "original_retrieved_by_retriever": retrieved_by_retriever,
            "failed_retrievers": failed_retrievers,
            "failure_type": failure_type,
        }

    hard_cases = list(hard_cases_by_qid.values())
    write_jsonl(hard_cases, out_path)
    return hard_cases
