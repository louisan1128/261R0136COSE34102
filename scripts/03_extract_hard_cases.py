import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.rewriting.policy import infer_failure_type
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_jsonl


INPUT_DIR = Path("data/outputs/original_retrieval")
OUTPUT_DIR = Path("data/outputs/hard_cases")
RANDOM_SEED = 7

DATASET_SPECS = {
    "korquad1": {
        "qa_path": "data/processed/korquad1_qa_pairs.jsonl",
        "sample_size": 50,
    },
    "klue_mrc": {
        "qa_path": "data/processed/klue_mrc_qa_pairs.jsonl",
        "sample_size": 100,
    },
    "korquad2": {
        "qa_path": "data/processed/korquad2_filtered_qa_pairs.jsonl",
        "sample_size": 150,
    },
}
RETRIEVERS = ["bm25", "dense", "hybrid"]


def main() -> None:
    config = read_yaml(root / "configs" / "default.yaml")
    ensure_dir(OUTPUT_DIR)

    qa_by_dataset = {
        dataset: _load_qa_by_qid(spec["qa_path"])
        for dataset, spec in DATASET_SPECS.items()
    }
    results_by_dataset = _load_all_results()

    all_summary: dict[str, dict[str, dict[str, Any]]] = {}
    hybrid_hard_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for dataset in DATASET_SPECS:
        all_summary[dataset] = {}
        qid_retriever_results = _build_qid_retriever_results(results_by_dataset[dataset])

        for retriever in RETRIEVERS:
            records = results_by_dataset[dataset][retriever]
            hard_records = [
                _make_hard_case_record(record, qa_by_dataset[dataset], qid_retriever_results)
                for record in records
                if _is_hard_case(record)
            ]
            hard_path = OUTPUT_DIR / f"{dataset}_{retriever}_hard.jsonl"
            write_jsonl(hard_records, hard_path)
            all_summary[dataset][retriever] = _summarize_results(records, hard_records)

            if retriever == "hybrid":
                hybrid_hard_by_dataset[dataset] = hard_records

            print(
                f"{dataset}/{retriever}: hard={len(hard_records)} "
                f"rate={all_summary[dataset][retriever]['hard_rate']:.4f}"
            )

    summary_path = OUTPUT_DIR / "all_retrievers_hard_case_summary.json"
    _write_json(all_summary, summary_path)

    sampled_by_dataset, hard_subset = _sample_hybrid_hard_cases(hybrid_hard_by_dataset)
    hard_case_summary = _build_sample_summary(hybrid_hard_by_dataset, sampled_by_dataset)

    for dataset, sampled in sampled_by_dataset.items():
        sample_size = DATASET_SPECS[dataset]["sample_size"]
        write_jsonl(sampled, OUTPUT_DIR / f"{dataset}_sample{sample_size}.jsonl")

    write_jsonl(hard_subset, OUTPUT_DIR / "hard_subset_300.jsonl")
    _write_json(hard_case_summary, OUTPUT_DIR / "hard_case_summary.json")

    config_hard_path = Path(config["data"].get("hard_cases_path", ""))
    if config_hard_path == OUTPUT_DIR / "hard_subset_300.jsonl":
        write_jsonl(hard_subset, config_hard_path)

    print(f"Saved all retriever summary to {summary_path}")
    print(f"Saved final hard subset with {len(hard_subset)} records to {OUTPUT_DIR / 'hard_subset_300.jsonl'}")


def _load_all_results() -> dict[str, dict[str, list[dict[str, Any]]]]:
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in DATASET_SPECS:
        results[dataset] = {}
        for retriever in RETRIEVERS:
            path = INPUT_DIR / f"{dataset}_{retriever}_results.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Original retrieval result file not found: {path}")
            results[dataset][retriever] = read_jsonl(path)
    return results


def _load_qa_by_qid(qa_path: str) -> dict[str, dict[str, Any]]:
    qa_records = read_jsonl(qa_path)
    if not qa_records:
        raise ValueError(f"QA file is empty: {qa_path}")
    return {str(record["qid"]): record for record in qa_records}


def _build_qid_retriever_results(
    dataset_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for retriever, records in dataset_results.items():
        for record in records:
            by_qid.setdefault(record["qid"], {})[retriever] = {
                "gold_rank": record.get("gold_rank"),
                "hit_at_10": bool(record.get("hit_at_10")),
                "recall_at_10": float(record.get("recall_at_10", 0.0)),
                "mrr": float(record.get("mrr", 0.0)),
                "top10_doc_ids": record.get("top10_doc_ids", []),
            }
    return by_qid


def _is_hard_case(record: dict[str, Any]) -> bool:
    return record.get("hit_at_10") is False or float(record.get("recall_at_10", 0.0)) == 0.0


def _make_hard_case_record(
    record: dict[str, Any],
    qa_by_qid: dict[str, dict[str, Any]],
    qid_retriever_results: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    qid = record["qid"]
    qa = qa_by_qid.get(qid, {})
    retriever_results = qid_retriever_results.get(qid, {})
    failed_retrievers = [
        retriever
        for retriever in RETRIEVERS
        if retriever in retriever_results and not retriever_results[retriever]["hit_at_10"]
    ]
    return {
        "qid": qid,
        "dataset": record.get("dataset", qa.get("dataset", "")),
        "question": record.get("question", qa.get("question", "")),
        "question_type": record.get("question_type", qa.get("question_type", "")),
        "answer": qa.get("answer", ""),
        "gold_doc_id": record.get("gold_doc_id", qa.get("gold_doc_id", "")),
        "gold_passage": qa.get("gold_passage", ""),
        "retriever": record["retriever"],
        "failure_type": infer_failure_type(record.get("question", "")),
        "gold_rank": record.get("gold_rank"),
        "hit_at_10": bool(record.get("hit_at_10")),
        "recall_at_10": float(record.get("recall_at_10", 0.0)),
        "mrr": float(record.get("mrr", 0.0)),
        "top10_doc_ids": record.get("top10_doc_ids", []),
        "failed_retrievers": failed_retrievers,
        "retriever_results": retriever_results,
    }


def _summarize_results(records: list[dict[str, Any]], hard_records: list[dict[str, Any]]) -> dict[str, Any]:
    num_queries = len(records)
    hit_ranks = [int(record["gold_rank"]) for record in records if record.get("gold_rank") is not None]
    return {
        "num_queries": num_queries,
        "hard_count": len(hard_records),
        "hard_rate": len(hard_records) / (num_queries or 1),
        "recall_at_10": sum(float(record.get("recall_at_10", 0.0)) for record in records) / (num_queries or 1),
        "mrr": sum(float(record.get("mrr", 0.0)) for record in records) / (num_queries or 1),
        "avg_gold_rank": sum(hit_ranks) / len(hit_ranks) if hit_ranks else None,
    }


def _sample_hybrid_hard_cases(
    hybrid_hard_by_dataset: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rng = random.Random(RANDOM_SEED)
    sampled_by_dataset = {}
    hard_subset = []
    for dataset, spec in DATASET_SPECS.items():
        hard_records = sorted(hybrid_hard_by_dataset.get(dataset, []), key=lambda record: record["qid"])
        sample_size = spec["sample_size"]
        if len(hard_records) > sample_size:
            sampled = rng.sample(hard_records, sample_size)
            sampled.sort(key=lambda record: record["qid"])
        else:
            sampled = hard_records
        sampled_by_dataset[dataset] = sampled
        hard_subset.extend(sampled)
    return sampled_by_dataset, hard_subset


def _build_sample_summary(
    hybrid_hard_by_dataset: dict[str, list[dict[str, Any]]],
    sampled_by_dataset: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    dataset_summaries = {}
    for dataset, spec in DATASET_SPECS.items():
        sampled = sampled_by_dataset.get(dataset, [])
        available = hybrid_hard_by_dataset.get(dataset, [])
        dataset_summaries[dataset] = {
            "source_retriever": "hybrid",
            "requested_sample_size": spec["sample_size"],
            "available_hybrid_hard_count": len(available),
            "sampled_count": len(sampled),
            "question_type_counts": dict(Counter(record.get("question_type", "") for record in sampled)),
            "failure_type_counts": dict(Counter(record.get("failure_type", "") for record in sampled)),
        }
    return {
        "sampling_seed": RANDOM_SEED,
        "total_sampled": sum(item["sampled_count"] for item in dataset_summaries.values()),
        "datasets": dataset_summaries,
    }


def _write_json(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
