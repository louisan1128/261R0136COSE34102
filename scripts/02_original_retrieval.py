import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_jsonl


DATASET_SPECS = [
    {
        "name": "korquad1",
        "corpus_path": "data/processed/korquad1_corpus.jsonl",
        "qa_path": "data/processed/korquad1_qa_pairs.jsonl",
    },
    {
        "name": "klue_mrc",
        "corpus_path": "data/processed/klue_mrc_corpus.jsonl",
        "qa_path": "data/processed/klue_mrc_qa_pairs.jsonl",
    },
    {
        "name": "korquad2",
        "corpus_path": "data/processed/korquad2_filtered_corpus.jsonl",
        "qa_path": "data/processed/korquad2_filtered_qa_pairs.jsonl",
    },
]
DATASET_CHOICES = ("korquad1", "klue_mrc", "korquad2", "all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate original retrieval by dataset.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        help="Dataset to evaluate. Use 'all' to evaluate every dataset.",
    )
    args = parser.parse_args()

    config = read_yaml(root / "configs" / "default.yaml")
    output_dir = Path("data/outputs/original_retrieval")
    ensure_dir(output_dir)

    top_k = 10
    summary_path = output_dir / "metrics_summary.json"
    metrics_summary = _load_metrics_summary(summary_path)
    selected_specs = _select_dataset_specs(args.dataset)

    for spec in selected_specs:
        dataset_name = spec["name"]
        corpus_path = spec["corpus_path"]
        qa_path = spec["qa_path"]
        print(f"Evaluating original retrieval for {dataset_name}")

        qa_pairs = read_jsonl(qa_path)
        corpus_doc_ids = _load_corpus_doc_ids(corpus_path)
        missing_gold_count = sum(1 for qa in qa_pairs if qa.get("gold_doc_id") not in corpus_doc_ids)
        skipped_count = _count_skipped_queries(qa_pairs)
        expected_count = len(qa_pairs) - skipped_count
        metrics_summary[dataset_name] = {}

        bm25 = None
        dense = None
        force_overwrite = args.dataset != "all" or dataset_name == "korquad2"

        for retriever_name in ("bm25", "dense", "hybrid"):
            result_path = output_dir / f"{dataset_name}_{retriever_name}_results.jsonl"
            existing_records = read_jsonl(result_path)
            if not force_overwrite and len(existing_records) == expected_count:
                metrics = _metrics_from_result_records(existing_records, missing_gold_count, skipped_count)
                metrics_summary[dataset_name][retriever_name] = metrics
                print(
                    f"  {retriever_name}: resume hit, "
                    f"recall@10={metrics['recall_at_10']:.4f}, mrr={metrics['mrr']:.4f}"
                )
                continue

            if retriever_name == "bm25":
                bm25 = bm25 or BM25Retriever(corpus_path)
                retriever = bm25
            elif retriever_name == "dense":
                dense = dense or _build_dense_retriever(config, corpus_path, dataset_name)
                retriever = dense
            else:
                bm25 = bm25 or BM25Retriever(corpus_path)
                dense = dense or _build_dense_retriever(config, corpus_path, dataset_name)
                retriever = HybridRetriever(bm25, dense, alpha=config["hybrid"]["alpha"])

            result_records, metrics = evaluate_original_retrieval(
                dataset_name=dataset_name,
                retriever_name=retriever_name,
                retriever=retriever,
                qa_pairs=qa_pairs,
                corpus_doc_ids=corpus_doc_ids,
                top_k=top_k,
            )
            metrics["missing_gold_count"] = missing_gold_count
            metrics_summary[dataset_name][retriever_name] = metrics

            write_jsonl(result_records, result_path)
            print(
                f"  {retriever_name}: recall@10={metrics['recall_at_10']:.4f}, "
                f"mrr={metrics['mrr']:.4f}, saved={result_path}"
            )

    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(metrics_summary, fout, ensure_ascii=False, indent=2)
    print(f"Saved metrics summary to {summary_path}")


def _select_dataset_specs(dataset: str) -> list[dict[str, str]]:
    if dataset == "all":
        return DATASET_SPECS
    return [spec for spec in DATASET_SPECS if spec["name"] == dataset]


def _load_metrics_summary(summary_path: Path) -> dict[str, dict[str, dict[str, float | int]]]:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"Metrics summary must be a JSON object: {summary_path}")
    return data


def evaluate_original_retrieval(
    dataset_name: str,
    retriever_name: str,
    retriever: Any,
    qa_pairs: list[dict[str, Any]],
    corpus_doc_ids: set[str],
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    result_records = []
    recall_sum = 0.0
    mrr_sum = 0.0
    skipped_count = 0

    for qa in qa_pairs:
        qid = str(qa.get("qid", "")).strip()
        question = str(qa.get("question", "")).strip()
        gold_doc_id = str(qa.get("gold_doc_id", "")).strip()
        if not qid or not question or not gold_doc_id:
            skipped_count += 1
            continue

        retrieved = retriever.retrieve(question, top_k=top_k)
        top10_doc_ids = [item["doc_id"] for item in retrieved[:top_k]]
        gold_rank = _gold_rank(top10_doc_ids, gold_doc_id)
        hit_at_10 = gold_rank is not None
        recall_at_10 = 1.0 if hit_at_10 else 0.0
        reciprocal_rank = 1.0 / gold_rank if gold_rank else 0.0

        recall_sum += recall_at_10
        mrr_sum += reciprocal_rank
        result_records.append(
            {
                "qid": qid,
                "dataset": qa.get("dataset") or dataset_name,
                "source_split": qa.get("source_split", ""),
                "question": question,
                "question_type": qa.get("question_type", ""),
                "gold_doc_id": gold_doc_id,
                "retriever": retriever_name,
                "gold_rank": gold_rank,
                "hit_at_10": hit_at_10,
                "recall_at_10": recall_at_10,
                "mrr": reciprocal_rank,
                "top10_doc_ids": top10_doc_ids,
            }
        )

    num_queries = len(result_records)
    denominator = num_queries or 1
    missing_gold_count = sum(1 for qa in qa_pairs if qa.get("gold_doc_id") not in corpus_doc_ids)
    metrics = {
        "num_queries": num_queries,
        "recall_at_10": recall_sum / denominator,
        "mrr": mrr_sum / denominator,
        "missing_gold_count": missing_gold_count,
        "skipped_count": skipped_count,
    }
    return result_records, metrics


def _build_dense_retriever(config: dict[str, Any], corpus_path: str, dataset_name: str) -> DenseRetriever:
    return DenseRetriever(
        corpus_path,
        model_name=config["model_name"],
        embedding_dir=str(_embedding_dir_for_corpus(config, corpus_path, dataset_name)),
        backend=config.get("dense_backend", "auto"),
        device=config.get("dense_device"),
    )


def _embedding_dir_for_corpus(config: dict[str, Any], corpus_path: str, dataset_name: str) -> Path:
    base_dir = Path(config["data"]["embedding_dir"]) / "original_retrieval" / dataset_name
    return base_dir / _corpus_fingerprint(Path(corpus_path))


def _corpus_fingerprint(corpus_path: Path) -> str:
    stat = corpus_path.stat()
    raw = f"{corpus_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _load_corpus_doc_ids(corpus_path: str) -> set[str]:
    corpus = read_jsonl(corpus_path)
    if not corpus:
        raise ValueError(f"Corpus file is empty: {corpus_path}")
    return {str(record["doc_id"]) for record in corpus}


def _gold_rank(top_doc_ids: list[str], gold_doc_id: str) -> int | None:
    try:
        return top_doc_ids.index(gold_doc_id) + 1
    except ValueError:
        return None


def _count_skipped_queries(qa_pairs: list[dict[str, Any]]) -> int:
    skipped_count = 0
    for qa in qa_pairs:
        if not str(qa.get("qid", "")).strip():
            skipped_count += 1
        elif not str(qa.get("question", "")).strip():
            skipped_count += 1
        elif not str(qa.get("gold_doc_id", "")).strip():
            skipped_count += 1
    return skipped_count


def _metrics_from_result_records(
    result_records: list[dict[str, Any]],
    missing_gold_count: int,
    skipped_count: int,
) -> dict[str, float | int]:
    num_queries = len(result_records)
    denominator = num_queries or 1
    return {
        "num_queries": num_queries,
        "recall_at_10": sum(float(record.get("recall_at_10", 0.0)) for record in result_records) / denominator,
        "mrr": sum(float(record.get("mrr", 0.0)) for record in result_records) / denominator,
        "missing_gold_count": missing_gold_count,
        "skipped_count": skipped_count,
    }


if __name__ == "__main__":
    main()
