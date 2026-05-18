import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_original import load_qa_pairs
from src.evaluation.metrics import mrr, recall_at_k
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.utils.io import read_yaml, write_csv


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    top_k = config.get("top_k", 10)
    alpha_values = [float(alpha) for alpha in config.get("hybrid", {}).get("alpha_values", [])]
    if not alpha_values:
        alpha_values = [float(config["hybrid"]["alpha"])]

    qa_pairs = load_qa_pairs(data_config["qa_path"])
    bm25 = BM25Retriever(data_config["corpus_path"])
    dense = DenseRetriever(
        data_config["corpus_path"],
        model_name=config["model_name"],
        embedding_dir=data_config["embedding_dir"],
        backend=config.get("dense_backend", "auto"),
        device=config.get("dense_device"),
    )

    totals = {
        alpha: {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        for alpha in alpha_values
    }

    for idx, qa in enumerate(qa_pairs, start=1):
        candidate_k = max(top_k * 5, 50)
        bm25_results = bm25.retrieve(qa["question"], top_k=candidate_k)
        dense_results = dense.retrieve(qa["question"], top_k=candidate_k)
        for alpha in alpha_values:
            retrieved = _rank_hybrid_from_results(bm25_results, dense_results, alpha, top_k)
            gold_doc_id = qa["gold_doc_id"]
            totals[alpha]["recall@1"] += recall_at_k(retrieved, gold_doc_id, 1)
            totals[alpha]["recall@5"] += recall_at_k(retrieved, gold_doc_id, 5)
            totals[alpha]["recall@10"] += recall_at_k(retrieved, gold_doc_id, 10)
            totals[alpha]["mrr"] += mrr(retrieved, gold_doc_id)
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(qa_pairs)} questions")

    rows = []
    total_questions = len(qa_pairs) or 1
    for alpha in alpha_values:
        metrics = {name: value / total_questions for name, value in totals[alpha].items()}
        row = {
            "retriever": "hybrid",
            "hybrid_alpha": alpha,
            "bm25_weight": alpha,
            "dense_weight": 1.0 - alpha,
            "dense_backend": dense.backend,
            "model_name": config["model_name"],
            "num_questions": total_questions,
            **metrics,
        }
        rows.append(row)
        print(f"alpha={alpha:.2f} recall@10={metrics['recall@10']:.4f} mrr={metrics['mrr']:.4f}")

    output_path = Path(data_config["hybrid_alpha_sweep_path"])
    write_csv(rows, output_path)
    print(f"Saved hybrid alpha sweep results to {output_path}")


def _rank_hybrid_from_results(bm25_results: list[dict], dense_results: list[dict], alpha: float, top_k: int) -> list[dict]:
    merged = {}
    for item in bm25_results:
        merged[item["doc_id"]] = {
            "doc_id": item["doc_id"],
            "text": item.get("text", ""),
            "bm25_score": float(item["score"]),
            "dense_score": 0.0,
        }
    for item in dense_results:
        if item["doc_id"] not in merged:
            merged[item["doc_id"]] = {
                "doc_id": item["doc_id"],
                "text": item.get("text", ""),
                "bm25_score": 0.0,
                "dense_score": float(item["score"]),
            }
        else:
            merged[item["doc_id"]]["dense_score"] = float(item["score"])

    docs = list(merged.values())
    bm25_norm = _normalize_scores([doc["bm25_score"] for doc in docs])
    dense_norm = _normalize_scores([doc["dense_score"] for doc in docs])
    ranked = []
    for doc, bm25_score, dense_score in zip(docs, bm25_norm, dense_norm):
        ranked.append(
            {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "score": alpha * bm25_score + (1.0 - alpha) * dense_score,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


if __name__ == "__main__":
    main()
