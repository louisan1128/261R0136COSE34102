from typing import Dict

from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever, alpha: float = 0.5):
        super().__init__(bm25_retriever.corpus_path)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        candidate_k = max(top_k * 5, 50)
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        dense_results = self.dense.retrieve(query, top_k=candidate_k)

        merged: Dict[str, Dict[str, float]] = {}
        for item in bm25_results:
            merged[item["doc_id"]] = {
                "doc_id": item["doc_id"],
                "text": item["text"],
                "bm25_score": float(item["score"]),
                "dense_score": 0.0,
            }

        for item in dense_results:
            if item["doc_id"] not in merged:
                merged[item["doc_id"]] = {
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "bm25_score": 0.0,
                    "dense_score": float(item["score"]),
                }
            else:
                merged[item["doc_id"]]["dense_score"] = float(item["score"])

        bm25_scores = [item["bm25_score"] for item in merged.values()]
        dense_scores = [item["dense_score"] for item in merged.values()]
        normalized_bm25 = self._normalize_scores(bm25_scores)
        normalized_dense = self._normalize_scores(dense_scores)

        ranked = []
        for doc, bm25_norm, dense_norm in zip(merged.values(), normalized_bm25, normalized_dense):
            hybrid_score = self.alpha * bm25_norm + (1.0 - self.alpha) * dense_norm
            ranked.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(hybrid_score),
                    "text": doc["text"],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]
