import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import answer_f1, mrr, recall_at_k
from src.evaluation.reward import RewardCalculator
from src.rewriting.candidate_generator import RewriteCandidateGenerator
from src.rewriting.policy import select_strategies
from src.utils.io import ensure_dir
from src.utils.text import tokenize


def evaluate_rewrites(
    hard_cases: list[dict[str, Any]],
    retrievers: dict[str, Any],
    reward_calculator: RewardCalculator,
    top_k: int = 10,
    out_path: str | None = None,
    candidate_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    generator = RewriteCandidateGenerator()
    candidates_by_qid = {record["qid"]: record.get("candidates", {}) for record in candidate_records or []}
    rewrite_results = []
    for hard_case in hard_cases:
        question = hard_case["question"]
        answer = hard_case.get("answer", "")
        gold_doc_id = hard_case["gold_doc_id"]
        candidates = candidates_by_qid.get(hard_case["qid"]) or generator.generate(question)
        recommended_strategies = select_strategies(hard_case.get("failure_type", "unlabeled"))
        rank_features = _rank_features(hard_case)
        original_metrics_by_retriever = {
            retriever_name: _evaluate_query(retriever, question, gold_doc_id, answer, top_k)
            for retriever_name, retriever in retrievers.items()
        }
        original_reward_by_retriever = {
            retriever_name: reward_calculator.compute_reward(
                metrics["recall@10"],
                metrics["mrr"],
                metrics["answer_f1"],
                question,
                question,
            )
            for retriever_name, metrics in original_metrics_by_retriever.items()
        }

        for strategy, candidate_query in candidates.items():
            for retriever_name, retriever in retrievers.items():
                metrics = _evaluate_query(retriever, candidate_query, gold_doc_id, answer, top_k)
                original_metrics = original_metrics_by_retriever[retriever_name]
                original_reward = original_reward_by_retriever[retriever_name]
                recall10 = metrics["recall@10"]
                score_mrr = metrics["mrr"]
                score_answer_f1 = metrics["answer_f1"]
                semantic_similarity = reward_calculator.semantic_similarity(candidate_query, question)
                keyword_overlap = _keyword_overlap(question, candidate_query)
                reward = reward_calculator.compute_reward(
                    recall10,
                    score_mrr,
                    score_answer_f1,
                    candidate_query,
                    question,
                )
                result = {
                    "qid": hard_case["qid"],
                    "question": question,
                    "answer": answer,
                    "failure_type": hard_case.get("failure_type", "unlabeled"),
                    "failed_retriever_count": len(hard_case.get("failed_retrievers", [])),
                    "retriever": retriever_name,
                    "strategy": strategy,
                    "policy_recommended": strategy in recommended_strategies,
                    "query": candidate_query,
                    "original_query_length": len(tokenize(question)),
                    "rewrite_query_length": len(tokenize(candidate_query)),
                    "keyword_overlap": keyword_overlap,
                    "semantic_similarity": semantic_similarity,
                    **rank_features,
                    "gold_rank": metrics["gold_rank"] or "",
                    "original_gold_rank": original_metrics["gold_rank"] or "",
                    "rank_improvement": _rank_improvement(original_metrics["gold_rank"], metrics["gold_rank"], top_k),
                    "original_recall@10": original_metrics["recall@10"],
                    "original_mrr": original_metrics["mrr"],
                    "original_answer_f1": original_metrics["answer_f1"],
                    "original_reward": original_reward,
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@10": recall10,
                    "mrr": score_mrr,
                    "answer_f1": score_answer_f1,
                    "reward": reward,
                    "recall10_improvement": recall10 - original_metrics["recall@10"],
                    "mrr_improvement": score_mrr - original_metrics["mrr"],
                    "answer_f1_improvement": score_answer_f1 - original_metrics["answer_f1"],
                    "reward_improvement": reward - original_reward,
                }
                rewrite_results.append(result)

    if out_path:
        ensure_dir(Path(out_path).parent)
        with Path(out_path).open("w", encoding="utf-8") as fout:
            for record in rewrite_results:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return rewrite_results


def _evaluate_query(retriever, query: str, gold_doc_id: str, answer: str, top_k: int) -> dict[str, float | int | None]:
    retrieved = retriever.retrieve(query, top_k=top_k)
    return {
        "gold_rank": _gold_rank(retrieved, gold_doc_id),
        "recall@1": recall_at_k(retrieved, gold_doc_id, 1),
        "recall@5": recall_at_k(retrieved, gold_doc_id, 5),
        "recall@10": recall_at_k(retrieved, gold_doc_id, 10),
        "mrr": mrr(retrieved, gold_doc_id),
        "answer_f1": answer_f1(retrieved, answer, k=top_k),
    }


def _gold_rank(retrieved: list[dict], gold_doc_id: str) -> int | None:
    for idx, item in enumerate(retrieved, start=1):
        if item.get("doc_id") == gold_doc_id:
            return idx
    return None


def _rank_improvement(original_rank: int | None, rewritten_rank: int | None, top_k: int) -> int:
    original_value = original_rank if original_rank is not None else top_k + 1
    rewritten_value = rewritten_rank if rewritten_rank is not None else top_k + 1
    return original_value - rewritten_value


def _keyword_overlap(original_query: str, rewritten_query: str) -> float:
    original_tokens = set(tokenize(original_query))
    rewritten_tokens = set(tokenize(rewritten_query))
    if not original_tokens:
        return 0.0
    return len(original_tokens & rewritten_tokens) / len(original_tokens)


def _rank_features(hard_case: dict[str, Any]) -> dict[str, int | str]:
    ranks = {
        retriever: _rank_of_gold(hard_case, retriever)
        for retriever in ("bm25", "dense", "hybrid")
    }
    return {
        "bm25_initial_rank": ranks["bm25"] or "",
        "dense_initial_rank": ranks["dense"] or "",
        "hybrid_initial_rank": ranks["hybrid"] or "",
        "rank_gap_bm25_dense": _rank_gap(ranks["bm25"], ranks["dense"]),
        "rank_gap_bm25_hybrid": _rank_gap(ranks["bm25"], ranks["hybrid"]),
        "rank_gap_dense_hybrid": _rank_gap(ranks["dense"], ranks["hybrid"]),
    }


def _rank_of_gold(hard_case: dict[str, Any], retriever: str) -> int | None:
    retrieved = hard_case.get("original_retrieved_by_retriever", {}).get(retriever, [])
    gold_doc_id = hard_case.get("gold_doc_id")
    if gold_doc_id in retrieved:
        return retrieved.index(gold_doc_id) + 1
    return None


def _rank_gap(left: int | None, right: int | None) -> int | str:
    if left is None or right is None:
        return ""
    return abs(left - right)
