import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_policies import build_final_policy_comparison, evaluate_rewrite_policies
from src.utils.io import read_jsonl, read_yaml, write_csv


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    rl_config = config.get("offline_rl", {})
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    original_contexts = _load_original_retrieval_contexts(Path("data/outputs/original_retrieval"))

    policy_rows, summary_rows = evaluate_rewrite_policies(
        rewrite_results,
        hard_cases,
        original_contexts=original_contexts,
        seed=rl_config.get("seed", 7),
        train_ratio=rl_config.get("train_ratio", 0.7),
        gamma=rl_config.get("gamma", 0.0),
    )

    policy_results_path = data_config.get("policy_results_path", "data/outputs/evaluation/policy_results.csv")
    policy_summary_path = data_config.get("policy_summary_path", "data/outputs/evaluation/policy_summary.csv")
    final_comparison_path = data_config.get("final_comparison_path", "data/outputs/evaluation/final_policy_comparison.csv")
    policy_hard_case_summary_path = data_config.get(
        "policy_summary_hard_cases_path",
        "data/outputs/evaluation/policy_summary_hard_cases.csv",
    )
    final_comparison_rows = build_final_policy_comparison(policy_rows, eval_split="test")

    write_csv(
        policy_rows,
        policy_results_path,
        fieldnames=[
            "qid",
            "retriever",
            "policy_name",
            "eval_split",
            "state_key",
            "selected_strategy",
            "failure_type",
            "failure_label",
            "refined_failure_label",
            "label_rule_group",
            "secondary_failure_label",
            "original_rank",
            "original_success",
            "originally_failed",
            "gold_rank",
            "reward",
            "policy_reward",
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "answer_f1",
            "original_gold_rank",
            "original_recall@10",
            "original_mrr",
            "original_answer_f1",
            "original_reward",
            "rank_improvement",
            "recall10_improvement",
            "mrr_improvement",
            "answer_f1_improvement",
            "reward_improvement",
            "original_query_length",
            "rewrite_query_length",
            "keyword_overlap",
            "semantic_similarity",
            "bm25_initial_rank",
            "dense_initial_rank",
            "hybrid_initial_rank",
            "rank_gap_bm25_dense",
            "rank_gap_bm25_hybrid",
            "rank_gap_dense_hybrid",
            "bm25_dense_overlap",
            "bm25_hybrid_overlap",
            "dense_hybrid_overlap",
            "retriever_consensus_overlap",
            "top1_score",
            "top2_score",
            "score_gap_top1_top2",
            "score_ratio_top1_top2",
            "original_top1_score",
            "original_top2_score",
            "original_score_gap_top1_top2",
            "original_score_ratio_top1_top2",
        ],
    )
    write_csv(summary_rows, policy_summary_path)
    write_csv(
        [row for row in summary_rows if row["subset"] == "retriever_originally_failed"],
        policy_hard_case_summary_path,
    )
    write_csv(
        final_comparison_rows,
        final_comparison_path,
        fieldnames=[
            "eval_split",
            "retriever",
            "comparison_method",
            "policy_name",
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "answer_f1",
            "avg_reward",
            "avg_policy_reward",
            "avg_rank_improvement",
            "recall10_improvement",
            "mrr_improvement",
            "answer_f1_improvement",
            "reward_improvement",
            "num_records",
        ],
    )

    print(f"Saved policy-level results to {policy_results_path}")
    print(f"Saved policy summary to {policy_summary_path}")
    print(f"Saved hard-case policy summary to {policy_hard_case_summary_path}")
    print(f"Saved final comparison to {final_comparison_path}")


def _load_original_retrieval_contexts(input_dir: Path) -> dict[str, dict[str, list[str]]]:
    contexts: dict[str, dict[str, list[str]]] = {}
    for path in sorted(input_dir.glob("*_results.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            qid = str(row.get("qid", ""))
            retriever = str(row.get("retriever", ""))
            top10 = [str(doc_id) for doc_id in row.get("top10_doc_ids", [])]
            if qid and retriever and top10:
                contexts.setdefault(qid, {})[retriever] = top10
    return contexts


if __name__ == "__main__":
    main()
