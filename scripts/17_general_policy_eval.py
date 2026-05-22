import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_policies import (
    _fit_linear_ranker,
    _recovery_utility,
    _select_calibrated_ranker_action,
    _tune_retriever_thresholds,
)
from src.evaluation.metrics import answer_f1, mrr, recall_at_k
from src.evaluation.reward import RewardCalculator
from src.retrievers.factory import build_retrievers
from src.rewriting.candidate_generator import RewriteCandidateGenerator
from src.utils.io import ensure_dir, read_jsonl, read_yaml
from src.utils.text import tokenize


DEFAULT_OUTPUT = Path("data/outputs/general_policy_eval.csv")
DEFAULT_SUMMARY = Path("data/outputs/general_policy_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rewrite policies on the general QA set.")
    parser.add_argument("--qa-path", default=None, help="QA JSONL path. Defaults to configs/default.yaml data.qa_path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Per-query policy output CSV path.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Policy summary CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional deterministic sample size.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic sampling seed when --limit is used.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="General-set split used to tune score gates.")
    parser.add_argument(
        "--dense-backend",
        choices=["config", "sentence_transformers", "lexical", "auto"],
        default="config",
        help="Override dense backend. Use lexical for offline smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading config and inputs...", flush=True)
    config = read_yaml(root / "configs" / "default.yaml")
    if args.dense_backend != "config":
        config["dense_backend"] = args.dense_backend
    data_config = config["data"]
    top_k = int(config.get("top_k", 10))

    qa_path = Path(args.qa_path or data_config["qa_path"])
    qa_records = read_jsonl(qa_path)
    if args.sample_size is not None:
        qa_records = _sample_records(qa_records, args.sample_size, args.seed)
    if args.limit is not None:
        qa_records = qa_records[: args.limit]

    hard_cases = read_jsonl(data_config["hard_cases_path"])
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])
    print(
        f"Loaded {len(qa_records)} QA records, {len(hard_cases)} hard cases, "
        f"{len(rewrite_results)} rewrite eval rows.",
        flush=True,
    )
    print("Fitting calibrated recovery policy from hard-case results...", flush=True)
    policy_fit = _fit_general_policy(rewrite_results, hard_cases)

    print(
        f"Building retrievers with dense_backend={config.get('dense_backend')} "
        f"and dense_device={config.get('dense_device')}...",
        flush=True,
    )
    retrievers = build_retrievers(config)
    print(f"Built retrievers: {', '.join(retrievers)}", flush=True)
    reward_calculator = RewardCalculator(
        alpha=config["reward"]["alpha"],
        beta=config["reward"]["beta"],
        answer_gamma=config["reward"].get("answer_gamma", 0.5),
        lambda_=config["reward"]["lambda"],
        drift_gamma=config["reward"].get("drift_gamma", 0.2),
    )

    generator = RewriteCandidateGenerator()
    rows = []
    train_qids, test_qids = _split_qids([str(record["qid"]) for record in qa_records], args.train_ratio, args.seed)
    iterator = tqdm(qa_records, desc="General policy eval", unit="question") if tqdm else qa_records
    for qa in iterator:
        rows.extend(
            _evaluate_question(
                qa,
                retrievers,
                generator,
                reward_calculator,
                policy_fit,
                top_k,
                eval_split=_eval_split(str(qa["qid"]), train_qids, test_qids),
            )
        )
    rows.extend(_build_gated_policy_rows(rows))

    summary_rows = _summarize(rows)
    _write_csv(rows, Path(args.output))
    _write_csv(summary_rows, Path(args.summary))
    print(f"Saved general policy eval rows to {args.output}")
    print(f"Saved general policy summary to {args.summary}")


def _fit_general_policy(
    rewrite_results: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
) -> tuple[list[str], Any, dict[str, float]]:
    hard_case_by_qid = {record["qid"]: record for record in hard_cases}
    train_qids = {record["qid"] for record in hard_cases}
    fit = _fit_linear_ranker(
        _index_rewrite_results(rewrite_results),
        hard_case_by_qid,
        train_qids,
        target_fn=_recovery_utility,
    )
    if fit is None:
        raise RuntimeError("Could not fit calibrated recovery policy.")
    feature_names, weights = fit
    thresholds = _tune_retriever_thresholds(
        _index_rewrite_results(rewrite_results),
        hard_case_by_qid,
        train_qids,
        feature_names,
        weights,
        target_fn=_recovery_utility,
    )
    return feature_names, weights, thresholds


def _index_rewrite_results(rewrite_results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict]]:
    indexed = defaultdict(dict)
    for record in rewrite_results:
        indexed[(record["qid"], record["retriever"])][record["strategy"]] = record
    return indexed


def _evaluate_question(
    qa: dict[str, Any],
    retrievers: dict[str, Any],
    generator: RewriteCandidateGenerator,
    reward_calculator: RewardCalculator,
    policy_fit: tuple[list[str], Any, dict[str, float]],
    top_k: int,
    eval_split: str,
) -> list[dict[str, Any]]:
    qid = str(qa["qid"])
    question = str(qa["question"])
    answer = str(qa.get("answer", ""))
    gold_doc_id = str(qa["gold_doc_id"])
    query_candidates = generator.generate(question)
    original_by_retriever = {
        name: retriever.retrieve(question, top_k=top_k)
        for name, retriever in retrievers.items()
    }
    hard_case = {
        "qid": qid,
        "question": question,
        "answer": answer,
        "gold_doc_id": gold_doc_id,
        "failure_type": "general",
        "failure_label": "",
        "secondary_failure_label": "",
        "question_type": qa.get("question_type", ""),
        "failed_retrievers": [
            name for name, retrieved in original_by_retriever.items() if not _gold_rank(retrieved, gold_doc_id)
        ],
        "original_retrieved_by_retriever": {
            name: [item["doc_id"] for item in retrieved]
            for name, retrieved in original_by_retriever.items()
        },
    }

    rows = []
    feature_names, weights, thresholds = policy_fit
    for retriever_name, retriever in retrievers.items():
        policy_candidates = _make_policy_candidate_records(qid, question, query_candidates)
        selected_rl = _select_calibrated_ranker_action(
            policy_candidates,
            hard_case,
            retriever_name,
            feature_names,
            weights,
            thresholds.get(retriever_name, 0.0),
        )
        for policy_name, strategy in (
            ("original_only", "original"),
            ("always_llm", "llm"),
            ("rl_selected", selected_rl),
        ):
            query = query_candidates[strategy]
            if strategy == "original":
                retrieved = original_by_retriever[retriever_name]
            else:
                retrieved = retriever.retrieve(query, top_k=top_k)
            metrics = _metrics(retrieved, gold_doc_id, answer, top_k)
            reward = reward_calculator.compute_reward(
                metrics["recall@10"],
                metrics["mrr"],
                metrics["answer_f1"],
                query,
                question,
            )
            rows.append(
                {
                    "qid": qid,
                    "dataset": qa.get("dataset", ""),
                    "retriever": retriever_name,
                    "policy_name": policy_name,
                    "eval_split": eval_split,
                    "selected_strategy": strategy,
                    "query": query,
                    "gold_rank": metrics["gold_rank"] or "",
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@10": metrics["recall@10"],
                    "mrr": metrics["mrr"],
                    "answer_f1": metrics["answer_f1"],
                    "reward": reward,
                    "original_success": bool(_gold_rank(original_by_retriever[retriever_name], gold_doc_id)),
                    "original_rank": _gold_rank(original_by_retriever[retriever_name], gold_doc_id) or "",
                    "original_top_score": _top_score(original_by_retriever[retriever_name]),
                    "original_score_gap": _score_gap(original_by_retriever[retriever_name]),
                }
            )
    return rows


def _split_qids(qid_order: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    train_ratio = min(max(train_ratio, 0.1), 0.9)
    shuffled = list(dict.fromkeys(qid_order))
    import random

    random.Random(seed).shuffle(shuffled)
    split_idx = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    return set(shuffled[:split_idx]), set(shuffled[split_idx:])


def _sample_records(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(records):
        return records
    import random

    indices = sorted(random.Random(seed).sample(range(len(records)), sample_size))
    return [records[index] for index in indices]


def _eval_split(qid: str, train_qids: set[str], test_qids: set[str]) -> str:
    if qid in train_qids:
        return "train"
    if qid in test_qids:
        return "test"
    return "unknown"


def _build_gated_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["qid"], row["retriever"])][row["policy_name"]] = row

    thresholds = _tune_score_gate_thresholds(by_key)
    gated_rows = []
    for (qid, retriever), policies in sorted(by_key.items()):
        original = policies.get("original_only")
        rl = policies.get("rl_selected")
        if not original or not rl:
            continue

        oracle_source = original if original["original_success"] else rl
        gated_rows.append(_copy_policy_row(oracle_source, "oracle_gated_rl", original, rl))

        confidence = _confidence(original)
        threshold = thresholds.get(retriever, float("inf"))
        score_source = original if confidence >= threshold else rl
        gated_rows.append(_copy_policy_row(score_source, "score_gated_rl", original, rl))
    return gated_rows


def _tune_score_gate_thresholds(by_key: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, float]:
    thresholds = {}
    retrievers = sorted({retriever for _, retriever in by_key})
    for retriever in retrievers:
        train_items = [
            policies
            for (qid, item_retriever), policies in by_key.items()
            if item_retriever == retriever
            and policies.get("original_only", {}).get("eval_split") == "train"
            and "original_only" in policies
            and "rl_selected" in policies
        ]
        if not train_items:
            thresholds[retriever] = float("inf")
            continue

        candidates = sorted({_confidence(policies["original_only"]) for policies in train_items})
        candidates = [float("-inf"), *candidates, float("inf")]
        best_threshold = float("inf")
        best_score = None
        for threshold in candidates:
            rewards = []
            for policies in train_items:
                original = policies["original_only"]
                rl = policies["rl_selected"]
                selected = original if _confidence(original) >= threshold else rl
                rewards.append(float(selected["reward"]))
            score = _mean(rewards)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[retriever] = best_threshold
    return thresholds


def _copy_policy_row(source: dict[str, Any], policy_name: str, original: dict[str, Any], rl: dict[str, Any]) -> dict[str, Any]:
    copied = dict(source)
    copied["policy_name"] = policy_name
    copied["selected_strategy"] = source["selected_strategy"]
    copied["rewrite_rate_source"] = "original" if source is original else "rl_selected"
    copied["rl_candidate_strategy"] = rl["selected_strategy"]
    return copied


def _confidence(original_row: dict[str, Any]) -> float:
    return float(original_row.get("original_top_score") or 0.0) + float(original_row.get("original_score_gap") or 0.0)


def _make_policy_candidate_records(qid: str, question: str, candidates: dict[str, str]) -> dict[str, dict[str, Any]]:
    original_len = len(tokenize(question))
    original_tokens = set(tokenize(question))
    records = {}
    for strategy, query in candidates.items():
        query_tokens = tokenize(query)
        records[strategy] = {
            "qid": qid,
            "question": question,
            "strategy": strategy,
            "query": query,
            "original_query_length": original_len,
            "rewrite_query_length": len(query_tokens),
            "keyword_overlap": _keyword_overlap(original_tokens, query_tokens),
            "semantic_similarity": _keyword_overlap(original_tokens, query_tokens),
            "failure_type": "general",
            "failure_label": "",
            "secondary_failure_label": "",
        }
    return records


def _keyword_overlap(original_tokens: set[str], query_tokens: list[str]) -> float:
    if not original_tokens:
        return 0.0
    return len(original_tokens & set(query_tokens)) / len(original_tokens)


def _metrics(retrieved: list[dict], gold_doc_id: str, answer: str, top_k: int) -> dict[str, float | int | None]:
    return {
        "gold_rank": _gold_rank(retrieved, gold_doc_id),
        "recall@1": recall_at_k(retrieved, gold_doc_id, 1),
        "recall@5": recall_at_k(retrieved, gold_doc_id, 5),
        "recall@10": recall_at_k(retrieved, gold_doc_id, 10),
        "mrr": mrr(retrieved, gold_doc_id),
        "answer_f1": answer_f1(retrieved, answer, k=top_k),
    }


def _gold_rank(retrieved: list[dict], gold_doc_id: str) -> int | None:
    for index, item in enumerate(retrieved, start=1):
        if item.get("doc_id") == gold_doc_id:
            return index
    return None


def _top_score(retrieved: list[dict]) -> float:
    return float(retrieved[0].get("score", 0.0)) if retrieved else 0.0


def _score_gap(retrieved: list[dict]) -> float:
    if len(retrieved) < 2:
        return 0.0
    return float(retrieved[0].get("score", 0.0)) - float(retrieved[1].get("score", 0.0))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[("all", row["retriever"], row["policy_name"])].append(row)
        grouped[(row.get("eval_split", "unknown"), row["retriever"], row["policy_name"])].append(row)
    summary = []
    for (subset, retriever, policy_name), items in sorted(grouped.items()):
        summary.append(
            {
                "subset": subset,
                "retriever": retriever,
                "policy_name": policy_name,
                "recall@1": _mean(item["recall@1"] for item in items),
                "recall@5": _mean(item["recall@5"] for item in items),
                "recall@10": _mean(item["recall@10"] for item in items),
                "mrr": _mean(item["mrr"] for item in items),
                "answer_f1": _mean(item["answer_f1"] for item in items),
                "reward": _mean(item["reward"] for item in items),
                "rewrite_rate": _mean(1.0 if item["selected_strategy"] != "original" else 0.0 for item in items),
                "num_records": len(items),
            }
        )
    return summary


def _mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
