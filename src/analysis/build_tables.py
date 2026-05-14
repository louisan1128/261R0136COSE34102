from collections import defaultdict

from src.evaluation.reward import RewardCalculator
from src.utils.io import write_csv


METRIC_FIELDS = ["recall@1", "recall@5", "recall@10", "mrr", "answer_f1", "reward"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_mean(records: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key, "") for key in keys)].append(record)

    rows = []
    for group_key, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            row[metric] = _mean([float(item.get(metric, 0.0)) for item in items])
        row["num_records"] = len(items)
        rows.append(row)
    return rows


def build_main_results(rewrite_results: list[dict], out_path: str) -> list[dict]:
    # TODO: Add confidence intervals once experiments run on full datasets.
    rows = _group_mean(rewrite_results, ["retriever", "strategy"], METRIC_FIELDS)
    write_csv(rows, out_path, fieldnames=["retriever", "strategy", *METRIC_FIELDS, "num_records"])
    return rows


def build_failure_type_analysis(rewrite_results: list[dict], out_path: str) -> list[dict]:
    rows = _group_mean(rewrite_results, ["failure_type", "retriever", "strategy"], ["recall@10", "mrr", "answer_f1", "reward"])
    write_csv(rows, out_path, fieldnames=["failure_type", "retriever", "strategy", "recall@10", "mrr", "answer_f1", "reward", "num_records"])
    return rows


def build_retriever_specific_results(rewrite_results: list[dict], out_path: str) -> list[dict]:
    rows = _group_mean(rewrite_results, ["retriever", "strategy", "policy_recommended"], ["recall@10", "mrr", "answer_f1", "reward"])
    write_csv(rows, out_path, fieldnames=["retriever", "strategy", "policy_recommended", "recall@10", "mrr", "answer_f1", "reward", "num_records"])
    return rows


def build_reward_ablation_results(rewrite_results: list[dict], out_path: str) -> list[dict]:
    settings = [
        {"reward_variant": "success_only", "alpha": 1.0, "beta": 0.0, "answer_gamma": 0.0, "lambda_": 0.0, "drift_gamma": 0.0},
        {"reward_variant": "success_plus_rank", "alpha": 1.0, "beta": 0.5, "answer_gamma": 0.0, "lambda_": 0.0, "drift_gamma": 0.0},
        {"reward_variant": "answer_aware", "alpha": 1.0, "beta": 0.5, "answer_gamma": 0.5, "lambda_": 0.0, "drift_gamma": 0.0},
        {"reward_variant": "full_reward", "alpha": 1.0, "beta": 0.5, "answer_gamma": 0.5, "lambda_": 0.02, "drift_gamma": 0.2},
        {"reward_variant": "rank_heavy", "alpha": 0.5, "beta": 1.0, "answer_gamma": 0.5, "lambda_": 0.02, "drift_gamma": 0.2},
    ]
    rows = []
    for setting in settings:
        variant = setting["reward_variant"]
        calculator = RewardCalculator(
            alpha=setting["alpha"],
            beta=setting["beta"],
            answer_gamma=setting["answer_gamma"],
            lambda_=setting["lambda_"],
            drift_gamma=setting["drift_gamma"],
        )
        best_by_query = {}
        for record in rewrite_results:
            reward = calculator.compute_reward(
                float(record["recall@10"]),
                float(record["mrr"]),
                float(record.get("answer_f1", 0.0)),
                record["query"],
                record.get("question"),
            )
            rescored = {**record, "reward": reward, "selected_strategy": record["strategy"]}
            key = (record["qid"], record["retriever"])
            current = best_by_query.get(key)
            if current is None or reward > current["reward"]:
                best_by_query[key] = rescored

        selected_records = list(best_by_query.values())
        for row in _group_mean(
            selected_records,
            ["retriever", "selected_strategy"],
            ["reward", "recall@1", "recall@5", "recall@10", "mrr", "answer_f1"],
        ):
            row["reward_variant"] = variant
            row["alpha"] = setting["alpha"]
            row["beta"] = setting["beta"]
            row["answer_gamma"] = setting["answer_gamma"]
            row["lambda"] = setting["lambda_"]
            row["drift_gamma"] = setting["drift_gamma"]
            rows.append(row)
    write_csv(
        rows,
        out_path,
        fieldnames=[
            "reward_variant",
            "retriever",
            "selected_strategy",
            "reward",
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "answer_f1",
            "num_records",
            "alpha",
            "beta",
            "answer_gamma",
            "lambda",
            "drift_gamma",
        ],
    )
    return rows
