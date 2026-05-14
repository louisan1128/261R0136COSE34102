import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_rewrites import evaluate_rewrites
from src.evaluation.reward import RewardCalculator
from src.retrievers.factory import build_retrievers
from src.utils.io import read_jsonl, read_yaml


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    top_k = config.get("top_k", 10)

    hard_cases = read_jsonl(data_config["hard_cases_path"])
    candidate_records = read_jsonl(data_config["rewrite_candidates_path"])
    retrievers = build_retrievers(config)
    reward_calculator = RewardCalculator(
        alpha=config["reward"]["alpha"],
        beta=config["reward"]["beta"],
        answer_gamma=config["reward"].get("answer_gamma", 0.5),
        lambda_=config["reward"]["lambda"],
        drift_gamma=config["reward"].get("drift_gamma", 0.2),
    )

    rewrite_results = evaluate_rewrites(
        hard_cases,
        retrievers,
        reward_calculator,
        top_k=top_k,
        out_path=data_config["rewrite_results_path"],
        candidate_records=candidate_records,
    )
    print(f"Saved rewrite evaluation results to {data_config['rewrite_results_path']}")
    print(f"Evaluated {len(rewrite_results)} rewrite candidate records.")


if __name__ == "__main__":
    main()
