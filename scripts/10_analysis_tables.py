import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.analysis.build_tables import (
    build_failure_type_analysis,
    build_main_results,
    build_retriever_specific_results,
    build_reward_ablation_results,
)
from src.utils.io import read_jsonl, read_yaml


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])

    build_main_results(rewrite_results, data_config["main_results_path"])
    build_failure_type_analysis(rewrite_results, data_config["failure_type_analysis_path"])
    build_retriever_specific_results(rewrite_results, data_config["retriever_specific_path"])
    build_reward_ablation_results(rewrite_results, data_config["reward_ablation_path"])

    print("Saved analysis tables:")
    print(f" - {data_config['main_results_path']}")
    print(f" - {data_config['failure_type_analysis_path']}")
    print(f" - {data_config['retriever_specific_path']}")
    print(f" - {data_config['reward_ablation_path']}")


if __name__ == "__main__":
    main()
