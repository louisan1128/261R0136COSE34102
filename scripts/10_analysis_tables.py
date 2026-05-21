import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.analysis.build_tables import (
    build_failure_label_analysis,
    build_failure_type_analysis,
    build_main_results,
    build_retriever_specific_results,
    build_reward_ablation_results,
)
from src.utils.io import read_jsonl, read_yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])

    table_jobs = [
        ("main results", build_main_results, data_config["main_results_path"]),
        ("failure type analysis", build_failure_type_analysis, data_config["failure_type_analysis_path"]),
    ]
    if data_config.get("failure_label_analysis_path"):
        table_jobs.append(
            ("failure label analysis", build_failure_label_analysis, data_config["failure_label_analysis_path"])
        )
    table_jobs.extend(
        [
            ("retriever specific results", build_retriever_specific_results, data_config["retriever_specific_path"]),
            ("reward ablation results", build_reward_ablation_results, data_config["reward_ablation_path"]),
        ]
    )

    progress = tqdm(table_jobs, desc="Building analysis tables", unit="table") if tqdm else table_jobs
    for _, builder, out_path in progress:
        builder(rewrite_results, out_path)

    print("Saved analysis tables:")
    print(f" - {data_config['main_results_path']}")
    print(f" - {data_config['failure_type_analysis_path']}")
    if data_config.get("failure_label_analysis_path"):
        print(f" - {data_config['failure_label_analysis_path']}")
    print(f" - {data_config['retriever_specific_path']}")
    print(f" - {data_config['reward_ablation_path']}")


if __name__ == "__main__":
    main()
