import json
import sys
import os
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import read_jsonl, read_yaml, ensure_dir, write_csv


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    hard_case_by_qid = {record["qid"]: record for record in hard_cases}
    if not rewrite_results:
        print(
            f"No rewrite results found at {data_config['rewrite_results_path']}. "
            "Run scripts/08_rewrite_retrieval_eval.py first."
        )
        write_csv(
            [],
            data_config["recovery_path"],
            fieldnames=["case_subset", "retriever", "recovered_at_10", "total_cases", "recovery@10"],
        )
        return

    best_by_retriever = {}
    for record in rewrite_results:
        key = (record["qid"], record["retriever"])
        current = best_by_retriever.get(key)
        if current is None or record["reward"] > current["reward"]:
            best_by_retriever[key] = record

    grouped = defaultdict(dict)
    for record in best_by_retriever.values():
        grouped[record["qid"]][record["retriever"]] = {
            "strategy": record["strategy"],
            "query": record["query"],
            "reward": record["reward"],
            "recall@10": record["recall@10"],
            "mrr": record["mrr"],
        }

    best_records = []
    for qid, best_query_by_retriever in grouped.items():
        sample = next(record for record in rewrite_results if record["qid"] == qid)
        best_records.append(
            {
                "qid": qid,
                "failure_type": sample.get("failure_type", "unlabeled"),
                "failed_retrievers": hard_case_by_qid.get(qid, {}).get("failed_retrievers", []),
                "best_query_by_retriever": best_query_by_retriever,
            }
        )

    output_path = Path(data_config["best_queries_path"])
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in best_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    recovery_rows = _build_recovery_rows(list(best_by_retriever.values()), hard_case_by_qid)
    for row in recovery_rows:
        print(
            f"{row['case_subset']} / {row['retriever']} recovery@10: "
            f"{row['recovered_at_10']}/{row['total_cases']} = {row['recovery@10']:.4f}"
        )

    recovery_path = Path(data_config["recovery_path"])
    write_csv(recovery_rows, recovery_path)
    print(f"Saved best query selections to {output_path}")
    print(f"Saved recovery summary to {recovery_path}")


def _build_recovery_rows(flat_best: list[dict], hard_case_by_qid: dict[str, dict]) -> list[dict]:
    rows = []
    for subset_name, include_record in [
        ("union_hard_cases", lambda record: True),
        (
            "retriever_originally_failed",
            lambda record: record["retriever"] in hard_case_by_qid.get(record["qid"], {}).get("failed_retrievers", []),
        ),
        (
            "all_retrievers_originally_failed",
            lambda record: len(hard_case_by_qid.get(record["qid"], {}).get("failed_retrievers", [])) >= 3,
        ),
    ]:
        totals = defaultdict(int)
        recovery_counts = defaultdict(int)
        for record in flat_best:
            if not include_record(record):
                continue
            retriever_name = record["retriever"]
            totals[retriever_name] += 1
            if record["recall@10"] > 0:
                recovery_counts[retriever_name] += 1

        for retriever_name in sorted({record["retriever"] for record in flat_best}):
            total = totals[retriever_name]
            count = recovery_counts[retriever_name]
            rows.append(
                {
                    "case_subset": subset_name,
                    "retriever": retriever_name,
                    "recovered_at_10": count,
                    "total_cases": total,
                    "recovery@10": count / total if total else 0.0,
                }
            )
    return rows


if __name__ == "__main__":
    main()
