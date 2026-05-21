import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.rewriting.policy import infer_failure_type
from src.utils.io import ensure_dir, read_jsonl, write_jsonl


INPUT_DIR = Path("data/outputs/original_retrieval")
OUTPUT_DIR = Path("data/outputs/hard_cases")
RANDOM_SEED = 42

DATASET_SPECS = {
    "korquad1": {
        "qa_path": "data/processed/korquad1_qa_pairs.jsonl",
        "target_sample_size": 50,
        "output_name": "korquad1_850_sample50.jsonl",
    },
    "klue_mrc": {
        "qa_path": "data/processed/klue_mrc_qa_pairs.jsonl",
        "target_sample_size": 350,
        "output_name": "klue_mrc_850_sample350.jsonl",
    },
    "korquad2": {
        "qa_path": "data/processed/korquad2_filtered_qa_pairs.jsonl",
        "target_sample_size": 450,
        "output_name": "korquad2_850_sample450.jsonl",
    },
}
SOURCE_RETRIEVER = "hybrid"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    rng = random.Random(RANDOM_SEED)

    sampled_by_dataset = {}
    summary_rows = []
    hard_subset = []

    for dataset, spec in DATASET_SPECS.items():
        qa_by_qid = _load_qa_by_qid(spec["qa_path"])
        valid_qids = _valid_qids_for_dataset(dataset, qa_by_qid)
        result_path = INPUT_DIR / f"{dataset}_{SOURCE_RETRIEVER}_results.jsonl"
        if not result_path.exists():
            raise FileNotFoundError(f"Original retrieval result file not found: {result_path}")

        records = read_jsonl(result_path)
        strict_hard = []
        near_hard = []
        for record in records:
            if valid_qids is not None and str(record.get("qid", "")) not in valid_qids:
                continue
            hardness_level = _hardness_level(record)
            if hardness_level is None:
                continue
            hard_case = _make_hard_case_record(record, qa_by_qid, hardness_level)
            if hardness_level == "strict_hard":
                strict_hard.append(hard_case)
            else:
                near_hard.append(hard_case)

        sampled_strict, sampled_near = _sample_strict_then_near(
            strict_hard=strict_hard,
            near_hard=near_hard,
            target_size=spec["target_sample_size"],
            rng=rng,
        )
        sampled = sorted(sampled_strict + sampled_near, key=lambda item: item["qid"])

        sampled_by_dataset[dataset] = sampled
        hard_subset.extend(sampled)
        write_jsonl(sampled, OUTPUT_DIR / spec["output_name"])

        summary_rows.append(
            _build_dataset_summary(
                dataset=dataset,
                qa_by_qid=qa_by_qid,
                target_sample_size=spec["target_sample_size"],
                strict_hard=strict_hard,
                near_hard=near_hard,
                sampled_strict=sampled_strict,
                sampled_near=sampled_near,
                sampled=sampled,
            )
        )

        print(
            f"{dataset}: strict={len(strict_hard)}, near={len(near_hard)}, "
            f"sampled={len(sampled)}/{spec['target_sample_size']}"
        )

    hard_subset.sort(key=lambda item: (item["dataset"], item["qid"]))
    write_jsonl(hard_subset, OUTPUT_DIR / "hard_subset_850.jsonl")
    _write_json(
        {
            "sampling_seed": RANDOM_SEED,
            "source_retriever": SOURCE_RETRIEVER,
            "total_sampled": len(hard_subset),
            "datasets": summary_rows,
        },
        OUTPUT_DIR / "hard_case_summary.json",
    )

    print(f"Saved final hard subset with {len(hard_subset)} records to {OUTPUT_DIR / 'hard_subset_850.jsonl'}")


def _load_qa_by_qid(qa_path: str) -> dict[str, dict[str, Any]]:
    qa_records = read_jsonl(qa_path)
    if not qa_records:
        raise ValueError(f"QA file is empty: {qa_path}")
    return {str(record["qid"]): record for record in qa_records}


def _valid_qids_for_dataset(dataset: str, qa_by_qid: dict[str, dict[str, Any]]) -> set[str] | None:
    if dataset != "korquad2":
        return None
    return {
        qid
        for qid, qa in qa_by_qid.items()
        if qa.get("synthetic_from_qa") is False
    }


def _build_dataset_summary(
    dataset: str,
    qa_by_qid: dict[str, dict[str, Any]],
    target_sample_size: int,
    strict_hard: list[dict[str, Any]],
    near_hard: list[dict[str, Any]],
    sampled_strict: list[dict[str, Any]],
    sampled_near: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "dataset": dataset,
        "target_sample_size": target_sample_size,
        "strict_hard_count": len(strict_hard),
        "near_hard_count": len(near_hard),
        "sampled_strict_hard": len(sampled_strict),
        "sampled_near_hard": len(sampled_near),
        "total_sampled": len(sampled),
        "question_type_counts": dict(Counter(item.get("question_type", "") for item in sampled)),
        "failure_type_counts": dict(Counter(item.get("failure_type", "") for item in sampled)),
    }
    if dataset == "korquad2":
        total_qa = len(qa_by_qid)
        real_qa_count = sum(1 for qa in qa_by_qid.values() if qa.get("synthetic_from_qa") is False)
        summary.update(
            {
                "korquad2_total_qa": total_qa,
                "korquad2_real_qa_count": real_qa_count,
                "korquad2_synthetic_excluded_count": total_qa - real_qa_count,
                "korquad2_strict_hard_real_only": len(strict_hard),
                "korquad2_near_hard_real_only": len(near_hard),
            }
        )
    return summary


def _hardness_level(record: dict[str, Any]) -> str | None:
    hit_at_10 = bool(record.get("hit_at_10"))
    recall_at_10 = float(record.get("recall_at_10", 0.0))
    gold_rank = record.get("gold_rank")

    if not hit_at_10 or recall_at_10 == 0.0:
        return "strict_hard"
    if hit_at_10 and gold_rank is not None and int(gold_rank) > 3:
        return "near_hard"
    return None


def _make_hard_case_record(
    record: dict[str, Any],
    qa_by_qid: dict[str, dict[str, Any]],
    hardness_level: str,
) -> dict[str, Any]:
    qid = record["qid"]
    qa = qa_by_qid.get(qid, {})
    return {
        "qid": qid,
        "dataset": record.get("dataset", qa.get("dataset", "")),
        "question": record.get("question", qa.get("question", "")),
        "question_type": record.get("question_type", qa.get("question_type", "")),
        "answer": qa.get("answer", ""),
        "gold_doc_id": record.get("gold_doc_id", qa.get("gold_doc_id", "")),
        "gold_passage": qa.get("gold_passage", ""),
        "retriever": record.get("retriever", SOURCE_RETRIEVER),
        "failure_type": infer_failure_type(record.get("question", "")),
        "hardness_level": hardness_level,
        "gold_rank": record.get("gold_rank"),
        "hit_at_10": bool(record.get("hit_at_10")),
        "recall_at_10": float(record.get("recall_at_10", 0.0)),
        "mrr": float(record.get("mrr", 0.0)),
        "top10_doc_ids": record.get("top10_doc_ids", []),
    }


def _sample_strict_then_near(
    strict_hard: list[dict[str, Any]],
    near_hard: list[dict[str, Any]],
    target_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strict_hard = sorted(strict_hard, key=lambda item: item["qid"])
    near_hard = sorted(near_hard, key=lambda item: item["qid"])

    if len(strict_hard) >= target_size:
        sampled_strict = rng.sample(strict_hard, target_size)
        sampled_strict.sort(key=lambda item: item["qid"])
        return sampled_strict, []

    sampled_strict = strict_hard
    remaining = target_size - len(sampled_strict)
    if len(near_hard) > remaining:
        sampled_near = rng.sample(near_hard, remaining)
        sampled_near.sort(key=lambda item: item["qid"])
    else:
        sampled_near = near_hard
    return sampled_strict, sampled_near


def _write_json(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
