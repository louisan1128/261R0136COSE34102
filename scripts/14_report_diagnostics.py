import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_policies import build_final_policy_comparison, evaluate_rewrite_policies
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_csv


DEFAULT_OUTPUT_DIR = Path("data/outputs/evaluation/report_diagnostics")
DEFAULT_SEEDS = [7, 13, 21, 42, 101]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report diagnostics for policy reliability and failure analysis.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for diagnostic CSV/JSON outputs.")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated policy split seeds for multi-seed evaluation.",
    )
    parser.add_argument("--skip-multiseed", action="store_true", help="Skip multi-seed hard-case policy evaluation.")
    parser.add_argument(
        "--general-eval",
        default="data/outputs/general_policy/general_policy_eval_5000.csv",
        help="General policy eval CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]

    policy_rows = _read_csv(Path(data_config["policy_results_path"]))
    final_rows = _read_csv(Path(data_config["final_comparison_path"]))
    annotation_rows = read_jsonl(data_config["hard_cases_path"])

    _write_csv(_build_action_distribution(policy_rows), output_dir / "action_distribution.csv")
    _write_csv(_build_action_distribution_by_label(policy_rows), output_dir / "action_distribution_by_label.csv")
    _write_csv(_build_hard_case_harm(policy_rows), output_dir / "rewrite_harm_hard_cases.csv")
    _write_csv(_build_oracle_gap(final_rows), output_dir / "oracle_gap_analysis.csv")
    _write_csv(_build_annotation_quality(annotation_rows), output_dir / "annotation_quality.csv")

    general_eval_path = Path(args.general_eval)
    if general_eval_path.exists():
        _write_csv(_build_general_harm(_read_csv(general_eval_path)), output_dir / "rewrite_harm_general.csv")

    _write_csv(
        _build_encoding_diagnostics(
            [
                Path(data_config["qa_path"]),
                Path(data_config["corpus_path"]),
                Path(data_config["hard_cases_path"]),
                Path(data_config["rewrite_candidates_path"]),
                Path(data_config["rewrite_results_path"]),
                general_eval_path,
            ]
        ),
        output_dir / "encoding_diagnostics.csv",
    )
    _write_json(_build_llm_rewrite_audit(config), output_dir / "llm_rewrite_audit.json")

    if not args.skip_multiseed:
        seeds = _parse_seeds(args.seeds)
        runs, summary = _build_multiseed_policy_summary(config, seeds)
        _write_csv(runs, output_dir / "policy_multiseed_runs.csv")
        _write_csv(summary, output_dir / "policy_multiseed_summary.csv")

    print(f"Saved report diagnostics to {output_dir}")


def _build_action_distribution(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        key = (row.get("eval_split", ""), row.get("retriever", ""), row.get("policy_name", ""))
        strategy = row.get("selected_strategy", "")
        grouped[key][strategy] += 1
        totals[key] += 1

    output = []
    for key, counts in sorted(grouped.items()):
        eval_split, retriever, policy_name = key
        total = totals[key]
        for strategy, count in sorted(counts.items()):
            output.append(
                {
                    "eval_split": eval_split,
                    "retriever": retriever,
                    "policy_name": policy_name,
                    "selected_strategy": strategy,
                    "count": count,
                    "rate": count / total if total else 0.0,
                    "num_records": total,
                }
            )
    return output


def _build_action_distribution_by_label(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    target_policies = {"refined_label_rule_model_policy", "retriever_tuned_bandit_policy", "state_recovery_bandit_policy"}
    grouped = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        if row.get("policy_name") not in target_policies:
            continue
        key = (
            row.get("eval_split", ""),
            row.get("retriever", ""),
            row.get("policy_name", ""),
            row.get("label_rule_group") or row.get("refined_failure_label") or row.get("failure_label", ""),
        )
        grouped[key][row.get("selected_strategy", "")] += 1
        totals[key] += 1

    output = []
    for key, counts in sorted(grouped.items()):
        eval_split, retriever, policy_name, label = key
        total = totals[key]
        for strategy, count in sorted(counts.items()):
            output.append(
                {
                    "eval_split": eval_split,
                    "retriever": retriever,
                    "policy_name": policy_name,
                    "label_group": label,
                    "selected_strategy": strategy,
                    "count": count,
                    "rate": count / total if total else 0.0,
                    "num_records": total,
                }
            )
    return output


def _build_hard_case_harm(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("eval_split", ""), row.get("retriever", ""), row.get("policy_name", ""))].append(row)
        grouped[("all", row.get("retriever", ""), row.get("policy_name", ""))].append(row)
    return [_harm_summary_row(key, items) for key, items in sorted(grouped.items())]


def _harm_summary_row(key: tuple[str, str, str], items: list[dict[str, str]]) -> dict[str, Any]:
    eval_split, retriever, policy_name = key
    total = len(items)
    rewrites = sum(1 for row in items if row.get("selected_strategy") != "original")
    harm = sum(1 for row in items if _float(row, "original_recall@10") > 0 and _float(row, "recall@10") <= 0)
    recovery = sum(1 for row in items if _float(row, "original_recall@10") <= 0 and _float(row, "recall@10") > 0)
    reward_delta = [_float(row, "reward_improvement") for row in items]
    recall_delta = [_float(row, "recall10_improvement") for row in items]
    return {
        "eval_split": eval_split,
        "retriever": retriever,
        "policy_name": policy_name,
        "num_records": total,
        "rewrite_rate": rewrites / total if total else 0.0,
        "recovery_count": recovery,
        "recovery_rate": recovery / total if total else 0.0,
        "harm_count": harm,
        "harm_rate": harm / total if total else 0.0,
        "mean_recall10_delta": _mean(recall_delta),
        "mean_reward_delta": _mean(reward_delta),
    }


def _build_general_harm(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row.get("qid", ""), row.get("retriever", ""))][row.get("policy_name", "")] = row

    grouped = defaultdict(list)
    for (_, retriever), policies in by_key.items():
        original = policies.get("original_only")
        if not original:
            continue
        for policy_name, row in policies.items():
            eval_split = row.get("eval_split", "")
            record = {
                "selected_strategy": row.get("selected_strategy", ""),
                "original_recall@10": original.get("recall@10", "0"),
                "recall@10": row.get("recall@10", "0"),
                "reward_improvement": _float(row, "reward") - _float(original, "reward"),
                "recall10_improvement": _float(row, "recall@10") - _float(original, "recall@10"),
            }
            grouped[(eval_split, retriever, policy_name)].append(record)
            grouped[("all", retriever, policy_name)].append(record)
    return [_harm_summary_row(key, items) for key, items in sorted(grouped.items())]


def _build_oracle_gap(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_retriever = defaultdict(dict)
    for row in rows:
        by_retriever[row.get("retriever", "")][row.get("policy_name", "")] = row

    output = []
    for retriever, policies in sorted(by_retriever.items()):
        original = policies.get("original_only")
        learned = policies.get("refined_label_rule_model_policy")
        oracle = policies.get("reward_selected") or policies.get("oracle_best_strategy")
        if not original or not learned or not oracle:
            continue
        for metric in ("recall@10", "mrr", "answer_f1", "avg_reward"):
            original_value = _float(original, metric)
            learned_value = _float(learned, metric)
            oracle_value = _float(oracle, metric)
            oracle_gain = oracle_value - original_value
            learned_gain = learned_value - original_value
            output.append(
                {
                    "retriever": retriever,
                    "metric": metric,
                    "original": original_value,
                    "learned_policy": learned_value,
                    "oracle": oracle_value,
                    "learned_gain": learned_gain,
                    "oracle_gain": oracle_gain,
                    "remaining_gap": oracle_value - learned_value,
                    "oracle_gap_closed": learned_gain / oracle_gain if oracle_gain else 0.0,
                }
            )
    return output


def _build_annotation_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    label_counts = Counter(str(row.get("failure_label", "") or "") for row in rows)
    secondary_counts = Counter(str(row.get("secondary_failure_label", "") or "") for row in rows)
    output = [
        {
            "scope": "primary_failure_label",
            "label": label,
            "count": count,
            "rate": count / total if total else 0.0,
            "num_records": total,
        }
        for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    output.extend(
        {
            "scope": "secondary_failure_label",
            "label": label,
            "count": count,
            "rate": count / total if total else 0.0,
            "num_records": total,
        }
        for label, count in sorted(secondary_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return output


def _build_multiseed_policy_summary(config: dict[str, Any], seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_config = config["data"]
    rl_config = config.get("offline_rl", {})
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    original_contexts = _load_original_retrieval_contexts(Path("data/outputs/original_retrieval"))

    runs = []
    for seed in seeds:
        policy_rows, _ = evaluate_rewrite_policies(
            rewrite_results,
            hard_cases,
            original_contexts=original_contexts,
            seed=seed,
            train_ratio=rl_config.get("train_ratio", 0.7),
            gamma=rl_config.get("gamma", 0.0),
        )
        for row in build_final_policy_comparison(policy_rows, eval_split="test"):
            row = dict(row)
            row["seed"] = seed
            runs.append(row)

    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["retriever"], row["policy_name"], row["comparison_method"])].append(row)

    summary = []
    for (retriever, policy_name, comparison_method), items in sorted(grouped.items()):
        base = {
            "retriever": retriever,
            "policy_name": policy_name,
            "comparison_method": comparison_method,
            "num_seeds": len(items),
        }
        for metric in ("recall@10", "mrr", "answer_f1", "avg_reward", "reward_improvement"):
            values = [_float(row, metric) for row in items]
            mean = _mean(values)
            std = _std(values)
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_ci95"] = 1.96 * std / math.sqrt(len(values)) if values else 0.0
            base[f"{metric}_min"] = min(values) if values else 0.0
            base[f"{metric}_max"] = max(values) if values else 0.0
        summary.append(base)
    return runs, summary


def _build_encoding_diagnostics(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        replacement_count = text.count("\ufffd")
        hangul_count = sum(1 for char in text if "\uac00" <= char <= "\ud7a3")
        non_ascii_count = sum(1 for char in text if ord(char) > 127)
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "line_count": line_count,
                "unicode_replacement_char_count": replacement_count,
                "lines_with_replacement_char": sum(1 for line in text.splitlines() if "\ufffd" in line),
                "hangul_char_count": hangul_count,
                "non_ascii_char_count": non_ascii_count,
                "hangul_share_of_non_ascii": hangul_count / non_ascii_count if non_ascii_count else 0.0,
            }
        )
    return rows


def _build_llm_rewrite_audit(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = config.get("llm_rewrite", {})
    data_config = config["data"]
    cache_path = Path(llm_config.get("cache_path", "data/outputs/cache/llm_rewrite_cache.jsonl"))
    summary_path = Path("data/outputs/rewrite_candidates/rewrite_candidate_summary_1000.json")
    candidate_path = Path(data_config["rewrite_candidates_path"])
    cache_rows = read_jsonl(cache_path)
    summary = _read_json(summary_path) if summary_path.exists() else {}
    candidates = read_jsonl(candidate_path)
    rewrite_type_counts = Counter()
    for record in candidates:
        for candidate in record.get("rewrite_candidates", []):
            rewrite_type_counts[str(candidate.get("rewrite_type", ""))] += 1
    return {
        "llm_config_enabled": bool(llm_config.get("enabled", False)),
        "llm_config_provider": llm_config.get("provider", ""),
        "llm_model_configured": llm_config.get("model", "") or os.environ.get(str(llm_config.get("model_env", "OPENAI_MODEL")), ""),
        "api_key_env_name": llm_config.get("api_key_env", "OPENAI_API_KEY"),
        "api_key_present_in_environment": bool(os.environ.get(str(llm_config.get("api_key_env", "OPENAI_API_KEY")), "")),
        "allow_fallback": bool(llm_config.get("allow_fallback", True)),
        "cache_path": str(cache_path),
        "cache_rows": len(cache_rows),
        "candidate_path": str(candidate_path),
        "candidate_questions": len(candidates),
        "rewrite_type_distribution": dict(sorted(rewrite_type_counts.items())),
        "generation_summary": summary,
        "interpretation": (
            "If cache_rows is zero or no API key/model was configured when script 07 ran, "
            "llm_rewrite should be reported as LLM-style fallback rather than verified external LLM output."
        ),
    }


def _load_original_retrieval_contexts(input_dir: Path) -> dict[str, dict[str, list[str]]]:
    contexts: dict[str, dict[str, list[str]]] = {}
    for path in sorted(input_dir.glob("*_results.jsonl")):
        for row in read_jsonl(path):
            qid = str(row.get("qid", ""))
            retriever = str(row.get("retriever", ""))
            top10 = [str(doc_id) for doc_id in row.get("top10_doc_ids", [])]
            if qid and retriever and top10:
                contexts.setdefault(qid, {})[retriever] = top10
    return contexts


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    write_csv(rows, path, fieldnames=fieldnames)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def _parse_seeds(value: str) -> list[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds or DEFAULT_SEEDS


def _float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key, 0.0)
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    main()
