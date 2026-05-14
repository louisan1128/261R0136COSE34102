import random
import math
from collections import defaultdict
from typing import Callable

from src.rewriting.policy import select_strategies
from src.utils.text import tokenize


STRATEGIES = ["original", "keyword", "expanded", "prompt", "structured", "llm"]


def evaluate_rewrite_policies(
    rewrite_results: list[dict],
    hard_cases: list[dict],
    seed: int = 7,
    train_ratio: float = 0.7,
    gamma: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    records_by_key = _index_rewrite_results(rewrite_results)
    hard_case_by_qid = {record["qid"]: record for record in hard_cases}
    qid_order = _stable_qid_order(rewrite_results)
    train_qids, test_qids = _split_qids(qid_order, train_ratio)

    policy_rows = []
    policy_rows.extend(_evaluate_static_policies(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_random_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids, seed=seed))
    policy_rows.extend(_evaluate_epsilon_greedy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids, seed=seed))
    policy_rows.extend(_evaluate_ucb_bandit(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_thompson_sampling(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids, seed=seed))
    policy_rows.extend(_evaluate_contextual_bandit(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(
        _evaluate_offline_q_learning(
            records_by_key,
            hard_case_by_qid,
            train_qids,
            test_qids,
            gamma=gamma,
        )
    )

    summary_rows = _summarize_policy_rows(policy_rows)
    return policy_rows, summary_rows


def _index_rewrite_results(rewrite_results: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    indexed = defaultdict(dict)
    for record in rewrite_results:
        indexed[(record["qid"], record["retriever"])][record["strategy"]] = record
    return indexed


def _stable_qid_order(rewrite_results: list[dict]) -> list[str]:
    seen = set()
    qids = []
    for record in rewrite_results:
        qid = record["qid"]
        if qid not in seen:
            seen.add(qid)
            qids.append(qid)
    return qids


def _split_qids(qid_order: list[str], train_ratio: float) -> tuple[set[str], set[str]]:
    if not qid_order:
        return set(), set()
    train_ratio = min(max(train_ratio, 0.1), 0.9)
    split_idx = max(1, min(len(qid_order) - 1, int(len(qid_order) * train_ratio)))
    return set(qid_order[:split_idx]), set(qid_order[split_idx:])


def _evaluate_static_policies(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    policies: dict[str, Callable[[dict[str, dict], dict], str]] = {
        "original_only": lambda candidates, hard_case: "original",
        "always_keyword": lambda candidates, hard_case: "keyword",
        "always_expanded": lambda candidates, hard_case: "expanded",
        "always_prompt": lambda candidates, hard_case: "prompt",
        "always_structured": lambda candidates, hard_case: "structured",
        "always_llm": lambda candidates, hard_case: "llm",
        "failure_type_policy": _select_failure_type_policy,
        "oracle_best_strategy": _select_oracle,
    }

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            for policy_name, selector in policies.items():
                strategy = selector(candidates, hard_case)
                rows.append(
                    _policy_row(
                        policy_name,
                        strategy,
                        candidates,
                        hard_case,
                        retriever,
                        eval_split=_eval_split(qid, train_qids, test_qids),
                    )
                )
    return rows


def _evaluate_random_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            strategy = rng.choice(available)
            rows.append(
                _policy_row(
                    "random_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_epsilon_greedy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    seed: int,
    epsilon: float = 0.1,
) -> list[dict]:
    rng = random.Random(seed)
    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    rows = []

    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            if rng.random() < epsilon or not any(counts[retriever].values()):
                strategy = rng.choice(available)
            else:
                strategy = max(available, key=lambda item: values[retriever][item])
            row = _policy_row(
                "epsilon_greedy_bandit",
                strategy,
                candidates,
                hard_case,
                retriever,
                eval_split=_eval_split(qid, train_qids, test_qids),
            )
            rows.append(row)
            counts[retriever][strategy] += 1
            step = counts[retriever][strategy]
            values[retriever][strategy] += (row["reward"] - values[retriever][strategy]) / step

    return rows


def _evaluate_ucb_bandit(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    c: float = 1.0,
) -> list[dict]:
    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    rows = []

    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            untried = [strategy for strategy in available if counts[retriever][strategy] == 0]
            if untried:
                strategy = untried[0]
            else:
                total = max(1, total_counts[retriever])
                strategy = max(
                    available,
                    key=lambda item: values[retriever][item]
                    + c * math.sqrt(math.log(total + 1) / counts[retriever][item]),
                )

            row = _policy_row(
                "ucb_bandit",
                strategy,
                candidates,
                hard_case,
                retriever,
                eval_split=_eval_split(qid, train_qids, test_qids),
            )
            rows.append(row)
            counts[retriever][strategy] += 1
            total_counts[retriever] += 1
            step = counts[retriever][strategy]
            values[retriever][strategy] += (row["reward"] - values[retriever][strategy]) / step
    return rows


def _evaluate_thompson_sampling(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    successes = defaultdict(lambda: defaultdict(float))
    failures = defaultdict(lambda: defaultdict(float))
    rows = []

    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            strategy = max(
                available,
                key=lambda item: rng.betavariate(
                    1.0 + successes[retriever][item],
                    1.0 + failures[retriever][item],
                ),
            )

            row = _policy_row(
                "thompson_sampling",
                strategy,
                candidates,
                hard_case,
                retriever,
                eval_split=_eval_split(qid, train_qids, test_qids),
            )
            rows.append(row)
            if row["recall@10"] > 0:
                successes[retriever][strategy] += 1.0
            else:
                failures[retriever][strategy] += 1.0
    return rows


def _evaluate_contextual_bandit(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    global_values = defaultdict(lambda: defaultdict(float))
    global_counts = defaultdict(lambda: defaultdict(int))
    rows = []

    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        context = hard_case.get("failure_type", "unlabeled")
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            recommended = [strategy for strategy in select_strategies(context) if strategy in candidates]
            search_space = recommended or available
            context_key = (retriever, context)

            if any(counts[context_key].values()):
                strategy = max(search_space, key=lambda item: values[context_key][item])
            elif any(global_counts[retriever].values()):
                strategy = max(search_space, key=lambda item: global_values[retriever][item])
            else:
                strategy = search_space[0]

            row = _policy_row(
                "contextual_bandit",
                strategy,
                candidates,
                hard_case,
                retriever,
                eval_split=_eval_split(qid, train_qids, test_qids),
            )
            rows.append(row)

            counts[context_key][strategy] += 1
            context_step = counts[context_key][strategy]
            values[context_key][strategy] += (row["reward"] - values[context_key][strategy]) / context_step

            global_counts[retriever][strategy] += 1
            global_step = global_counts[retriever][strategy]
            global_values[retriever][strategy] += (row["reward"] - global_values[retriever][strategy]) / global_step

    return rows


def _evaluate_offline_q_learning(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
    test_qids: set[str],
    gamma: float,
) -> list[dict]:
    """Train a one-step offline Q policy from logged rewrite rewards.

    Query rewriting is modeled as a contextual one-step MDP:
    state = query failure/retriever context, action = rewrite strategy,
    reward = retrieval reward, next state is terminal. Since the next state is
    terminal, gamma is kept for notation but has no effect when set to 0.
    """

    q_values = defaultdict(lambda: defaultdict(list))
    fallback_values = defaultdict(lambda: defaultdict(list))

    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            state = _state_key(hard_case, retriever)
            candidates = records_by_key[(qid, retriever)]
            for strategy, record in candidates.items():
                target = float(record["reward"]) + gamma * 0.0
                q_values[(retriever, state)][strategy].append(target)
                fallback_values[retriever][strategy].append(target)

    rows = []
    for qid in sorted(test_qids):
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            state = _state_key(hard_case, retriever)
            strategy = _select_q_action(available, q_values[(retriever, state)], fallback_values[retriever])
            rows.append(
                _policy_row(
                    "offline_q_learning",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split="test",
                    state_key=state,
                )
            )
    return rows


def _select_q_action(
    available: list[str],
    state_values: dict[str, list[float]],
    fallback_values: dict[str, list[float]],
) -> str:
    def mean_or_missing(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    scored = []
    for strategy in available:
        value = mean_or_missing(state_values.get(strategy, []))
        if value is None:
            value = mean_or_missing(fallback_values.get(strategy, []))
        if value is not None:
            scored.append((value, strategy))
    if scored:
        return max(scored)[1]
    return "original" if "original" in available else available[0]


def _select_failure_type_policy(candidates: dict[str, dict], hard_case: dict) -> str:
    recommended = [strategy for strategy in select_strategies(hard_case.get("failure_type", "unlabeled")) if strategy in candidates]
    if not recommended:
        recommended = [strategy for strategy in STRATEGIES if strategy in candidates]
    return recommended[0]


def _select_oracle(candidates: dict[str, dict], hard_case: dict) -> str:
    return max(candidates, key=lambda strategy: float(candidates[strategy]["reward"]))


def _retrievers_for_qid(records_by_key: dict[tuple[str, str], dict[str, dict]], qid: str) -> list[str]:
    return sorted(retriever for candidate_qid, retriever in records_by_key if candidate_qid == qid)


def _policy_row(
    policy_name: str,
    strategy: str,
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    eval_split: str,
    state_key: str | None = None,
) -> dict:
    if strategy not in candidates:
        strategy = "original" if "original" in candidates else next(iter(candidates))
    record = candidates[strategy]
    original_rank = _original_rank(hard_case, retriever)
    state_key = state_key or _state_key(hard_case, retriever)
    return {
        "qid": record["qid"],
        "retriever": retriever,
        "policy_name": policy_name,
        "eval_split": eval_split,
        "state_key": state_key,
        "selected_strategy": strategy,
        "failure_type": hard_case.get("failure_type", record.get("failure_type", "unlabeled")),
        "original_rank": original_rank or "",
        "original_success": bool(original_rank),
        "originally_failed": retriever in hard_case.get("failed_retrievers", []),
        "reward": float(record["reward"]),
        "recall@1": float(record["recall@1"]),
        "recall@5": float(record["recall@5"]),
        "recall@10": float(record["recall@10"]),
        "mrr": float(record["mrr"]),
        "answer_f1": float(record.get("answer_f1", 0.0)),
        "original_query_length": record.get("original_query_length", ""),
        "rewrite_query_length": record.get("rewrite_query_length", ""),
        "keyword_overlap": record.get("keyword_overlap", ""),
        "semantic_similarity": record.get("semantic_similarity", ""),
        "bm25_initial_rank": record.get("bm25_initial_rank", ""),
        "dense_initial_rank": record.get("dense_initial_rank", ""),
        "hybrid_initial_rank": record.get("hybrid_initial_rank", ""),
        "rank_gap_bm25_dense": record.get("rank_gap_bm25_dense", ""),
        "rank_gap_bm25_hybrid": record.get("rank_gap_bm25_hybrid", ""),
        "rank_gap_dense_hybrid": record.get("rank_gap_dense_hybrid", ""),
    }


def _original_rank(hard_case: dict, retriever: str) -> int | None:
    retrieved = hard_case.get("original_retrieved_by_retriever", {}).get(retriever, [])
    gold_doc_id = hard_case.get("gold_doc_id")
    if gold_doc_id in retrieved:
        return retrieved.index(gold_doc_id) + 1
    return None


def _state_key(hard_case: dict, retriever: str) -> str:
    failure_type = hard_case.get("failure_type", "unlabeled")
    failed_retrievers = hard_case.get("failed_retrievers", [])
    query_length = len(tokenize(hard_case.get("question", "")))
    original_rank = _original_rank(hard_case, retriever)
    if original_rank is None:
        rank_bucket = "failed"
    elif original_rank == 1:
        rank_bucket = "rank_1"
    elif original_rank <= 5:
        rank_bucket = "rank_2_5"
    else:
        rank_bucket = "rank_6_10"
    failure_count = len(failed_retrievers)
    if query_length <= 4:
        length_bucket = "short"
    elif query_length <= 10:
        length_bucket = "medium"
    else:
        length_bucket = "long"
    return f"{failure_type}|{length_bucket}|{rank_bucket}|failed_count_{failure_count}"


def _eval_split(qid: str, train_qids: set[str], test_qids: set[str]) -> str:
    if qid in train_qids:
        return "train"
    if qid in test_qids:
        return "test"
    return "unknown"


def _summarize_policy_rows(policy_rows: list[dict]) -> list[dict]:
    subsets = {
        "all": lambda row: True,
        "offline_rl_test": lambda row: row.get("eval_split") == "test",
        "retriever_originally_failed": lambda row: row["originally_failed"],
        "retriever_failed_test": lambda row: row["originally_failed"] and row.get("eval_split") == "test",
        "original_rank_gt_5": lambda row: isinstance(row["original_rank"], int) and row["original_rank"] > 5,
    }
    summary = []
    for subset, include_row in subsets.items():
        grouped = defaultdict(list)
        for row in policy_rows:
            if include_row(row):
                grouped[(row["retriever"], row["policy_name"])].append(row)
        for (retriever, policy_name), rows in sorted(grouped.items()):
            summary.append(
                {
                    "subset": subset,
                    "retriever": retriever,
                    "policy_name": policy_name,
                    "recall@1": _mean(row["recall@1"] for row in rows),
                    "recall@5": _mean(row["recall@5"] for row in rows),
                    "recall@10": _mean(row["recall@10"] for row in rows),
                    "mrr": _mean(row["mrr"] for row in rows),
                    "answer_f1": _mean(row["answer_f1"] for row in rows),
                    "avg_reward": _mean(row["reward"] for row in rows),
                    "num_records": len(rows),
                }
            )
    return summary


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
