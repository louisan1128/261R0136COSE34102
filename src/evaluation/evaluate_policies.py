import random
import math
import re
from collections import defaultdict
from typing import Callable

try:
    import numpy as np
except ImportError:
    np = None

from src.rewriting.policy import select_strategies
from src.utils.text import tokenize


STRATEGIES = ["original", "keyword", "expanded", "structured", "llm"]

FINAL_POLICY_LABELS = {
    "original_only": "original_query",
    "failure_type_policy": "rule_based_rewrite",
    "always_llm": "llm_rewrite",
    "label_retriever_policy": "label_retriever_policy",
    "reward_ranker_policy": "reward_ranker_policy",
    "recovery_ranker_policy": "recovery_ranker_policy",
    "calibrated_recovery_policy": "rl_selected_rewrite",
    "state_recovery_bandit_policy": "state_recovery_contextual_bandit",
    "retriever_tuned_bandit_policy": "retriever_tuned_contextual_bandit",
    "conservative_linucb_policy": "conservative_contextual_bandit",
    "observable_only_selector": "observable_only_selector",
    "refined_label_rule_model_policy": "refined_label_rule_model_policy",
    "reward_selected": "reward_selected_rewrite",
    "offline_q_learning": "q_table_rewrite",
    "oracle_best_strategy": "oracle_best_rewrite",
}


def evaluate_rewrite_policies(
    rewrite_results: list[dict],
    hard_cases: list[dict],
    original_contexts: dict[str, dict[str, list[str]]] | None = None,
    seed: int = 7,
    train_ratio: float = 0.7,
    gamma: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    records_by_key = _index_rewrite_results(rewrite_results)
    hard_case_by_qid = {record["qid"]: record for record in hard_cases}
    hard_case_by_qid = _augment_hard_cases_with_rewrite_contexts(hard_case_by_qid, records_by_key)
    if original_contexts:
        hard_case_by_qid = _augment_hard_cases_with_original_contexts(hard_case_by_qid, original_contexts)
    hard_case_by_qid = _augment_hard_cases_with_refined_labels(hard_case_by_qid)
    qid_order = _stable_qid_order(rewrite_results)
    train_qids, test_qids = _split_qids(qid_order, train_ratio, seed)

    policy_rows = []
    policy_rows.extend(_evaluate_static_policies(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_random_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids, seed=seed))
    policy_rows.extend(_evaluate_epsilon_greedy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids, seed=seed))
    policy_rows.extend(_evaluate_ucb_bandit(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_thompson_sampling(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids, seed=seed))
    policy_rows.extend(_evaluate_contextual_bandit(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_label_retriever_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_reward_ranker_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_recovery_ranker_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_calibrated_recovery_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_state_recovery_bandit_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_retriever_tuned_bandit_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_conservative_linucb_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_observable_only_selector(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
    policy_rows.extend(_evaluate_refined_label_rule_model_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids))
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


def _split_qids(qid_order: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    if not qid_order:
        return set(), set()
    train_ratio = min(max(train_ratio, 0.1), 0.9)
    shuffled_qids = list(qid_order)
    random.Random(seed).shuffle(shuffled_qids)
    split_idx = max(1, min(len(shuffled_qids) - 1, int(len(shuffled_qids) * train_ratio)))
    return set(shuffled_qids[:split_idx]), set(shuffled_qids[split_idx:])


def _augment_hard_cases_with_original_contexts(
    hard_case_by_qid: dict[str, dict],
    original_contexts: dict[str, dict[str, list[str]]],
) -> dict[str, dict]:
    augmented = {}
    for qid, hard_case in hard_case_by_qid.items():
        copied = dict(hard_case)
        copied["original_context_by_retriever"] = {
            retriever: top10
            for retriever, top10 in original_contexts.get(qid, {}).items()
        }
        augmented[qid] = copied
    return augmented


def _augment_hard_cases_with_rewrite_contexts(
    hard_case_by_qid: dict[str, dict],
    records_by_key: dict[tuple[str, str], dict[str, dict]],
) -> dict[str, dict]:
    augmented = {qid: dict(hard_case) for qid, hard_case in hard_case_by_qid.items()}
    for (qid, retriever), candidates in records_by_key.items():
        if qid not in augmented:
            continue
        original = candidates.get("original")
        if not original:
            continue
        augmented[qid].setdefault("original_top1_score_by_retriever", {})[retriever] = _safe_float(
            original.get("original_top1_score", original.get("top1_score"))
        )
        augmented[qid].setdefault("original_score_gap_by_retriever", {})[retriever] = _safe_float(
            original.get("original_score_gap_top1_top2", original.get("score_gap_top1_top2"))
        )
        augmented[qid].setdefault("original_score_ratio_by_retriever", {})[retriever] = _safe_float(
            original.get("original_score_ratio_top1_top2", original.get("score_ratio_top1_top2"))
        )
        top10 = original.get("original_top10_doc_ids") or original.get("top10_doc_ids")
        if isinstance(top10, list):
            augmented[qid].setdefault("original_scored_top10_by_retriever", {})[retriever] = [
                str(doc_id) for doc_id in top10
            ]
    return augmented


def _augment_hard_cases_with_refined_labels(hard_case_by_qid: dict[str, dict]) -> dict[str, dict]:
    augmented = {}
    for qid, hard_case in hard_case_by_qid.items():
        copied = dict(hard_case)
        copied["refined_failure_label"] = _refined_failure_label(copied)
        copied["label_rule_group"] = _label_rule_group(copied)
        augmented[qid] = copied
    return augmented


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
        "always_structured": lambda candidates, hard_case: "structured",
        "always_llm": lambda candidates, hard_case: "llm",
        "failure_type_policy": _select_failure_type_policy,
        "reward_selected": _select_reward_selected,
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

    _fit_strategy_values(records_by_key, train_qids, values, counts)

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

    _fit_strategy_values(records_by_key, train_qids, values, counts, total_counts)

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

    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            for strategy, record in candidates.items():
                if _policy_reward(record) > 0:
                    successes[retriever][strategy] += 1.0
                else:
                    failures[retriever][strategy] += 1.0

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

    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            context = _state_key(hard_case, retriever)
            context_key = (retriever, context)
            for strategy, record in candidates.items():
                reward = _policy_reward(record)
                counts[context_key][strategy] += 1
                context_step = counts[context_key][strategy]
                values[context_key][strategy] += (reward - values[context_key][strategy]) / context_step

                global_counts[retriever][strategy] += 1
                global_step = global_counts[retriever][strategy]
                global_values[retriever][strategy] += (reward - global_values[retriever][strategy]) / global_step

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            failure_signal = hard_case.get("failure_label") or hard_case.get("failure_type", "unlabeled")
            recommended = [strategy for strategy in select_strategies(failure_signal) if strategy in candidates]
            search_space = recommended or available
            context = _state_key(hard_case, retriever)
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

    return rows


def _evaluate_label_retriever_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    """Learn a retriever- and failure-label-aware rewrite selector.

    This is a contextual bandit table with deliberately simple backoff:
    (retriever, failure_label) -> (retriever, failure_type) -> retriever.
    It gives the policy a strong "keep original" option for dense retrieval
    while still letting BM25/hybrid prefer LLM or keyword rewrites when the
    training rewards support that choice.
    """

    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            for context in _label_policy_contexts(hard_case, retriever):
                for strategy, record in candidates.items():
                    counts[context][strategy] += 1
                    step = counts[context][strategy]
                    reward = _policy_reward(record)
                    values[context][strategy] += (reward - values[context][strategy]) / step

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            strategy = _select_from_context_tables(
                available,
                _label_policy_contexts(hard_case, retriever),
                values,
                counts,
            )
            rows.append(
                _policy_row(
                    "label_retriever_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_reward_ranker_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    l2: float = 1.0,
) -> list[dict]:
    """Train a small linear reward ranker and select the top predicted action."""

    if np is None:
        return _evaluate_label_retriever_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids)

    feature_names = _build_reward_ranker_feature_names(records_by_key, hard_case_by_qid, train_qids)
    x_train = []
    y_train = []
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            for strategy, record in records_by_key[(qid, retriever)].items():
                x_train.append(_reward_ranker_features(feature_names, hard_case, retriever, strategy, record))
                y_train.append(_policy_reward(record))

    if not x_train:
        return []

    x_matrix = np.asarray(x_train, dtype=float)
    y_vector = np.asarray(y_train, dtype=float)
    regularizer = l2 * np.eye(x_matrix.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    try:
        weights = np.linalg.solve(x_matrix.T @ x_matrix + regularizer, x_matrix.T @ y_vector)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x_matrix.T @ x_matrix + regularizer) @ x_matrix.T @ y_vector

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            scored = []
            for strategy, record in candidates.items():
                features = _reward_ranker_features(feature_names, hard_case, retriever, strategy, record)
                scored.append((float(np.dot(weights, features)), strategy))
            strategy = max(scored)[1] if scored else ("original" if "original" in candidates else next(iter(candidates)))
            rows.append(
                _policy_row(
                    "reward_ranker_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_recovery_ranker_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    return _evaluate_linear_ranker_policy(
        "recovery_ranker_policy",
        records_by_key,
        hard_case_by_qid,
        qid_order,
        train_qids,
        test_qids,
        target_fn=_recovery_utility,
    )


def _evaluate_calibrated_recovery_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    """Recovery ranker with retriever-specific thresholds against original."""

    if np is None:
        return _evaluate_label_retriever_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids)

    feature_names, weights = _fit_linear_ranker(
        records_by_key,
        hard_case_by_qid,
        train_qids,
        target_fn=_recovery_utility,
    )
    thresholds = _tune_retriever_thresholds(
        records_by_key,
        hard_case_by_qid,
        train_qids,
        feature_names,
        weights,
        target_fn=_recovery_utility,
    )

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            strategy = _select_calibrated_ranker_action(
                candidates,
                hard_case,
                retriever,
                feature_names,
                weights,
                thresholds.get(retriever, 0.0),
            )
            rows.append(
                _policy_row(
                    "calibrated_recovery_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_state_recovery_bandit_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    """State-aware contextual bandit tuned for hard-case recovery.

    This deliberately keeps the policy low-variance: it learns mean recovery
    utility per (retriever, state_key, action), then backs off to retriever and
    global action values when the exact state is sparse. In this dataset that
    outperforms heavier regressors because hard-case labels are informative
    and the split is small.
    """

    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            for context in _state_recovery_contexts(hard_case, retriever):
                for strategy, record in candidates.items():
                    counts[context][strategy] += 1
                    step = counts[context][strategy]
                    reward = _recovery_utility(record)
                    values[context][strategy] += (reward - values[context][strategy]) / step

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            strategy = _select_from_context_tables(
                available,
                _state_recovery_contexts(hard_case, retriever),
                values,
                counts,
            )
            rows.append(
                _policy_row(
                    "state_recovery_bandit_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_retriever_tuned_bandit_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    """Retriever-specific contextual bandit.

    BM25, dense, and hybrid respond to different rewrite risks, so this policy
    uses a small retriever-specific objective/context choice while still
    estimating all action values from the training split only.
    """

    configs = {
        "bm25": (_state_recovery_contexts, lambda record: float(record.get("mrr_improvement", 0.0))),
        "dense": (_label_policy_contexts, _policy_reward),
        "hybrid": (_label_state_recovery_contexts, _recovery_utility),
    }
    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            context_fn, target_fn = configs.get(retriever, (_state_recovery_contexts, _recovery_utility))
            candidates = records_by_key[(qid, retriever)]
            for context in context_fn(hard_case, retriever):
                for strategy, record in candidates.items():
                    counts[context][strategy] += 1
                    step = counts[context][strategy]
                    reward = target_fn(record)
                    values[context][strategy] += (reward - values[context][strategy]) / step

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            context_fn, _ = configs.get(retriever, (_state_recovery_contexts, _recovery_utility))
            candidates = records_by_key[(qid, retriever)]
            available = [strategy for strategy in STRATEGIES if strategy in candidates]
            strategy = _select_from_context_tables(
                available,
                context_fn(hard_case, retriever),
                values,
                counts,
            )
            rows.append(
                _policy_row(
                    "retriever_tuned_bandit_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_conservative_linucb_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    alpha: float = 0.15,
) -> list[dict]:
    """Conservative contextual bandit for selective query rewriting.

    Each rewrite action gets a ridge reward model conditioned on query and
    retrieval uncertainty features. At selection time, non-original actions
    must clear the original action by a tuned margin after subtracting a small
    uncertainty penalty. This keeps the RL framing but matches the one-step
    nature of selective query rewriting better than multi-step Q-learning.
    """

    if np is None:
        return _evaluate_label_retriever_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids)

    fit = _fit_linucb_models(records_by_key, hard_case_by_qid, train_qids, target_fn=_recovery_utility)
    if fit is None:
        return []
    feature_names, models = fit
    thresholds = _tune_linucb_thresholds(
        records_by_key,
        hard_case_by_qid,
        train_qids,
        feature_names,
        models,
        alpha=alpha,
        target_fn=_recovery_utility,
    )

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            strategy = _select_conservative_linucb_action(
                candidates,
                hard_case,
                retriever,
                feature_names,
                models,
                thresholds.get(retriever, 0.0),
                alpha=alpha,
            )
            rows.append(
                _policy_row(
                    "conservative_linucb_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_observable_only_selector(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    """Select rewrites using only inference-time observable signals.

    The training target may use logged rewards, but test-time features exclude
    manual labels, gold ranks, gold-passage success flags, and action outcomes.
    """

    rules = _fit_observable_only_rules(records_by_key, hard_case_by_qid, train_qids)

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            strategy = _select_observable_only_rule_action(candidates, hard_case, retriever, rules)
            rows.append(
                _policy_row(
                    "observable_only_selector",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                    state_key="observable_only",
                )
            )
    return rows


def _evaluate_refined_label_rule_model_policy(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
) -> list[dict]:
    """Safely blend refined-label action rules with a learned base policy.

    Manual failure labels are useful but noisy. This policy first maps them to a
    more stable label group, uses the retriever-tuned contextual bandit as a
    learned base policy, and only overrides that base when the train split shows
    a clear action advantage for the same (retriever, refined label group,
    original-rank bucket). This gives label-specific rules without letting noisy
    labels rewrite everything.
    """

    label_values, label_counts = _fit_refined_label_action_tables(records_by_key, hard_case_by_qid, train_qids)
    base_values, base_counts = _fit_retriever_tuned_action_tables(records_by_key, hard_case_by_qid, train_qids)

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            strategy = _select_refined_label_rule_model_action(
                candidates,
                hard_case,
                retriever,
                label_values,
                label_counts,
                base_values,
                base_counts,
            )
            rows.append(
                _policy_row(
                    "refined_label_rule_model_policy",
                    strategy,
                    candidates,
                    hard_case,
                    retriever,
                    eval_split=_eval_split(qid, train_qids, test_qids),
                )
            )
    return rows


def _evaluate_linear_ranker_policy(
    policy_name: str,
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    qid_order: list[str],
    train_qids: set[str],
    test_qids: set[str],
    target_fn: Callable[[dict], float],
    l2: float = 1.0,
) -> list[dict]:
    if np is None:
        return _evaluate_label_retriever_policy(records_by_key, hard_case_by_qid, qid_order, train_qids, test_qids)

    fit = _fit_linear_ranker(records_by_key, hard_case_by_qid, train_qids, target_fn, l2=l2)
    if fit is None:
        return []
    feature_names, weights = fit

    rows = []
    for qid in qid_order:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            scored = []
            for strategy, record in candidates.items():
                features = _reward_ranker_features(feature_names, hard_case, retriever, strategy, record)
                scored.append((float(np.dot(weights, features)), strategy))
            strategy = max(scored)[1] if scored else ("original" if "original" in candidates else next(iter(candidates)))
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


def _fit_linear_ranker(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
    target_fn: Callable[[dict], float],
    l2: float = 1.0,
) -> tuple[list[str], object] | None:
    if np is None:
        return None

    feature_names = _build_reward_ranker_feature_names(records_by_key, hard_case_by_qid, train_qids)
    x_train = []
    y_train = []
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            for strategy, record in records_by_key[(qid, retriever)].items():
                x_train.append(_reward_ranker_features(feature_names, hard_case, retriever, strategy, record))
                y_train.append(target_fn(record))

    if not x_train:
        return None

    x_matrix = np.asarray(x_train, dtype=float)
    y_vector = np.asarray(y_train, dtype=float)
    regularizer = l2 * np.eye(x_matrix.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    try:
        weights = np.linalg.solve(x_matrix.T @ x_matrix + regularizer, x_matrix.T @ y_vector)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x_matrix.T @ x_matrix + regularizer) @ x_matrix.T @ y_vector
    return feature_names, weights


def _tune_retriever_thresholds(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
    feature_names: list[str],
    weights,
    target_fn: Callable[[dict], float],
) -> dict[str, float]:
    thresholds = {}
    grid = [step / 100.0 for step in range(-30, 51)]
    retrievers = sorted({retriever for _, retriever in records_by_key})
    for retriever in retrievers:
        best_threshold = 0.0
        best_score = None
        for threshold in grid:
            selected = []
            for qid in train_qids:
                if (qid, retriever) not in records_by_key:
                    continue
                hard_case = hard_case_by_qid.get(qid, {})
                candidates = records_by_key[(qid, retriever)]
                strategy = _select_calibrated_ranker_action(
                    candidates,
                    hard_case,
                    retriever,
                    feature_names,
                    weights,
                    threshold,
                )
                selected.append(target_fn(candidates[strategy]))
            score = _mean(selected)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[retriever] = best_threshold
    return thresholds


def _select_calibrated_ranker_action(
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    feature_names: list[str],
    weights,
    threshold: float,
) -> str:
    original = "original" if "original" in candidates else next(iter(candidates))
    original_score = _predict_ranker_score(feature_names, weights, hard_case, retriever, original, candidates[original])
    best_strategy = original
    best_score = original_score
    for strategy, record in candidates.items():
        score = _predict_ranker_score(feature_names, weights, hard_case, retriever, strategy, record)
        if score > best_score:
            best_score = score
            best_strategy = strategy
    if best_strategy != original and best_score - original_score < threshold:
        return original
    return best_strategy


def _fit_linucb_models(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
    target_fn: Callable[[dict], float],
    l2: float = 1.0,
) -> tuple[list[str], dict[tuple[str, str], dict[str, object]]] | None:
    if np is None:
        return None

    feature_names = _build_context_feature_names(records_by_key, hard_case_by_qid, train_qids)
    grouped = defaultdict(list)
    targets = defaultdict(list)
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            x = _context_features(feature_names, hard_case, retriever)
            for strategy, record in records_by_key[(qid, retriever)].items():
                grouped[(retriever, strategy)].append(x)
                targets[(retriever, strategy)].append(target_fn(record))

    models = {}
    for key, rows in grouped.items():
        x_matrix = np.asarray(rows, dtype=float)
        y_vector = np.asarray(targets[key], dtype=float)
        regularizer = l2 * np.eye(x_matrix.shape[1], dtype=float)
        regularizer[0, 0] = 0.0
        a_matrix = x_matrix.T @ x_matrix + regularizer
        try:
            a_inv = np.linalg.inv(a_matrix)
        except np.linalg.LinAlgError:
            a_inv = np.linalg.pinv(a_matrix)
        theta = a_inv @ x_matrix.T @ y_vector
        models[key] = {"theta": theta, "a_inv": a_inv, "n": len(rows)}

    return (feature_names, models) if models else None


def _tune_linucb_thresholds(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
    feature_names: list[str],
    models: dict[tuple[str, str], dict[str, object]],
    alpha: float,
    target_fn: Callable[[dict], float],
) -> dict[str, float]:
    thresholds = {}
    grid = [step / 100.0 for step in range(-10, 41)]
    retrievers = sorted({retriever for _, retriever in records_by_key})
    for retriever in retrievers:
        best_threshold = 0.0
        best_score = None
        for threshold in grid:
            selected = []
            for qid in train_qids:
                if (qid, retriever) not in records_by_key:
                    continue
                hard_case = hard_case_by_qid.get(qid, {})
                candidates = records_by_key[(qid, retriever)]
                strategy = _select_conservative_linucb_action(
                    candidates,
                    hard_case,
                    retriever,
                    feature_names,
                    models,
                    threshold,
                    alpha=alpha,
                )
                selected.append(target_fn(candidates[strategy]))
            score = _mean(selected)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[retriever] = best_threshold
    return thresholds


def _select_conservative_linucb_action(
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    feature_names: list[str],
    models: dict[tuple[str, str], dict[str, object]],
    threshold: float,
    alpha: float,
) -> str:
    available = [strategy for strategy in STRATEGIES if strategy in candidates]
    original = "original" if "original" in candidates else available[0]
    x = np.asarray(_context_features(feature_names, hard_case, retriever), dtype=float)

    original_score = _linucb_score(models, retriever, original, x, alpha=0.0)
    if original_score is None:
        original_score = 0.0

    best_strategy = original
    best_score = original_score
    for strategy in available:
        score = _linucb_score(models, retriever, strategy, x, alpha=alpha if strategy != original else 0.0)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_strategy = strategy

    if best_strategy != original and best_score - original_score < threshold:
        return original
    return best_strategy


def _linucb_score(
    models: dict[tuple[str, str], dict[str, object]],
    retriever: str,
    strategy: str,
    x,
    alpha: float,
) -> float | None:
    model = models.get((retriever, strategy))
    if model is None:
        return None
    theta = model["theta"]
    a_inv = model["a_inv"]
    prediction = float(np.dot(theta, x))
    uncertainty = float(np.sqrt(max(0.0, x @ a_inv @ x)))
    return prediction - alpha * uncertainty


def _fit_observable_only_rules(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
) -> dict[str, tuple[str, str, str, float, str]]:
    rules = {}
    retrievers = sorted({retriever for _, retriever in records_by_key})
    for retriever in retrievers:
        best_rule = ("always", "original", "ge", 0.5, "original")
        best_score = None
        for feature in _observable_rule_features():
            thresholds = _observable_rule_thresholds(records_by_key, hard_case_by_qid, train_qids, retriever, feature)
            for default_strategy in STRATEGIES:
                for alternate_strategy in STRATEGIES:
                    for op in ("lt", "ge"):
                        for threshold in thresholds:
                            selected_rewards = []
                            for qid in train_qids:
                                if (qid, retriever) not in records_by_key:
                                    continue
                                hard_case = hard_case_by_qid.get(qid, {})
                                candidates = records_by_key[(qid, retriever)]
                                strategy = _apply_observable_rule(
                                    candidates,
                                    hard_case,
                                    retriever,
                                    feature,
                                    default_strategy,
                                    alternate_strategy,
                                    op,
                                    threshold,
                                )
                                selected_rewards.append(_observable_rule_objective(candidates[strategy], retriever))
                            score = _mean(selected_rewards)
                            if best_score is None or score > best_score:
                                best_score = score
                                best_rule = (feature, default_strategy, op, threshold, alternate_strategy)
        rules[retriever] = best_rule
    return rules


def _observable_rule_objective(record: dict, retriever: str) -> float:
    if retriever == "dense":
        return _policy_reward(record)
    return float(record.get("recall@10", 0.0))


def _select_observable_only_rule_action(
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    rules: dict[str, tuple],
) -> str:
    rule = rules.get(retriever)
    if not rule:
        return "original" if "original" in candidates else next(iter(candidates))
    feature, default_strategy, op, threshold, alternate_strategy = rule
    return _apply_observable_rule(
        candidates,
        hard_case,
        retriever,
        feature,
        default_strategy,
        alternate_strategy,
        op,
        threshold,
    )


def _apply_observable_rule(
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    feature: str,
    default_strategy: str,
    alternate_strategy: str,
    op: str,
    threshold: float,
) -> str:
    available = [strategy for strategy in STRATEGIES if strategy in candidates]
    default_strategy = default_strategy if default_strategy in candidates else ("original" if "original" in candidates else available[0])
    alternate_strategy = alternate_strategy if alternate_strategy in candidates else default_strategy
    feature_value = _observable_rule_value(hard_case, retriever, feature)
    use_alternate = feature_value < threshold if op == "lt" else feature_value >= threshold
    return alternate_strategy if use_alternate else default_strategy


def _observable_rule_features() -> list[str]:
    return [
        "always",
        "query_len",
        "has_digit",
        "has_latin",
        "original_top1_score",
        "original_score_gap_top1_top2",
        "original_score_ratio_top1_top2",
    ]


def _observable_rule_thresholds(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
    retriever: str,
    feature: str,
) -> list[float]:
    if feature == "always":
        return [0.5]
    values = sorted(
        {
            _observable_rule_value(hard_case_by_qid.get(qid, {}), retriever, feature)
            for qid in train_qids
            if (qid, retriever) in records_by_key
        }
    )
    if not values:
        return [0.0]
    step = max(1, len(values) // 40)
    return values[::step] + [values[-1] + 1.0]


def _observable_rule_value(hard_case: dict, retriever: str, feature: str) -> float:
    question = str(hard_case.get("question") or "")
    if feature == "always":
        return 0.0
    if feature == "query_len":
        return float(len(tokenize(question)))
    if feature == "has_digit":
        return 1.0 if any(char.isdigit() for char in question) else 0.0
    if feature == "has_latin":
        return 1.0 if any(char.isascii() and char.isalpha() for char in question) else 0.0
    if feature == "original_top1_score":
        return _safe_float(hard_case.get("original_top1_score_by_retriever", {}).get(retriever))
    if feature == "original_score_gap_top1_top2":
        return _safe_float(hard_case.get("original_score_gap_by_retriever", {}).get(retriever))
    if feature == "original_score_ratio_top1_top2":
        return _safe_float(hard_case.get("original_score_ratio_by_retriever", {}).get(retriever))
    return 0.0


def _fit_refined_label_action_tables(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
) -> tuple[dict, dict]:
    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            candidates = records_by_key[(qid, retriever)]
            contexts = _refined_label_contexts(hard_case, retriever, candidates)
            for context in contexts:
                for strategy, record in candidates.items():
                    counts[context][strategy] += 1
                    step = counts[context][strategy]
                    reward = _recovery_utility(record)
                    values[context][strategy] += (reward - values[context][strategy]) / step
    return values, counts


def _fit_retriever_tuned_action_tables(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
) -> tuple[dict, dict]:
    configs = {
        "bm25": (_state_recovery_contexts, lambda record: float(record.get("mrr_improvement", 0.0))),
        "dense": (_label_policy_contexts, _policy_reward),
        "hybrid": (_label_state_recovery_contexts, _recovery_utility),
    }
    values = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            context_fn, target_fn = configs.get(retriever, (_state_recovery_contexts, _recovery_utility))
            candidates = records_by_key[(qid, retriever)]
            for context in context_fn(hard_case, retriever):
                for strategy, record in candidates.items():
                    counts[context][strategy] += 1
                    step = counts[context][strategy]
                    reward = target_fn(record)
                    values[context][strategy] += (reward - values[context][strategy]) / step
    return values, counts


def _select_refined_label_rule_model_action(
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    label_values: dict,
    label_counts: dict,
    base_values: dict,
    base_counts: dict,
    threshold: float = 0.02,
) -> str:
    available = [strategy for strategy in STRATEGIES if strategy in candidates]
    base_strategy = _select_from_context_tables(
        available,
        _retriever_tuned_contexts(hard_case, retriever),
        base_values,
        base_counts,
    )
    table_strategy, table_score, base_score = _refined_label_rank_rule(
        available,
        candidates,
        hard_case,
        retriever,
        base_strategy,
        label_values,
        label_counts,
    )
    if table_strategy != base_strategy and table_score - base_score > threshold:
        return table_strategy
    return base_strategy


def _refined_label_rank_rule(
    available: list[str],
    candidates: dict[str, dict],
    hard_case: dict,
    retriever: str,
    base_strategy: str,
    label_values: dict,
    label_counts: dict,
) -> tuple[str, float, float]:
    label_group = str(hard_case.get("label_rule_group") or _label_rule_group(hard_case))
    rank_bucket = _rank_bucket(_candidate_original_rank(candidates))
    context = ("label_group_rank", retriever, label_group, rank_bucket)
    for min_count in (8, 5, 3, 1):
        scored = [
            (label_values[context][strategy], strategy)
            for strategy in available
            if label_counts[context][strategy] >= min_count
        ]
        if scored:
            best_score, best_strategy = max(
                scored,
                key=lambda item: (item[0], _strategy_tiebreak(item[1])),
            )
            base_score = label_values[context][base_strategy] if label_counts[context][base_strategy] else 0.0
            return best_strategy, best_score, base_score
    return base_strategy, 0.0, 0.0


def _candidate_original_rank(candidates: dict[str, dict]) -> int | None:
    original = candidates.get("original")
    if not original:
        return None
    rank = original.get("original_gold_rank")
    if rank in (None, ""):
        return None
    try:
        return int(rank)
    except (TypeError, ValueError):
        return None


def _predict_ranker_score(
    feature_names: list[str],
    weights,
    hard_case: dict,
    retriever: str,
    strategy: str,
    record: dict,
) -> float:
    features = _reward_ranker_features(feature_names, hard_case, retriever, strategy, record)
    return float(np.dot(weights, features)) if np is not None else 0.0


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
                target = _recovery_utility(record) + gamma * 0.0
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
    min_state_samples: int = 8,
    state_weight: float = 0.35,
) -> str:
    def mean_or_missing(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    scored = []
    for strategy in available:
        fallback = mean_or_missing(fallback_values.get(strategy, []))
        state_samples = state_values.get(strategy, [])
        state = mean_or_missing(state_samples)
        if fallback is None and state is None:
            continue
        if state is not None and len(state_samples) >= min_state_samples and fallback is not None:
            value = state_weight * state + (1.0 - state_weight) * fallback
        elif state is not None and len(state_samples) >= min_state_samples:
            value = state
        else:
            value = fallback if fallback is not None else state
        if value is not None:
            scored.append((value, strategy))
    if scored:
        return max(scored, key=lambda item: (item[0], _strategy_tiebreak(item[1])))[1]
    return "original" if "original" in available else available[0]


def _label_policy_contexts(hard_case: dict, retriever: str) -> list[tuple[str, ...]]:
    failure_type = str(hard_case.get("failure_type", "unlabeled") or "unlabeled")
    failure_label = str(hard_case.get("failure_label") or "").strip()
    question_type = str(hard_case.get("question_type", "") or "").strip()
    original_rank = _original_rank(hard_case, retriever)
    rank_bucket = _rank_bucket(original_rank)
    contexts = []
    if failure_label:
        contexts.append(("retriever_label_rank", retriever, failure_label, rank_bucket))
        contexts.append(("retriever_label", retriever, failure_label))
    if question_type:
        contexts.append(("retriever_question_type", retriever, question_type))
    contexts.extend(
        [
            ("retriever_type_rank", retriever, failure_type, rank_bucket),
            ("retriever_type", retriever, failure_type),
            ("retriever_rank", retriever, rank_bucket),
            ("retriever", retriever),
            ("global",),
        ]
    )
    return contexts


def _state_recovery_contexts(hard_case: dict, retriever: str) -> list[tuple[str, ...]]:
    return [
        ("state", retriever, _state_key(hard_case, retriever)),
        ("retriever", retriever),
        ("global",),
    ]


def _label_state_recovery_contexts(hard_case: dict, retriever: str) -> list[tuple[str, ...]]:
    failure_label = str(hard_case.get("failure_label") or "").strip()
    return [
        ("label_state", retriever, failure_label, _state_key(hard_case, retriever)),
        ("state", retriever, _state_key(hard_case, retriever)),
        ("label", retriever, failure_label),
        ("retriever", retriever),
        ("global",),
    ]


def _retriever_tuned_contexts(hard_case: dict, retriever: str) -> list[tuple[str, ...]]:
    if retriever == "dense":
        return _label_policy_contexts(hard_case, retriever)
    if retriever == "hybrid":
        return _label_state_recovery_contexts(hard_case, retriever)
    return _state_recovery_contexts(hard_case, retriever)


def _refined_label_contexts(
    hard_case: dict,
    retriever: str,
    candidates: dict[str, dict] | None = None,
) -> list[tuple[str, ...]]:
    refined_label = str(hard_case.get("refined_failure_label") or _refined_failure_label(hard_case))
    label_group = str(hard_case.get("label_rule_group") or _label_rule_group(hard_case))
    failure_label = str(hard_case.get("failure_label") or "unlabeled")
    question_type = str(hard_case.get("question_type") or "unknown")
    rank = _candidate_original_rank(candidates) if candidates else _original_rank(hard_case, retriever)
    rank_bucket = _rank_bucket(rank)
    failed_scope = _failed_scope(hard_case)
    return [
        ("refined_label_rank_scope", retriever, refined_label, rank_bucket, failed_scope),
        ("refined_label_rank", retriever, refined_label, rank_bucket),
        ("refined_label", retriever, refined_label),
        ("label_group_rank", retriever, label_group, rank_bucket),
        ("label_group", retriever, label_group),
        ("failure_label", retriever, failure_label),
        ("question_type", retriever, question_type),
        ("retriever_rank", retriever, rank_bucket),
        ("retriever", retriever),
        ("global",),
    ]


def _select_from_context_tables(
    available: list[str],
    contexts: list[tuple[str, ...]],
    values: dict,
    counts: dict,
) -> str:
    for context in contexts:
        if any(counts[context][strategy] for strategy in available):
            return max(available, key=lambda strategy: (values[context][strategy], _strategy_tiebreak(strategy)))
    return "original" if "original" in available else available[0]


def _strategy_tiebreak(strategy: str) -> int:
    preference = {
        "original": 5,
        "llm": 4,
        "keyword": 3,
        "structured": 2,
        "expanded": 1,
    }
    return preference.get(strategy, -1)


def _build_reward_ranker_feature_names(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
) -> list[str]:
    names = {
        "bias",
        "original_success",
        "original_rank_missing",
        "original_rank_inverse",
        "original_rank_gt_5",
        "originally_failed",
        "failed_retriever_count",
        "question_len",
        "question_len_short",
        "question_len_long",
        "has_digit",
        "has_latin",
        "rewrite_len",
        "rewrite_len_delta",
        "keyword_overlap",
        "semantic_similarity",
        "top1_score",
        "score_gap_top1_top2",
        "score_ratio_top1_top2",
        "original_top1_score",
        "original_score_gap_top1_top2",
        "original_score_ratio_top1_top2",
    }
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            for strategy, record in records_by_key[(qid, retriever)].items():
                names.update(_categorical_feature_names(hard_case, retriever, strategy, record))
    return ["bias", *sorted(name for name in names if name != "bias")]


def _reward_ranker_features(
    feature_names: list[str],
    hard_case: dict,
    retriever: str,
    strategy: str,
    record: dict,
) -> list[float]:
    values = defaultdict(float)
    values["bias"] = 1.0

    original_rank = _original_rank(hard_case, retriever)
    values["original_success"] = 1.0 if original_rank else 0.0
    values["original_rank_missing"] = 1.0 if original_rank is None else 0.0
    values["original_rank_inverse"] = 0.0 if original_rank is None else 1.0 / original_rank
    values["original_rank_gt_5"] = 1.0 if original_rank is not None and original_rank > 5 else 0.0
    values["originally_failed"] = 1.0 if retriever in _failed_retrievers(hard_case) else 0.0
    values["failed_retriever_count"] = float(len(_failed_retrievers(hard_case)))

    question = str(hard_case.get("question") or record.get("question") or "")
    question_tokens = tokenize(question)
    question_len = len(question_tokens)
    values["question_len"] = min(question_len, 30) / 30.0
    values["question_len_short"] = 1.0 if question_len <= 4 else 0.0
    values["question_len_long"] = 1.0 if question_len > 10 else 0.0
    values["has_digit"] = 1.0 if any(char.isdigit() for char in question) else 0.0
    values["has_latin"] = 1.0 if any(char.isascii() and char.isalpha() for char in question) else 0.0

    original_len = _safe_float(record.get("original_query_length"))
    rewrite_len = _safe_float(record.get("rewrite_query_length"))
    values["rewrite_len"] = min(rewrite_len, 40.0) / 40.0
    values["rewrite_len_delta"] = max(min(rewrite_len - original_len, 30.0), -30.0) / 30.0
    values["keyword_overlap"] = _safe_float(record.get("keyword_overlap"))
    values["semantic_similarity"] = _safe_float(record.get("semantic_similarity"))
    values["original_top1_score"] = _scaled_score(record.get("original_top1_score"))
    values["original_score_gap_top1_top2"] = _scaled_score(record.get("original_score_gap_top1_top2"))
    values["original_score_ratio_top1_top2"] = min(_safe_float(record.get("original_score_ratio_top1_top2")), 10.0) / 10.0
    values["top1_score"] = _scaled_score(record.get("top1_score"))
    values["score_gap_top1_top2"] = _scaled_score(record.get("score_gap_top1_top2"))
    values["score_ratio_top1_top2"] = min(_safe_float(record.get("score_ratio_top1_top2")), 10.0) / 10.0

    for name in _categorical_feature_names(hard_case, retriever, strategy, record):
        values[name] = 1.0

    return [values[name] for name in feature_names]


def _build_context_feature_names(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    hard_case_by_qid: dict[str, dict],
    train_qids: set[str],
) -> list[str]:
    names = {
        "bias",
        "original_success",
        "original_rank_missing",
        "original_rank_inverse",
        "original_rank_gt_5",
        "originally_failed",
        "failed_retriever_count",
        "question_len",
        "question_len_short",
        "question_len_long",
        "has_digit",
        "has_latin",
        "has_long_token",
        "bm25_initial_rank_missing",
        "dense_initial_rank_missing",
        "hybrid_initial_rank_missing",
        "bm25_initial_rank_inverse",
        "dense_initial_rank_inverse",
        "hybrid_initial_rank_inverse",
        "rank_gap_bm25_dense",
        "rank_gap_bm25_hybrid",
        "rank_gap_dense_hybrid",
        "bm25_dense_overlap",
        "bm25_hybrid_overlap",
        "dense_hybrid_overlap",
        "retriever_consensus_overlap",
        "original_top1_score",
        "original_score_gap_top1_top2",
        "original_score_ratio_top1_top2",
    }
    for qid in train_qids:
        hard_case = hard_case_by_qid.get(qid, {})
        for retriever in _retrievers_for_qid(records_by_key, qid):
            names.update(_context_categorical_feature_names(hard_case, retriever))
    return ["bias", *sorted(name for name in names if name != "bias")]


def _context_features(feature_names: list[str], hard_case: dict, retriever: str) -> list[float]:
    values = defaultdict(float)
    values["bias"] = 1.0

    original_rank = _original_rank(hard_case, retriever)
    values["original_success"] = 1.0 if original_rank else 0.0
    values["original_rank_missing"] = 1.0 if original_rank is None else 0.0
    values["original_rank_inverse"] = 0.0 if original_rank is None else 1.0 / original_rank
    values["original_rank_gt_5"] = 1.0 if original_rank is not None and original_rank > 5 else 0.0
    values["originally_failed"] = 1.0 if retriever in _failed_retrievers(hard_case) else 0.0
    values["failed_retriever_count"] = float(len(_failed_retrievers(hard_case)))

    question = str(hard_case.get("question") or "")
    question_tokens = tokenize(question)
    question_len = len(question_tokens)
    values["question_len"] = min(question_len, 30) / 30.0
    values["question_len_short"] = 1.0 if question_len <= 4 else 0.0
    values["question_len_long"] = 1.0 if question_len > 10 else 0.0
    values["has_digit"] = 1.0 if any(char.isdigit() for char in question) else 0.0
    values["has_latin"] = 1.0 if any(char.isascii() and char.isalpha() for char in question) else 0.0
    values["has_long_token"] = 1.0 if any(len(token) >= 7 for token in question_tokens) else 0.0

    initial_ranks = _initial_rank_features(hard_case)
    for name, rank in initial_ranks.items():
        values[f"{name}_missing"] = 1.0 if rank is None else 0.0
        values[f"{name}_inverse"] = 0.0 if rank is None else 1.0 / rank
    values["rank_gap_bm25_dense"] = _rank_gap(initial_ranks["bm25_initial_rank"], initial_ranks["dense_initial_rank"])
    values["rank_gap_bm25_hybrid"] = _rank_gap(initial_ranks["bm25_initial_rank"], initial_ranks["hybrid_initial_rank"])
    values["rank_gap_dense_hybrid"] = _rank_gap(initial_ranks["dense_initial_rank"], initial_ranks["hybrid_initial_rank"])
    overlap_features = _retriever_overlap_features(hard_case)
    values["bm25_dense_overlap"] = overlap_features["bm25_dense_overlap"]
    values["bm25_hybrid_overlap"] = overlap_features["bm25_hybrid_overlap"]
    values["dense_hybrid_overlap"] = overlap_features["dense_hybrid_overlap"]
    values["retriever_consensus_overlap"] = overlap_features["retriever_consensus_overlap"]
    values["original_top1_score"] = _scaled_score(
        hard_case.get("original_top1_score_by_retriever", {}).get(retriever)
    )
    values["original_score_gap_top1_top2"] = _scaled_score(
        hard_case.get("original_score_gap_by_retriever", {}).get(retriever)
    )
    values["original_score_ratio_top1_top2"] = min(
        _safe_float(hard_case.get("original_score_ratio_by_retriever", {}).get(retriever)),
        10.0,
    ) / 10.0

    for name in _context_categorical_feature_names(hard_case, retriever):
        values[name] = 1.0

    return [values[name] for name in feature_names]


def _initial_rank_features(hard_case: dict) -> dict[str, int | None]:
    return {
        "bm25_initial_rank": _original_rank(hard_case, "bm25"),
        "dense_initial_rank": _original_rank(hard_case, "dense"),
        "hybrid_initial_rank": _original_rank(hard_case, "hybrid"),
    }


def _rank_gap(left: int | None, right: int | None) -> float:
    left_value = 11 if left is None else left
    right_value = 11 if right is None else right
    return max(min(float(left_value - right_value), 10.0), -10.0) / 10.0


def _scaled_score(value) -> float:
    score = _safe_float(value)
    if score <= 0.0:
        return 0.0
    return min(score, 100.0) / 100.0


def _retriever_overlap_features(hard_case: dict) -> dict[str, float]:
    retrieved = (
        hard_case.get("original_scored_top10_by_retriever")
        or hard_case.get("original_context_by_retriever")
        or hard_case.get("original_retrieved_by_retriever", {})
    )
    bm25 = set(retrieved.get("bm25", []))
    dense = set(retrieved.get("dense", []))
    hybrid = set(retrieved.get("hybrid", []))
    bm25_dense = _topk_overlap(bm25, dense)
    bm25_hybrid = _topk_overlap(bm25, hybrid)
    dense_hybrid = _topk_overlap(dense, hybrid)
    present = [items for items in (bm25, dense, hybrid) if items]
    if len(present) < 2:
        consensus = 0.0
    else:
        consensus = (bm25_dense + bm25_hybrid + dense_hybrid) / 3.0
    return {
        "bm25_dense_overlap": bm25_dense,
        "bm25_hybrid_overlap": bm25_hybrid,
        "dense_hybrid_overlap": dense_hybrid,
        "retriever_consensus_overlap": consensus,
    }


def _topk_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _context_categorical_feature_names(hard_case: dict, retriever: str) -> set[str]:
    failure_type = str(hard_case.get("failure_type", "unlabeled") or "unlabeled")
    failure_label = str(hard_case.get("failure_label", "") or "unlabeled")
    refined_label = str(hard_case.get("refined_failure_label", "") or _refined_failure_label(hard_case))
    label_group = str(hard_case.get("label_rule_group", "") or _label_rule_group(hard_case))
    secondary_label = str(hard_case.get("secondary_failure_label", "") or "none")
    question_type = str(hard_case.get("question_type", "") or "unknown")
    rank_bucket = _rank_bucket(_original_rank(hard_case, retriever))
    failed_scope = "failed_" + "_".join(sorted(_failed_retrievers(hard_case))) if _failed_retrievers(hard_case) else "no_failure"
    overlap = _retriever_overlap_features(hard_case)
    bm25_dense_overlap_bucket = _overlap_bucket(overlap["bm25_dense_overlap"])
    consensus_overlap_bucket = _overlap_bucket(overlap["retriever_consensus_overlap"])
    return {
        f"retriever={retriever}",
        f"failure_type={failure_type}",
        f"failure_label={failure_label}",
        f"refined_failure_label={refined_label}",
        f"label_rule_group={label_group}",
        f"secondary_failure_label={secondary_label}",
        f"question_type={question_type}",
        f"rank_bucket={rank_bucket}",
        f"failed_scope={failed_scope}",
        f"retriever_label={retriever}:{failure_label}",
        f"retriever_refined_label={retriever}:{refined_label}",
        f"retriever_label_group={retriever}:{label_group}",
        f"retriever_type={retriever}:{failure_type}",
        f"retriever_rank={retriever}:{rank_bucket}",
        f"bm25_dense_overlap={bm25_dense_overlap_bucket}",
        f"consensus_overlap={consensus_overlap_bucket}",
        f"retriever_consensus_overlap={retriever}:{consensus_overlap_bucket}",
    }


def _overlap_bucket(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value < 0.2:
        return "low"
    if value < 0.5:
        return "medium"
    return "high"


def _categorical_feature_names(hard_case: dict, retriever: str, strategy: str, record: dict) -> set[str]:
    failure_type = str(hard_case.get("failure_type", record.get("failure_type", "unlabeled")) or "unlabeled")
    failure_label = str(hard_case.get("failure_label", record.get("failure_label", "")) or "unlabeled")
    refined_label = str(hard_case.get("refined_failure_label", "") or _refined_failure_label(hard_case))
    label_group = str(hard_case.get("label_rule_group", "") or _label_rule_group(hard_case))
    secondary_label = str(hard_case.get("secondary_failure_label", record.get("secondary_failure_label", "")) or "none")
    question_type = str(hard_case.get("question_type", record.get("question_type", "")) or "unknown")
    rank_bucket = _rank_bucket(_original_rank(hard_case, retriever))
    return {
        f"retriever={retriever}",
        f"strategy={strategy}",
        f"failure_type={failure_type}",
        f"failure_label={failure_label}",
        f"refined_failure_label={refined_label}",
        f"label_rule_group={label_group}",
        f"secondary_failure_label={secondary_label}",
        f"question_type={question_type}",
        f"rank_bucket={rank_bucket}",
        f"retriever_strategy={retriever}:{strategy}",
        f"retriever_label={retriever}:{failure_label}",
        f"retriever_refined_label={retriever}:{refined_label}",
        f"retriever_label_group={retriever}:{label_group}",
        f"label_strategy={failure_label}:{strategy}",
        f"refined_label_strategy={refined_label}:{strategy}",
        f"label_group_strategy={label_group}:{strategy}",
        f"retriever_label_strategy={retriever}:{failure_label}:{strategy}",
        f"retriever_refined_label_strategy={retriever}:{refined_label}:{strategy}",
        f"retriever_label_group_strategy={retriever}:{label_group}:{strategy}",
        f"retriever_rank_strategy={retriever}:{rank_bucket}:{strategy}",
    }


def _rank_bucket(original_rank: int | None) -> str:
    if original_rank is None:
        return "failed"
    if original_rank == 1:
        return "rank_1"
    if original_rank <= 5:
        return "rank_2_5"
    return "rank_6_10"


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _refined_failure_label(hard_case: dict) -> str:
    label = str(hard_case.get("failure_label") or hard_case.get("failure_type") or "unlabeled")
    question = str(hard_case.get("question") or "")
    question_type = str(hard_case.get("question_type") or "")
    secondary = str(hard_case.get("secondary_failure_label") or "")
    failed_count = len(_failed_retrievers(hard_case))

    numeric_intent = _has_numeric_intent(question, question_type)
    if numeric_intent and label in {"lexical_mismatch", "missing_key_term", "entity_mismatch", "semantic_mismatch"}:
        label = "numeric_temporal_mismatch"
    elif label == "semantic_mismatch":
        label = "lexical_mismatch"
    elif label == "ambiguous" and len(tokenize(question)) <= 4:
        label = "missing_key_term"

    suffixes = []
    if secondary and secondary not in {"none", "unlabeled", label}:
        suffixes.append(secondary)
    elif _has_latin(question) and label in {"entity_mismatch", "lexical_mismatch"}:
        suffixes.append("latin_surface")
    elif failed_count >= 2 and label in {"missing_key_term", "context_boundary_issue"}:
        suffixes.append("multi_retriever_failure")
    elif len(tokenize(question)) <= 4 and label in {"missing_key_term", "ambiguous"}:
        suffixes.append("short_query")

    return "+".join([label, *suffixes[:1]]) if suffixes else label


def _label_rule_group(hard_case: dict) -> str:
    refined = _refined_failure_label(hard_case)
    return refined.split("+", 1)[0]


def _has_numeric_intent(question: str, question_type: str) -> bool:
    return question_type in {"numeric", "when"} or bool(
        re.search(r"\d|몇|언제|연도|년도|날짜|시기|몇 년|몇 월|몇 일|순위|몇 위", question)
    )


def _has_latin(question: str) -> bool:
    return bool(re.search(r"[A-Za-z]", question))


def _failed_scope(hard_case: dict) -> str:
    failed = _failed_retrievers(hard_case)
    if not failed:
        return "no_failure"
    if len(failed) >= 3:
        return "failed_all"
    if len(failed) == 2:
        return "failed_pair"
    return f"failed_{failed[0]}"


def _fit_strategy_values(
    records_by_key: dict[tuple[str, str], dict[str, dict]],
    train_qids: set[str],
    values: dict,
    counts: dict,
    total_counts: dict | None = None,
) -> None:
    for qid in train_qids:
        for retriever in _retrievers_for_qid(records_by_key, qid):
            for strategy, record in records_by_key[(qid, retriever)].items():
                counts[retriever][strategy] += 1
                if total_counts is not None:
                    total_counts[retriever] += 1
                step = counts[retriever][strategy]
                reward = _policy_reward(record)
                values[retriever][strategy] += (reward - values[retriever][strategy]) / step


def _select_failure_type_policy(candidates: dict[str, dict], hard_case: dict) -> str:
    failure_signal = hard_case.get("failure_label") or hard_case.get("failure_type", "unlabeled")
    recommended = [strategy for strategy in select_strategies(failure_signal) if strategy in candidates]
    if not recommended:
        recommended = [strategy for strategy in STRATEGIES if strategy in candidates]
    return recommended[0]


def _select_reward_selected(candidates: dict[str, dict], hard_case: dict) -> str:
    return max(candidates, key=lambda strategy: _policy_reward(candidates[strategy]))


def _select_oracle(candidates: dict[str, dict], hard_case: dict) -> str:
    return _select_reward_selected(candidates, hard_case)


def _policy_reward(record: dict) -> float:
    return float(record.get("reward_improvement", record.get("reward", 0.0)))


def _recovery_utility(record: dict) -> float:
    return (
        float(record.get("recall10_improvement", 0.0))
        + 0.25 * float(record.get("mrr_improvement", 0.0))
        + 0.10 * float(record.get("answer_f1_improvement", 0.0))
        + 0.05 * float(record.get("reward_improvement", 0.0))
    )


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
    overlap_features = _retriever_overlap_features(hard_case)
    return {
        "qid": record["qid"],
        "retriever": retriever,
        "policy_name": policy_name,
        "eval_split": eval_split,
        "state_key": state_key,
        "selected_strategy": strategy,
        "failure_type": hard_case.get("failure_type", record.get("failure_type", "unlabeled")),
        "failure_label": hard_case.get("failure_label", record.get("failure_label", "")),
        "refined_failure_label": hard_case.get("refined_failure_label", ""),
        "label_rule_group": hard_case.get("label_rule_group", ""),
        "secondary_failure_label": hard_case.get(
            "secondary_failure_label",
            record.get("secondary_failure_label", ""),
        ),
        "original_rank": original_rank or "",
        "original_success": bool(original_rank),
        "originally_failed": retriever in _failed_retrievers(hard_case),
        "gold_rank": record.get("gold_rank", ""),
        "reward": float(record["reward"]),
        "policy_reward": _policy_reward(record),
        "recall@1": float(record["recall@1"]),
        "recall@5": float(record["recall@5"]),
        "recall@10": float(record["recall@10"]),
        "mrr": float(record["mrr"]),
        "answer_f1": float(record.get("answer_f1", 0.0)),
        "original_gold_rank": record.get("original_gold_rank", ""),
        "original_recall@10": float(record.get("original_recall@10", 0.0)),
        "original_mrr": float(record.get("original_mrr", 0.0)),
        "original_answer_f1": float(record.get("original_answer_f1", 0.0)),
        "original_reward": float(record.get("original_reward", 0.0)),
        "rank_improvement": int(record.get("rank_improvement", 0) or 0),
        "recall10_improvement": float(record.get("recall10_improvement", 0.0)),
        "mrr_improvement": float(record.get("mrr_improvement", 0.0)),
        "answer_f1_improvement": float(record.get("answer_f1_improvement", 0.0)),
        "reward_improvement": float(record.get("reward_improvement", 0.0)),
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
        "bm25_dense_overlap": overlap_features["bm25_dense_overlap"],
        "bm25_hybrid_overlap": overlap_features["bm25_hybrid_overlap"],
        "dense_hybrid_overlap": overlap_features["dense_hybrid_overlap"],
        "retriever_consensus_overlap": overlap_features["retriever_consensus_overlap"],
        "top1_score": record.get("top1_score", ""),
        "top2_score": record.get("top2_score", ""),
        "score_gap_top1_top2": record.get("score_gap_top1_top2", ""),
        "score_ratio_top1_top2": record.get("score_ratio_top1_top2", ""),
        "original_top1_score": record.get("original_top1_score", ""),
        "original_top2_score": record.get("original_top2_score", ""),
        "original_score_gap_top1_top2": record.get("original_score_gap_top1_top2", ""),
        "original_score_ratio_top1_top2": record.get("original_score_ratio_top1_top2", ""),
    }


def _original_rank(hard_case: dict, retriever: str) -> int | None:
    if hard_case.get("retriever") == retriever:
        gold_rank = hard_case.get("gold_rank")
        if gold_rank not in (None, ""):
            return int(gold_rank)

    retrieved = hard_case.get("original_retrieved_by_retriever", {}).get(retriever, [])
    gold_doc_id = hard_case.get("gold_doc_id")
    if gold_doc_id in retrieved:
        return retrieved.index(gold_doc_id) + 1
    return None


def _state_key(hard_case: dict, retriever: str) -> str:
    failure_type = hard_case.get("failure_type", "unlabeled")
    failure_label = hard_case.get("failure_label") or failure_type
    failed_retrievers = _failed_retrievers(hard_case)
    question = hard_case.get("question", "")
    query_tokens = tokenize(question)
    query_length = len(query_tokens)
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
    failed_scope = "failed_" + "_".join(sorted(failed_retrievers)) if failed_retrievers else "no_failure"
    digit_feature = "has_digit" if any(char.isdigit() for char in question) else "no_digit"
    latin_feature = (
        "has_latin"
        if any(char.isascii() and char.isalpha() for char in question)
        else "no_latin"
    )
    long_token_feature = "has_long_token" if any(len(token) >= 7 for token in query_tokens) else "no_long_token"
    return (
        f"{failure_label}|{failure_type}|{length_bucket}|{rank_bucket}|failed_count_{failure_count}|"
        f"{failed_scope}|{digit_feature}|{latin_feature}|{long_token_feature}"
    )


def _failed_retrievers(hard_case: dict) -> list[str]:
    failed = hard_case.get("failed_retrievers")
    if isinstance(failed, list):
        return [str(item) for item in failed if str(item).strip()]
    source_retriever = str(hard_case.get("retriever", "")).strip()
    return [source_retriever] if source_retriever else []


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
                    "avg_policy_reward": _mean(row["policy_reward"] for row in rows),
                    "avg_rank_improvement": _mean(row["rank_improvement"] for row in rows),
                    "recall10_improvement": _mean(row["recall10_improvement"] for row in rows),
                    "mrr_improvement": _mean(row["mrr_improvement"] for row in rows),
                    "answer_f1_improvement": _mean(row["answer_f1_improvement"] for row in rows),
                    "reward_improvement": _mean(row["reward_improvement"] for row in rows),
                    "num_records": len(rows),
                }
            )
    return summary


def build_final_policy_comparison(policy_rows: list[dict], eval_split: str = "test") -> list[dict]:
    """Build the report-facing comparison for items 13/14.

    The learned policy is evaluated on the held-out split, so the fixed
    baselines are also filtered to that same split for an apples-to-apples
    final table.
    """

    rows = []
    grouped = defaultdict(list)
    for row in policy_rows:
        if row["policy_name"] not in FINAL_POLICY_LABELS:
            continue
        if row.get("eval_split") != eval_split:
            continue
        grouped[(row["retriever"], row["policy_name"])].append(row)

    for (retriever, policy_name), items in sorted(grouped.items()):
        rows.append(
            {
                "eval_split": eval_split,
                "retriever": retriever,
                "comparison_method": FINAL_POLICY_LABELS[policy_name],
                "policy_name": policy_name,
                "recall@1": _mean(row["recall@1"] for row in items),
                "recall@5": _mean(row["recall@5"] for row in items),
                "recall@10": _mean(row["recall@10"] for row in items),
                "mrr": _mean(row["mrr"] for row in items),
                "answer_f1": _mean(row["answer_f1"] for row in items),
                "avg_reward": _mean(row["reward"] for row in items),
                "avg_policy_reward": _mean(row["policy_reward"] for row in items),
                "avg_rank_improvement": _mean(row["rank_improvement"] for row in items),
                "recall10_improvement": _mean(row["recall10_improvement"] for row in items),
                "mrr_improvement": _mean(row["mrr_improvement"] for row in items),
                "answer_f1_improvement": _mean(row["answer_f1_improvement"] for row in items),
                "reward_improvement": _mean(row["reward_improvement"] for row in items),
                "num_records": len(items),
            }
        )
    return rows


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
