# RL Policy Development Direction

## Problem framing

Selective query rewriting is closer to a contextual bandit problem than a multi-step RL problem. Each query is observed once, the policy chooses one action, and the retrieval reward is observed immediately.

Recommended formulation:

> We formulate selective query rewriting as a contextual decision-making problem, where the policy selects a rewrite action conditioned on retrieval uncertainty and query characteristics.

## Core claim

General queries are already well served by the original query, so unconditional rewriting can introduce unnecessary semantic drift. Hard cases, however, show that some rewrite actions can recover retrieval failures. The role of the policy is therefore not to generate rewrites unconditionally, but to choose whether and how to rewrite.

Action space:

- `original`
- `keyword`
- `expanded`
- `structured`
- `llm`

## Current diagnosis

The current learned policies do not fully close the gap to oracle action selection. This does not invalidate the project claim. It indicates that the main bottleneck is action selection, not rewrite generation.

Most important evidence:

- BM25 hard-case test: oracle/reward-selected improves recall@10 from `0.1400` to `0.2300`.
- Hybrid hard-case test: oracle/reward-selected improves recall@10 from `0.1467` to `0.3133`.
- Hybrid has the largest oracle reward improvement among retrievers.

## New experimental policy

The code now includes three stronger one-step policy baselines:

- `state_recovery_bandit_policy`: learns recovery utility by retriever and failure state.
- `retriever_tuned_bandit_policy`: uses retriever-specific contexts/objectives.
- `conservative_linucb_policy`: learns a separate ridge reward model for each retriever/action pair and penalizes uncertain rewrite actions.
- `refined_label_rule_model_policy`: uses the retriever-tuned policy as a base and applies refined-label overrides only when the train split shows a clear advantage.

This keeps the reinforcement-learning framing while matching the one-step structure of the task.

## Current result after richer state/features

The strongest setting found so far uses hybrid retrieval with `alpha=0.1`.

The implementation now logs and uses richer policy features:

- original retrieval confidence: top-1/top-2 score, score gap, score ratio
- rewrite retrieval confidence: top-1/top-2 score, score gap, score ratio
- retriever agreement: BM25/dense/hybrid top-k overlap
- manual labels: failure label, secondary failure label, question type, failed retriever scope

One important finding is that score/overlap buckets should not be inserted directly into the sparse table-bandit state key. That increased state fragmentation and hurt held-out performance. They are more useful as numeric/context features for ranker/LinUCB policies and as analysis columns.

The latest policy, `refined_label_rule_model_policy`, adds three changes:

- Refines noisy `failure_label` values into `refined_failure_label` and broader `label_rule_group`.
- Uses the retriever-tuned contextual bandit as the learned base policy.
- Overrides the learned base only when the train split shows a clear action advantage for the same `(retriever, label_rule_group, original-rank bucket)`.

This effectively mixes manual failure-type evidence, more trainable action cases across retriever/action views, and a learned policy while avoiding unconditional label-based rewriting.

On all hybrid retriever-originally-failed hard cases:

- `original_only` recall@10: `0.131`
- `always_llm` recall@10: `0.184`
- `calibrated_recovery_policy` recall@10: `0.204`
- `state_recovery_bandit_policy` recall@10: `0.218`
- `retriever_tuned_bandit_policy` recall@10: `0.219`
- `conservative_linucb_policy` recall@10: `0.229`
- `refined_label_rule_model_policy` recall@10: `0.236`
- `oracle_best_strategy` recall@10: `0.325`

On held-out hybrid hard-case test:

- `original_only` recall@10: `0.1467`
- `always_llm` recall@10: `0.1867`
- `calibrated_recovery_policy` recall@10: `0.1767`
- `state_recovery_bandit_policy` recall@10: `0.2067`
- `retriever_tuned_bandit_policy` recall@10: `0.2100`
- `conservative_linucb_policy` recall@10: `0.1933`
- `refined_label_rule_model_policy` recall@10: `0.2400`
- `oracle_best_strategy` recall@10: `0.3133`

Interpretation: the contextual bandit direction is appropriate. The strongest held-out hybrid policy is currently `refined_label_rule_model_policy`, a safe rule/model hybrid that uses refined manual labels only as conditional overrides over a learned contextual bandit. The oracle gap still shows substantial headroom.

## Next experiment priority

1. Regenerate `scripts/14_report_diagnostics.py` outputs before final reporting.
2. Use hybrid retrieval as the main target, especially low-confidence original queries.
3. Train on a mixed dataset: general original-success cases, hard recovered cases, and rewrite-harm cases.
4. Improve manual failure label quality and consistency. The current best table policies rely heavily on these labels, so noisy labels directly limit action selection.
5. Tune the policy objective for conservative improvement over original, not raw rewrite reward.
6. Report oracle selection as an upper bound, not as deployable policy performance.

## Added robustness checks

`scripts/14_report_diagnostics.py` now generates the supporting analysis needed
for a cleaner report:

- multi-seed held-out hard-case policy stability
- policy action distribution overall and by failure-label group
- rewrite recovery and rewrite harm rates
- oracle gap decomposition
- annotation label quality counts
- LLM rewrite/cache audit
- encoding diagnostics for the main generated artifacts

## Report wording

> The substantial gap between the learned policy and oracle action selection indicates considerable headroom for improved reinforcement learning-based decision policies. In particular, hybrid retrieval exhibits the largest oracle gain on hard cases, suggesting that the primary challenge is not generating potentially useful rewrites, but learning when and which rewrite to apply.
