# KorQR-RL Report Outline

## Title

KorQR-RL: Offline Reinforcement Learning for Korean Query Rewriting in RAG

## Abstract

Describe Korean RAG hard retrieval cases, the one-step offline RL formulation, the rewrite action space, the reward table, and the key result that adaptive action selection is useful but still below oracle performance.

## 1. Dataset Preparation

- Dataset: KorQuAD dev by default; KLUE-MRC is planned for external validation.
- Build passages, questions, answer spans, and gold passage ids.
- Run original-query retrieval and collect hard cases where the gold passage is absent from top-10.

## 2. Initial Retrieval

- Evaluate original questions with BM25, dense, and hybrid retrieval.
- Metrics: Recall@1, Recall@5, Recall@10, MRR, and Answer F1.
- Dense uses `sentence_transformers` by default for final experiments.
- The lexical backend remains available only as a fast smoke-test fallback.
- Current original-query result: BM25 Recall@10 `0.9790`, dense Recall@10 `0.9707`, hybrid Recall@10 `0.9863`.

## 3. Failure Type Analysis

- Assign heuristic failure labels:
  - `expression_mismatch`
  - `ellipsis`
  - `compound_noun`
  - `colloquial_mismatch`
  - `abbreviation`
  - `temporal_numeric`
  - `unlabeled`
- Discuss limitations of heuristic labels and the need for manual annotation.
- Use `reports/failure_type_manual_check.csv` as a 100-example review sheet for partial human validation.

## 4. Rewrite Candidate Generation

- `original`: no rewrite
- `keyword`: core keyword extraction
- `expanded`: keyword plus synonym/domain expansion
- `prompt`: search-intent style rewrite
- `structured`: field-like query with question and target information
- `llm`: external LLM-generated rewrite, with deterministic fallback for reproducible runs

## 5. Reward Table Construction

- For every hard case, evaluate every rewrite action with every retriever.
- Store `(state, action, retriever, metrics, reward)` in `rewrite_results.jsonl`.
- Current real-dense run: 265 union hard cases and 4,770 reward-table rows.
- Reward:

```text
reward = alpha * Recall@10
       + beta * MRR
       + answer_gamma * Answer F1
       - lambda * length_penalty
       - drift_gamma * semantic_drift
```

## 6. State Feature Construction

Current logged features:

- Original question
- Original query length
- Rewrite query length
- Keyword overlap
- Semantic similarity between original and rewrite
- BM25 initial rank
- Dense initial rank
- Hybrid initial rank
- Retriever rank gaps
- Failed retriever count
- Failure pattern

The tabular offline Q policy compresses these into a coarse `state_key`; richer learned policies can use the full feature table.

## 7. Policy Learning

Policies currently compared:

- `original_only`
- `always_keyword`
- `always_expanded`
- `always_prompt`
- `always_structured`
- `always_llm`
- `random_policy`
- `failure_type_policy`
- `epsilon_greedy_bandit`
- `ucb_bandit`
- `thompson_sampling`
- `contextual_bandit`
- `offline_q_learning`
- `oracle_best_strategy`

Learning objective: choose the rewrite action with the highest expected reward for the observed retrieval state.

## 8. Baseline Comparison

Compare:

- Original query only
- Random policy
- Fixed rewrite policies
- Failure-type rule policy
- Bandit policies
- Offline Q-learning
- Oracle best action

## 9. Final Evaluation

Report:

- Overall Recall@10, MRR, Answer F1, and reward
- Hard-case recovery rate
- BM25, dense, and hybrid policy differences
- Failure-type-specific action effectiveness
- Reward ablation
- Hybrid alpha sweep
- Oracle gap analysis
- Qualitative examples from `reports/qualitative_examples.csv`
- Failure-type manual-check summary from `reports/failure_type_manual_check.csv`

## 10. Conclusion

Expected conclusion:

- Rewriting every question is not always beneficial.
- The best action depends on query state and retriever behavior.
- Adaptive query rewriting policies can recover Korean RAG hard retrieval cases.
- Stronger state features, neural dense retrieval, broader LLM rewrite evaluation, and learned contextual policies are the next steps.
