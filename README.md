# KorQR-RL

**KorQR-RL: Offline Reinforcement Learning for Korean Query Rewriting in RAG**

KorQR-RL treats Korean query rewriting as a one-step offline reinforcement learning problem for RAG retrieval. Given an original question and retrieval-state features, a policy selects one rewrite action and receives reward from BM25, dense, and hybrid retrieval results.

The default run now uses the real `sentence-transformers` dense backend (`intfloat/multilingual-e5-base`). For fast smoke tests, set `dense_backend: lexical` in `configs/default.yaml`.

## Research Questions

1. Which Korean QA queries become hard retrieval cases?
2. Which rewrite actions recover failed retrieval cases?
3. Does the best action depend on retriever state and failure type?
4. Can adaptive offline policies outperform fixed rewrite rules?
5. How much room remains between learned policies and oracle action selection?

## Offline RL Formulation

| RL component | Project definition |
|---|---|
| State `s` | Query length, failure type, original rank buckets, retriever-rank gaps, failed retriever count, keyword overlap, semantic similarity |
| Action `a` | `original`, `keyword`, `expanded`, `prompt`, `structured`, `llm` |
| Reward `r` | `Recall@10 + MRR + Answer F1 - semantic drift penalty - length penalty` |
| Transition | Terminal after one rewrite action |
| Policies | Fixed rules, random, epsilon-greedy, UCB, Thompson sampling, contextual bandit, offline Q-learning |

Because query rewriting is modeled as a one-step MDP, the offline Q-learning target is:

```text
Q(s, a) = E[r | s, a]
```

The `llm` action can use an external OpenAI-compatible LLM. By default it falls back to a deterministic LLM-style proxy so the pipeline remains reproducible without API access.

## Pipeline

```bash
python scripts/01_build_dataset.py
python scripts/02_run_original_retrieval.py
python scripts/03_build_hard_cases.py
python scripts/04_generate_rewrites.py
python scripts/05_evaluate_rewrites.py
python scripts/06_reward_selection.py
python scripts/07_build_analysis_tables.py
python scripts/09_sweep_hybrid_alpha.py
python scripts/10_evaluate_rewrite_policies.py
python scripts/11_build_report_artifacts.py
python scripts/12_build_results_dashboard.py
```

Optional real LLM rewrite generation:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="..."
python scripts/04_generate_rewrites.py --use-external-llm
```

Generated LLM rewrites are cached at `data/outputs/llm_rewrite_cache.jsonl` to avoid repeated API calls.

Optional dense-only baseline:

```bash
python scripts/08_run_sentence_dense_retrieval.py
```

## Outputs

- `data/outputs/original_results.csv`: Initial retrieval metrics with Recall@K, MRR, and Answer F1.
- `data/outputs/hard_cases.jsonl`: Original-query hard retrieval cases.
- `data/outputs/rewrite_candidates.jsonl`: Rewrite candidates for each hard case.
- `data/outputs/rewrite_results.jsonl`: Full reward table for every `(question, retriever, action)`.
- `data/outputs/hard_case_recovery.csv`: Reward-selected recovery rate.
- `data/outputs/main_results.csv`: Strategy-level aggregate results.
- `data/outputs/failure_type_analysis.csv`: Failure-type-level analysis.
- `data/outputs/reward_ablation_results.csv`: Reward ablation.
- `data/outputs/policy_results.csv`: Per-query policy decisions and state features.
- `data/outputs/policy_summary.csv`: Policy comparison.
- `data/outputs/final_policy_comparison.csv`: Final held-out comparison of original, rule-based, LLM, reward-selected, and RL-selected rewrites.
- `data/outputs/hybrid_alpha_sweep.csv`: Hybrid BM25/dense weight sweep.
- `results/qualitative_examples.csv`: Report-ready recovery and non-recovery examples.
- `results/failure_type_manual_check.csv`: 100-example failure label review sheet.
- `docs/results_dashboard.html`: One-page visual dashboard for the current experiment outputs.
- `docs/results_summary.md`: Compact text summary of the current experiment outputs.

Report-facing CSVs are mirrored under `results/`.

## Current Sentence-Transformers Run

KorQuAD dev: 5,774 QA pairs and 961 passages.

| retriever | original Recall@10 | original MRR | original Answer F1 |
|---|---:|---:|---:|
| BM25 | 0.9790 | 0.9072 | 0.9846 |
| dense sentence-transformers | 0.9707 | 0.8472 | 0.9803 |
| hybrid alpha=0.5 | 0.9863 | 0.9183 | 0.9905 |

The real dense run yields 265 union hard cases. The reward table contains 4,770 `(question, retriever, action)` records.

Reward-selected hard-case recovery:

| subset | BM25 | dense sentence-transformers | hybrid |
|---|---:|---:|---:|
| retriever originally failed | 0.1653 | 0.3018 | 0.3418 |
| all retrievers originally failed | 0.1818 | 0.3182 | 0.2727 |

Held-out `offline_rl_test` summary for `offline_q_learning`:

| retriever | Recall@10 | MRR | Answer F1 |
|---|---:|---:|---:|
| BM25 | 0.5750 | 0.3781 | 0.7260 |
| dense sentence-transformers | 0.4875 | 0.2881 | 0.6884 |
| hybrid | 0.7375 | 0.4340 | 0.8258 |

Hybrid alpha sweep found the strongest original-query setting at `alpha=0.1` with Recall@10 `0.9868` and MRR `0.9285`.

## Current Interpretation

- Rewriting is not universally beneficial; BM25 often performs best with original or conservative rewrites.
- Real dense retrieval is much stronger than the lexical fallback, so the hard-case set is smaller and more meaningful.
- Hybrid retrieval is strongest overall, but the best BM25/dense weight is closer to dense-heavy fusion (`alpha=0.1`) than the default `0.5`.
- Oracle performance is still meaningfully above learned policies, so better state features and learned contextual models are promising.

## 보완할 점

- Run the final policy/reward table with actual LLM-generated rewrites.
- Complete human review of the generated 100-example failure-type check sheet.
- Add KLUE-MRC or another Korean QA/RAG dataset for generalization.
- Add confidence intervals and paired significance tests.
- Train a stronger contextual policy such as LinUCB, FQI, or a small neural Q-scorer.
- Generate final figures automatically for policy comparison, reward ablation, failure-type recovery, and hybrid alpha sweep.
