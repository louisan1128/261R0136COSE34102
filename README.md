# KorQR-RL

**Offline Reinforcement Learning for Korean Query Rewriting in RAG**

KorQR-RL treats Korean query rewriting as a one-step offline RL problem for
retrieval-augmented generation. Given an original question and retrieval-state
features, a policy selects one rewrite action and receives reward from BM25,
dense, and hybrid retrieval results.

The current experiment uses the real `sentence-transformers` dense backend with
`intfloat/multilingual-e5-base`. For quick smoke runs, set
`dense_backend: lexical` in `configs/default.yaml`.

## Current Snapshot

- Processed QA set: 22,484 questions across KorQuAD 1.0, KLUE-MRC, and filtered KorQuAD 2.0.
- Current hard-case subset: 1,000 questions.
- Hard-case mix: 350 KLUE-MRC, 200 KorQuAD 1.0, 450 KorQuAD 2.0.
- Final annotation file: 1,000 rows; 948 rows currently have a primary failure label.
- Rewrite reward table: 15,000 rows = 1,000 questions x 5 strategies x 3 retrievers.
- Hard-case policy evaluation: 54,900 rows over 1,000 hard cases.
- Latest general policy evaluation: 5,000 sampled QA questions, 75,000 rows.

## Directory Layout

```text
configs/
  default.yaml                         Main experiment paths and parameters.
data/
  raw/                                 Local raw datasets.
  processed/                           Built corpora and QA JSONL files.
  outputs/
    original_retrieval/                Per-dataset original retrieval logs.
    retrieval_baselines/               Original, dense-only, and hybrid-alpha summaries.
    hard_cases/                        Extracted hard cases and 1,000-case subset.
    annotation/                        Annotation sheets and finalized labels.
    rewrite_candidates/                Generated rewrite candidates.
    evaluation/                        Hard-case reward, policy, and analysis tables.
    general_policy/                    General QA policy evaluation outputs.
    cache/                             LLM rewrite cache.
docs/                                  Report notes and generated markdown previews.
reports/                               Generated report-ready CSV exports.
scripts/                               Main pipeline entry points.
scripts/reporting/                     Optional reporting and review utilities.
scripts/korquad2_debug/                KorQuAD2 chunk diagnosis/rebuild utilities.
src/                                   Reusable preprocessing, retrieval, rewrite, and evaluation code.
```

## Pipeline

```bash
NLP/bin/python scripts/01_build_dataset.py
NLP/bin/python scripts/02_original_retrieval.py
NLP/bin/python scripts/03_extract_hard_cases.py
NLP/bin/python scripts/06_annotation_finalize.py --input data/outputs/hard_cases/hard_subset_1000.jsonl
NLP/bin/python scripts/07_generate_rewrites.py --input data/outputs/annotation/hard_subset_1000_annotation_final.jsonl --output-dir data/outputs/rewrite_candidates --overwrite
NLP/bin/python scripts/08_rewrite_retrieval_eval.py
NLP/bin/python scripts/09_reward_selection.py
NLP/bin/python scripts/10_analysis_tables.py
NLP/bin/python scripts/11_sentence_dense_retrieval.py
NLP/bin/python scripts/12_hybrid_alpha_sweep.py
NLP/bin/python scripts/13_rewrite_policy_eval.py
NLP/bin/python scripts/17_general_policy_eval.py --sample-size 5000 --output data/outputs/general_policy/general_policy_eval_5000.csv --summary data/outputs/general_policy/general_policy_summary_5000.csv
```

Optional report assets:

```bash
NLP/bin/python scripts/reporting/report_builder.py
```

Optional real LLM rewrite generation:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="..."
NLP/bin/python scripts/07_generate_rewrites.py --input data/outputs/annotation/hard_subset_1000_annotation_final.jsonl --output-dir data/outputs/rewrite_candidates --overwrite
```

Generated LLM rewrites are cached at
`data/outputs/cache/llm_rewrite_cache.jsonl`.

## Key Outputs

- `data/outputs/hard_cases/hard_subset_1000.jsonl`: Current 1,000-case hard subset.
- `data/outputs/annotation/hard_subset_1000_annotation_final.jsonl`: Final labels for the hard subset.
- `data/outputs/rewrite_candidates/hard_subset_1000_rewrite_candidates.jsonl`: Rewrite candidates.
- `data/outputs/evaluation/rewrite_results.jsonl`: Per-action retrieval and reward table.
- `data/outputs/evaluation/main_results.csv`: Strategy-level hard-case metrics.
- `data/outputs/evaluation/policy_results.csv`: Per-query policy decisions.
- `data/outputs/evaluation/policy_summary.csv`: Policy comparison summary.
- `data/outputs/evaluation/final_policy_comparison.csv`: Held-out hard-case policy comparison.
- `data/outputs/general_policy/general_policy_eval_5000.csv`: Latest 5,000-question general policy evaluation.
- `data/outputs/general_policy/general_policy_summary_5000.csv`: Summary of the 5,000-question evaluation.
- `data/outputs/retrieval_baselines/hybrid_alpha_sweep.csv`: Hybrid BM25/dense weight sweep.

## Retrieval Baselines

Original retrieval summary on the KorQuAD 1.0 dev baseline:

| retriever | Recall@10 | MRR | Answer F1 |
|---|---:|---:|---:|
| BM25 | 0.9790 | 0.9072 | 0.9846 |
| dense | 0.9707 | 0.8472 | 0.9803 |
| hybrid alpha=0.5 | 0.9863 | 0.9183 | 0.9905 |

Across the multi-dataset original retrieval logs, hybrid retrieval is strongest
on all three datasets:

| dataset | best retriever | Recall@10 | MRR |
|---|---|---:|---:|
| KorQuAD 1.0 | hybrid | 0.9792 | 0.9038 |
| KLUE-MRC | hybrid | 0.9389 | 0.8361 |
| KorQuAD 2.0 filtered | hybrid | 0.8003 | 0.6513 |

The best hybrid alpha by original-query MRR is `0.1` with Recall@10 `0.9868`
and MRR `0.9285`.

## Hard-Case Results

Best mean hard-case reward by retriever in `main_results.csv`:

| retriever | best strategy | Recall@10 | MRR | Answer F1 | Reward |
|---|---|---:|---:|---:|---:|
| BM25 | llm | 0.2080 | 0.0878 | 0.4679 | 0.4066 |
| dense | original | 0.0690 | 0.0311 | 0.0847 | 0.1225 |
| hybrid | llm | 0.0920 | 0.0363 | 0.3995 | 0.2307 |

Held-out hard-case policy comparison, 300 test rows per retriever:

| retriever | policy | Recall@10 | MRR | Answer F1 | Reward |
|---|---|---:|---:|---:|---:|
| BM25 | original_only | 0.1400 | 0.0630 | 0.3951 | 0.3665 |
| BM25 | offline_q_learning | 0.1933 | 0.0748 | 0.4413 | 0.3716 |
| BM25 | reward_selected | 0.2300 | 0.1077 | 0.5230 | 0.5274 |
| dense | original_only | 0.0767 | 0.0315 | 0.0777 | 0.1287 |
| dense | offline_q_learning | 0.0767 | 0.0335 | 0.0778 | 0.1220 |
| dense | reward_selected | 0.1067 | 0.0450 | 0.1257 | 0.1830 |
| hybrid | original_only | 0.0367 | 0.0061 | 0.3475 | 0.2109 |
| hybrid | offline_q_learning | 0.0767 | 0.0339 | 0.3623 | 0.1953 |
| hybrid | reward_selected | 0.1600 | 0.0636 | 0.4690 | 0.4096 |

`reward_selected` is an oracle-style upper bound from the logged action table.
The learned policies improve some retrieval metrics on hard cases, but they do
not yet reliably improve reward across retrievers.

## General Policy Evaluation

Latest 5,000-question evaluation, test split only:

| retriever | policy | Recall@10 | MRR | Answer F1 | Reward | Rewrite rate |
|---|---|---:|---:|---:|---:|---:|
| BM25 | original_only | 0.8753 | 0.7725 | 0.8621 | 1.6885 | 0.0000 |
| BM25 | score_gated_rl | 0.8760 | 0.7728 | 0.8624 | 1.6894 | 0.0047 |
| BM25 | oracle_gated_rl | 0.8833 | 0.7740 | 0.8715 | 1.6830 | 0.1247 |
| dense | original_only | 0.6167 | 0.5082 | 0.0458 | 0.8896 | 0.0000 |
| dense | score_gated_rl | 0.6167 | 0.5082 | 0.0458 | 0.8896 | 0.0000 |
| dense | oracle_gated_rl | 0.6267 | 0.5106 | 0.0457 | 0.8389 | 0.3833 |
| hybrid | original_only | 0.8647 | 0.6404 | 0.8413 | 1.6014 | 0.0000 |
| hybrid | score_gated_rl | 0.8647 | 0.6404 | 0.8413 | 1.6014 | 0.0000 |
| hybrid | oracle_gated_rl | 0.8847 | 0.6446 | 0.8516 | 1.6056 | 0.1353 |

On the broader 5,000-question sample, unconditional rewriting hurts reward.
The score-gated policy is intentionally conservative and mostly preserves the
original query, while oracle gating shows remaining headroom.

## Interpretation

- Rewriting is useful mainly for selected hard cases, not as a default action.
- Hybrid retrieval remains the strongest original-query retriever.
- LLM-style rewrites are the best average hard-case strategy for BM25 and hybrid, but not for dense retrieval.
- The current learned policies still lag the oracle/reward-selected upper bound.
- Better state features and a stronger contextual policy are the clearest next steps.
