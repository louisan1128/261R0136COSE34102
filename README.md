# Ko-RL_QR

Retriever-aware selective query rewriting for Korean retrieval-augmented
generation (RAG).

This repository contains the code and experiment artifacts for the Korea
University COSE461 final project, **Retriever-Aware Selective Query Rewriting
for Korean RAG**. The project studies when query rewriting helps retrieval, and
when preserving the original query is safer.

## Overview

Query rewriting is often used as a preprocessing step before retrieval, but
rewriting every query can hurt cases where the original query already retrieves
the right evidence. This project formulates rewriting as a selective decision
problem over five actions:

- `original`
- `keyword`
- `expanded`
- `structured`
- `llm`

The framework evaluates BM25, dense, and hybrid retrievers, then uses a
confidence gate and retriever-aware policy to decide whether and how to rewrite.

## Main Results

On 300 held-out challenging Korean QA cases, the refined oracle-feature policy
improves Hybrid Recall@10 from `0.1467` to `0.2400`. BM25 also benefits from
selective rewriting, while dense retrieval requires more conservative handling.

On a 1,500-query general split, unconditional LLM rewriting harms dense and
hybrid retrieval, while score-gated rewriting largely preserves original-query
performance.

The full write-up is in [`paper.tex`](paper.tex).

## Repository Layout

```text
configs/        Experiment configuration.
data/           Local datasets and generated experiment outputs.
docs/           Notes, annotation guide, and report planning docs.
reports/        Report-facing generated summaries.
scripts/        Reproducible pipeline entry points.
src/            Reusable preprocessing, retrieval, rewriting, and evaluation code.
paper.tex       Final project paper source.
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dense retrieval uses `intfloat/multilingual-e5-base` through
Sentence-Transformers. For a fast smoke run, set `dense_backend: lexical` in
`configs/default.yaml`.

## Data

Experiments use Korean QA data from KorQuAD 1.0, KLUE-MRC, and KorQuAD 2.0.
Raw datasets should be placed under `data/raw/`; processed corpora, retrieval
logs, rewrite candidates, and evaluation tables are generated under `data/`.

Large local artifacts such as dense embeddings are written to `embeddings/` and
should not be committed unless intentionally publishing a frozen artifact.

## Pipeline

Run the main stages in order:

```bash
python scripts/01_build_dataset.py
python scripts/02_original_retrieval.py
python scripts/03_extract_hard_cases.py
python scripts/06_annotation_finalize.py --input data/outputs/hard_cases/hard_subset_1000.jsonl
python scripts/07_generate_rewrites.py --input data/outputs/annotation/hard_subset_1000_annotation_final.jsonl --output-dir data/outputs/rewrite_candidates --overwrite
python scripts/08_rewrite_retrieval_eval.py
python scripts/09_reward_selection.py
python scripts/10_analysis_tables.py
python scripts/11_sentence_dense_retrieval.py
python scripts/12_hybrid_alpha_sweep.py
python scripts/13_rewrite_policy_eval.py
python scripts/14_report_diagnostics.py
python scripts/17_general_policy_eval.py --sample-size 5000
```

LLM rewrites are optional. To use an OpenAI-compatible API, set:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="..."
```

Generated LLM rewrites are cached at
`data/outputs/cache/llm_rewrite_cache.jsonl`.

## Notes

- The refined policy reported in the paper uses manual failure labels and
  logged gold-rank buckets, so it is an offline oracle-feature analysis rather
  than a fully deployable system.
- The observable-feature selector removes oracle features from selection, but
  still shares the same candidate pool as the oracle-feature analysis.
- Before publishing, check whether raw/processed datasets and cached LLM outputs
  should remain tracked in Git for your release.
