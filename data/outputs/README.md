# Output Layout

This directory keeps generated artifacts grouped by pipeline stage.

- `original_retrieval/`: full original-query retrieval logs by dataset and retriever.
- `retrieval_baselines/`: compact baseline summaries and hybrid-alpha sweeps.
- `hard_cases/`: extracted hard cases and the current `hard_subset_1000.jsonl`.
- `annotation/`: annotation sheets, assist files, needs-review rows, and final labels.
- `rewrite_candidates/`: generated rewrite candidates for each hard case.
- `evaluation/`: hard-case reward tables, policy summaries, and analysis tables.
- `general_policy/`: broader QA-set policy evaluations, including the latest 5,000-question run.
- `cache/`: reusable caches such as LLM rewrite outputs.
