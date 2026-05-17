# Results Summary

- Total QA pairs: 5,774
- Union hard cases: 265
- Random hard subset: 100
- Rewrite candidates: 100
- Rewrite evaluation rows: 1,800
- LLM rewrite cache entries: 109

## Main Takeaways

- Best original retriever: `hybrid` with Recall@10 98.6% and MRR 0.9183.
- Best hybrid alpha by MRR: `0.1` with MRR 0.9285.
- Final test winner for `bm25`: `reward_selected_rewrite` (Recall@10 76.7%, MRR 0.5842).
- Final test winner for `dense`: `reward_selected_rewrite` (Recall@10 43.3%, MRR 0.2070).
- Final test winner for `hybrid`: `reward_selected_rewrite` (Recall@10 83.3%, MRR 0.5719).

## Consistency Checks

- OK: rewrite_results rows match candidates x 6 strategies x 3 retrievers (1800).
- OK: sampled hard cases and rewrite candidates use the same count.
- OK: LLM cache has at least as many entries as current rewrite candidates.

HTML dashboard: `docs\results_dashboard.html`
