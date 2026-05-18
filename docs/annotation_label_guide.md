# Annotation Label Guide

Use `data/outputs/annotation/hard_subset_300_annotation.csv` for manual review.
Leave `failure_label` empty until a human annotator chooses one label. Use
`secondary_failure_label` only when a second cause is clearly relevant.

| label | meaning |
|---|---|
| `lexical_mismatch` | 질문 표현과 gold passage 표현이 달라 BM25가 실패한 경우 |
| `missing_key_term` | 검색에 필요한 핵심 단어가 질문에 없는 경우 |
| `entity_ambiguity` | 사람/장소/기관 이름이 모호한 경우 |
| `too_broad_query` | 질문이 너무 넓어서 여러 문서가 비슷하게 검색되는 경우 |
| `too_specific_query` | 질문이 너무 세부적이거나 문장 구조가 복잡한 경우 |
| `paraphrase_mismatch` | 의미는 같지만 표현이 달라 dense/hybrid도 못 잡은 경우 |
| `numeric_temporal_mismatch` | 날짜, 수치, 시간 표현 때문에 실패한 경우 |
| `question_type_mismatch` | how/why/comparison 등 질문 유형 때문에 실패한 경우 |
| `gold_mismatch` | gold_doc_id 자체가 잘못됐거나 정답 passage가 부적절한 경우 |
| `synthetic_chunk_issue` | KorQuAD2 synthetic chunk 때문에 생긴 실패 |
| `retrieval_model_failure` | 라벨상 문제는 명확하지 않고 retriever 점수화 실패로 보이는 경우 |
| `other` | 위에 해당하지 않는 경우 |

For KorQuAD2 rows, `suggested_failure_label` may contain
`synthetic_chunk_issue` because the local KorQuAD2 corpus was synthesized from
title and answer text. This is only a hint; keep `failure_label` blank until
manual annotation.
