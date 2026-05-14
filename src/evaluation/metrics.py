
from collections import Counter

from src.utils.text import tokenize


def recall_at_k(retrieved: list[dict], gold_doc_id: str, k: int) -> int:
    retrieved_ids = [item["doc_id"] for item in retrieved[:k]]
    return 1 if gold_doc_id in retrieved_ids else 0


def mrr(retrieved: list[dict], gold_doc_id: str) -> float:
    retrieved_ids = [item["doc_id"] for item in retrieved]
    if gold_doc_id in retrieved_ids:
        rank = retrieved_ids.index(gold_doc_id) + 1
        return 1.0 / rank
    return 0.0


def answer_f1(retrieved: list[dict], answer: str, k: int = 10) -> float:
    """Proxy answer-span F1 against retrieved document text.

    KorQuAD gives an answer span, while this project evaluates retrieval rather
    than span extraction. If the answer string appears in a retrieved passage,
    the support score is 1. Otherwise, use maximum token overlap F1.
    """

    if not answer or not answer.strip():
        return 0.0

    answer_norm = _compact(answer)
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    best = 0.0
    for item in retrieved[:k]:
        text = item.get("text", "")
        if answer_norm and answer_norm in _compact(text):
            return 1.0
        best = max(best, _token_f1(answer_tokens, tokenize(text)))
    return best


def _token_f1(gold_tokens: list[str], pred_tokens: list[str]) -> float:
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counter = Counter(gold_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((gold_counter & pred_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _compact(text: str) -> str:
    return "".join((text or "").lower().split())
