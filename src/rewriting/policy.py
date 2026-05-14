
import re

from src.utils.text import tokenize


FAILURE_TYPE_POLICY = {
    "expression_mismatch": ["keyword", "expanded", "llm"],
    "ellipsis": ["structured", "expanded"],
    "compound_noun": ["keyword", "structured"],
    "colloquial_mismatch": ["prompt", "structured", "llm"],
    "abbreviation": ["expanded", "keyword"],
    "temporal_numeric": ["keyword", "structured"],
    "unlabeled": ["original", "keyword", "expanded", "llm"],
}


def select_strategies(failure_type: str) -> list[str]:
    return FAILURE_TYPE_POLICY.get(failure_type, ["original", "keyword", "expanded"])


def infer_failure_type(question: str) -> str:
    """Assign a lightweight failure label for aggregate analysis.

    The labels are heuristic, but they keep the analysis table from collapsing
    into a single unlabeled bucket until manual annotation is available.
    """

    tokens = tokenize(question)
    if not tokens:
        return "unlabeled"

    if re.search(r"[A-Z]{2,}|[A-Za-z]+-[A-Za-z0-9]+", question):
        return "abbreviation"
    if re.search(r"\d", question):
        return "temporal_numeric"
    if len(tokens) <= 4 or any(token in {"그", "그것", "그곳", "그녀", "그는", "이곳"} for token in tokens):
        return "ellipsis"
    if any(len(token) >= 7 for token in tokens):
        return "compound_noun"
    if question.endswith(("?", "요?", "나요?", "인가요?")) and len(tokens) >= 8:
        return "expression_mismatch"
    return "unlabeled"
