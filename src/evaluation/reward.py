from dataclasses import dataclass

from src.utils.text import tokenize


@dataclass
class RewardCalculator:
    alpha: float = 1.0
    beta: float = 0.5
    answer_gamma: float = 0.5
    lambda_: float = 0.05
    drift_gamma: float = 0.2

    def compute_reward(
        self,
        recall_score: float,
        mrr_score: float,
        answer_f1_score: float,
        query: str,
        original_query: str | None = None,
    ) -> float:
        query_tokens = tokenize(query)
        # Do not punish normal Korean questions; penalize only verbose rewrites.
        length_penalty = self.lambda_ * max(0, len(query_tokens) - 12)
        drift_penalty = self.drift_gamma * self.semantic_drift(query, original_query)
        return (
            self.alpha * recall_score
            + self.beta * mrr_score
            + self.answer_gamma * answer_f1_score
            - length_penalty
            - drift_penalty
        )

    def semantic_similarity(self, query: str, original_query: str | None) -> float:
        if not original_query:
            return 1.0

        query_tokens = set(tokenize(query))
        original_tokens = set(tokenize(original_query))
        if not query_tokens or not original_tokens:
            return 1.0

        return len(query_tokens & original_tokens) / len(original_tokens)

    def semantic_drift(self, query: str, original_query: str | None) -> float:
        return 1.0 - self.semantic_similarity(query, original_query)
