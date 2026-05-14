from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Base retriever interface for KorQR-RL."""

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError("Subclasses must implement retrieve()")
