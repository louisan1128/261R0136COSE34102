from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever


def build_retrievers(config: dict) -> dict:
    data_config = config["data"]
    bm25 = BM25Retriever(data_config["corpus_path"])
    dense = DenseRetriever(
        data_config["corpus_path"],
        model_name=config["model_name"],
        embedding_dir=data_config["embedding_dir"],
        backend=config.get("dense_backend", "auto"),
        device=config.get("dense_device"),
    )
    hybrid = HybridRetriever(bm25, dense, alpha=config["hybrid"]["alpha"])
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}
