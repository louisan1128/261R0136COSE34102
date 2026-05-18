import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_original import evaluate_named_retrievers, load_qa_pairs
from src.retrievers.factory import build_retrievers
from src.utils.io import read_yaml, write_csv


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    config["dense_backend"] = "sentence_transformers"

    data_config = config["data"]
    qa_pairs = load_qa_pairs(data_config["qa_path"])
    retrievers = build_retrievers(config)
    results = evaluate_named_retrievers(retrievers, qa_pairs, top_k=config.get("top_k", 10))

    for record in results:
        record["dense_backend"] = config["dense_backend"]
        record["model_name"] = config["model_name"]
        record["hybrid_alpha"] = config["hybrid"]["alpha"] if record["retriever"] == "hybrid" else ""

    output_path = Path(data_config["sentence_dense_results_path"])
    write_csv(results, output_path)
    print(f"Saved sentence-transformers dense baseline results to {output_path}")


if __name__ == "__main__":
    main()
