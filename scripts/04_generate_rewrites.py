import sys
import os
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.rewriting.candidate_generator import RewriteCandidateGenerator
from src.rewriting.llm_client import OpenAICompatibleRewriter, load_rewrite_cache, save_rewrite_cache
from src.utils.io import read_jsonl, write_jsonl, read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rewrite candidates for KorQR-RL hard cases.")
    parser.add_argument(
        "--use-external-llm",
        action="store_true",
        help="Use an OpenAI-compatible external LLM for the llm rewrite action.",
    )
    parser.add_argument(
        "--no-external-llm",
        action="store_true",
        help="Force deterministic llm-style fallback even when config enables external LLM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of hard cases to rewrite for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    if args.limit:
        hard_cases = hard_cases[: args.limit]

    generator = RewriteCandidateGenerator()
    llm_config = config.get("llm_rewrite", {})
    use_external_llm = (bool(llm_config.get("enabled", False)) or args.use_external_llm) and not args.no_external_llm
    allow_fallback = bool(llm_config.get("allow_fallback", True))
    rewriter = None
    rewrite_cache = {}
    cache_path = llm_config.get("cache_path", "data/outputs/llm_rewrite_cache.jsonl")

    if use_external_llm:
        rewriter = OpenAICompatibleRewriter.from_config(llm_config)
        rewrite_cache = load_rewrite_cache(cache_path)
        print(f"External LLM rewrite enabled with model '{rewriter.config.model}'.")
        print(f"Loaded {len(rewrite_cache)} cached LLM rewrites from {cache_path}.")

    candidates = []
    api_calls = 0
    fallback_count = 0
    fallback_examples = []
    for hard_case in hard_cases:
        question = hard_case["question"]
        llm_query = None
        if rewriter:
            llm_query = rewrite_cache.get(question)
            if not llm_query:
                try:
                    llm_query = rewriter.rewrite(question, hard_case.get("failure_type", "unlabeled"))
                    rewrite_cache[question] = llm_query
                    api_calls += 1
                except Exception as exc:
                    if not allow_fallback:
                        raise
                    fallback_count += 1
                    if len(fallback_examples) < 5:
                        fallback_examples.append(f"qid={hard_case['qid']}: {exc}")

        candidate_queries = generator.generate(question, llm_query=llm_query)
        candidates.append({
            "qid": hard_case["qid"],
            "question": question,
            "failure_type": hard_case.get("failure_type", "unlabeled"),
            "candidates": candidate_queries,
        })

    output_path = Path(data_config["rewrite_candidates_path"])
    write_jsonl(candidates, output_path)
    if rewriter:
        save_rewrite_cache(rewrite_cache, cache_path)
        print(f"Saved {len(rewrite_cache)} cached LLM rewrites to {cache_path}.")
        print(f"LLM API calls: {api_calls}; deterministic fallbacks: {fallback_count}")
        if fallback_examples:
            print("Fallback examples:")
            for example in fallback_examples:
                print(f"- {example}")
    print(f"Saved rewrite candidates to {output_path}")


if __name__ == "__main__":
    main()
