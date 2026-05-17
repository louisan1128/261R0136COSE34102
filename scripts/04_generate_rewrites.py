import sys
import os
import argparse
import random
import re
import time
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
        help="Optional number of first hard cases to rewrite for smoke tests.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many hard cases before rewrite generation.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed used with --sample-size.",
    )
    parser.add_argument(
        "--llm-delay-seconds",
        type=float,
        default=None,
        help="Optional delay after each external LLM API call. Useful for low RPM limits.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=None,
        help="Optional number of retries for rate-limit errors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    llm_config = config.get("llm_rewrite", {})
    sample_size = args.sample_size if args.sample_size is not None else llm_config.get("sample_size")
    sample_seed = args.sample_seed if args.sample_seed is not None else int(llm_config.get("sample_seed", 7))
    sample_path = llm_config.get("sample_path", "data/outputs/hard_cases_random_sample.jsonl")

    if sample_size:
        hard_cases = _sample_hard_cases(hard_cases, int(sample_size), sample_seed)
        write_jsonl(hard_cases, sample_path)
        print(f"Sampled {len(hard_cases)} hard cases with seed={sample_seed}; saved to {sample_path}.")

    if args.limit:
        hard_cases = hard_cases[: args.limit]
        print(f"Using first {len(hard_cases)} hard cases after sampling/filtering because --limit was set.")

    generator = RewriteCandidateGenerator()
    use_external_llm = (bool(llm_config.get("enabled", False)) or args.use_external_llm) and not args.no_external_llm
    allow_fallback = bool(llm_config.get("allow_fallback", True))
    llm_delay_seconds = (
        args.llm_delay_seconds
        if args.llm_delay_seconds is not None
        else float(llm_config.get("request_delay_seconds", 0.0))
    )
    llm_max_retries = (
        args.llm_max_retries
        if args.llm_max_retries is not None
        else int(llm_config.get("max_retries", 4))
    )
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
    last_llm_request_at = 0.0
    for hard_case in hard_cases:
        question = hard_case["question"]
        llm_query = None
        if rewriter:
            llm_query = rewrite_cache.get(question)
            if not llm_query:
                try:
                    last_llm_request_at = _wait_for_request_slot(last_llm_request_at, llm_delay_seconds)
                    llm_query = _rewrite_with_retries(
                        rewriter,
                        question,
                        hard_case.get("failure_type", "unlabeled"),
                        max_retries=llm_max_retries,
                    )
                    last_llm_request_at = time.monotonic()
                    rewrite_cache[question] = llm_query
                    api_calls += 1
                    save_rewrite_cache(rewrite_cache, cache_path)
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


def _rewrite_with_retries(rewriter, question: str, failure_type: str, max_retries: int) -> str:
    attempts = 0
    while True:
        try:
            return rewriter.rewrite(question, failure_type)
        except Exception as exc:
            attempts += 1
            if attempts > max_retries or not _is_rate_limit_error(exc):
                raise
            wait_seconds = _retry_after_seconds(exc, attempts)
            print(f"Rate limit hit; waiting {wait_seconds:.1f}s before retry {attempts}/{max_retries}.")
            time.sleep(wait_seconds)


def _sample_hard_cases(hard_cases: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0 or sample_size >= len(hard_cases):
        return list(hard_cases)
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(hard_cases)), sample_size))
    return [hard_cases[index] for index in sampled_indices]


def _wait_for_request_slot(last_request_at: float, delay_seconds: float) -> float:
    if delay_seconds <= 0 or last_request_at <= 0:
        return last_request_at
    elapsed = time.monotonic() - last_request_at
    remaining = delay_seconds - elapsed
    if remaining > 0:
        print(f"Waiting {remaining:.1f}s before next LLM request.")
        time.sleep(remaining)
    return last_request_at


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "http 429" in message or "rate limit" in message


def _retry_after_seconds(exc: Exception, attempts: int) -> float:
    message = str(exc)
    match = re.search(r"try again in ([0-9.]+)\s*s", message, flags=re.IGNORECASE)
    if match:
        return max(1.0, float(match.group(1)) + 2.0)
    return min(300.0, 21.0 * (2 ** max(0, attempts - 1)))


if __name__ == "__main__":
    main()
