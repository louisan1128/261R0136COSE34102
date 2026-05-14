import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import extract_keywords


SYSTEM_PROMPT = """You rewrite Korean QA questions into retrieval queries for RAG.
Preserve the original answer intent and named entities.
Do not answer the question.
Do not add unsupported facts.
Return exactly one concise Korean search query."""


@dataclass
class LLMRewriteConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 96
    timeout_seconds: int = 30


class OpenAICompatibleRewriter:
    """Small dependency-free client for OpenAI-compatible chat completion APIs."""

    def __init__(self, config: LLMRewriteConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleRewriter":
        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        model_env = str(config.get("model_env", "OPENAI_MODEL"))
        api_key = os.environ.get(api_key_env, "").strip()
        model = os.environ.get(model_env, "").strip() or str(config.get("model", "")).strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or str(
            config.get("base_url", "https://api.openai.com/v1")
        ).strip()

        if not api_key:
            raise ValueError(f"External LLM rewrite needs ${api_key_env}.")
        if not model:
            raise ValueError(f"External LLM rewrite needs ${model_env} or llm_rewrite.model in config.")

        return cls(
            LLMRewriteConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=float(config.get("temperature", 0.2)),
                max_tokens=int(config.get("max_tokens", 96)),
                timeout_seconds=int(config.get("timeout_seconds", 30)),
            )
        )

    def rewrite(self, question: str, failure_type: str = "unlabeled") -> str:
        question = question.strip()
        if not question:
            return ""

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_prompt(question, failure_type),
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = urllib.request.Request(
            url=self._chat_completions_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM rewrite request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM rewrite request failed: {exc.reason}") from exc

        content = self._extract_content(json.loads(body))
        return _clean_single_query(content)

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _build_user_prompt(self, question: str, failure_type: str) -> str:
        keywords = extract_keywords(question)
        return (
            f"Original Korean question: {question}\n"
            f"Detected failure type: {failure_type}\n"
            f"Extracted keywords: {keywords}\n\n"
            "Rewrite it as one retrieval query for finding the answer evidence passage. "
            "Prefer key entities, answer target words, and safe synonyms. "
            "Keep it under 40 Korean eojeol. Return only the query text."
        )

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM rewrite response has no choices: {response}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if not content:
            raise RuntimeError(f"LLM rewrite response has empty content: {response}")
        return str(content)


def load_rewrite_cache(path: str | Path) -> dict[str, str]:
    cache = {}
    for record in read_jsonl(path):
        question = str(record.get("question", "")).strip()
        rewrite = str(record.get("rewrite", "")).strip()
        if question and rewrite:
            cache[question] = rewrite
    return cache


def save_rewrite_cache(cache: dict[str, str], path: str | Path) -> None:
    records = [{"question": question, "rewrite": rewrite} for question, rewrite in sorted(cache.items())]
    write_jsonl(records, path)


def _clean_single_query(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:\w+)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.strip("\"'` ")
    if "\n" in cleaned:
        cleaned = next((line.strip("-*0123456789. \t") for line in cleaned.splitlines() if line.strip()), "")
    return re.sub(r"\s+", " ", cleaned).strip()
