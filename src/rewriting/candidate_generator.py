from src.utils.text import expanded_query, extract_keywords, llm_style_query, prompt_style_query, structured_query


class RewriteCandidateGenerator:
    """Generate rule-based query rewrite candidates for hard cases."""

    def generate(self, question: str, llm_query: str | None = None) -> dict[str, str]:
        if not question or not question.strip():
            return {
                "original": "",
                "keyword": "",
                "prompt": "",
                "expanded": "",
                "structured": "",
                "llm": "",
            }

        original = question.strip()
        keyword = extract_keywords(original)
        prompt = prompt_style_query(original)
        expanded = expanded_query(original)
        structured = structured_query(original)
        llm = llm_query.strip() if llm_query and llm_query.strip() else llm_style_query(original)

        return {
            "original": original,
            "keyword": keyword,
            "prompt": prompt,
            "expanded": expanded,
            "structured": structured,
            "llm": llm,
        }
