import re
from collections import Counter


KOREAN_STOPWORDS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "에게",
    "으로",
    "로",
    "와",
    "과",
    "의",
    "도",
    "만",
    "및",
    "그리고",
    "또는",
    "무엇",
    "누구",
    "언제",
    "어디",
    "어떤",
    "몇",
    "하는",
    "되는",
    "된",
    "한",
    "것",
    "수",
    "날",
}

SYNONYM_EXPANSIONS = {
    "세종대왕": ["세종", "조선", "훈민정음"],
    "훈민정음": ["한글", "문자", "창제"],
    "한글": ["훈민정음", "문자", "창제"],
    "조선": ["왕조", "국가", "시대"],
    "대통령": ["정부", "국가원수", "정치"],
    "국회": ["의회", "정치", "입법"],
    "여의도": ["국회", "서울", "정치"],
    "시위": ["집회", "농성", "운동"],
    "폭력": ["사건", "혐의", "충돌"],
    "지명수배": ["수배", "혐의", "경찰"],
    "조사": ["수사", "경찰", "기관"],
    "군": ["군대", "부대", "장교"],
    "장군": ["군인", "지휘관", "군대"],
    "작품": ["저서", "책", "문학"],
    "발간": ["출판", "출간", "공개"],
    "결혼": ["배우자", "가족", "혼인"],
    "독립": ["해방", "운동", "일제"],
}

PARTICLE_SUFFIXES = (
    "으로부터",
    "에서",
    "에게",
    "으로",
    "부터",
    "까지",
    "처럼",
    "보다",
    "이다",
    "였다",
    "했다",
    "한다",
    "되는",
    "하는",
    "이며",
    "이고",
    "라고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "와",
    "과",
    "로",
)

NO_PARTICLE_STRIP = {
    "강원도",
    "경기도",
    "경상도",
    "여의도",
    "전라도",
    "제주도",
    "충청도",
    "한반도",
}


def tokenize(text: str) -> list[str]:
    """Tokenize Korean text with a dependency-light rule-based tokenizer."""

    if not text:
        return []
    cleaned = re.sub(r"[^0-9A-Za-z가-힣]+", " ", text)
    raw_tokens = [token for token in cleaned.strip().split() if token]
    tokens = [_normalize_token(token) for token in raw_tokens]
    return [token for token in tokens if token]


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    for suffix in PARTICLE_SUFFIXES:
        if len(suffix) == 1 and token in NO_PARTICLE_STRIP:
            continue
        min_stem_length = 2 if len(suffix) == 1 else len(suffix) + 1
        if len(token) > min_stem_length and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def extract_keywords(text: str, max_keywords: int = 8) -> str:
    tokens = tokenize(text)
    candidates = [token for token in tokens if len(token) > 1 and token not in KOREAN_STOPWORDS]
    counts = Counter(candidates)
    first_seen = {token: idx for idx, token in enumerate(candidates)}
    ranked = sorted(
        counts,
        key=lambda token: (-counts[token], -len(token), first_seen[token]),
    )
    return " ".join(ranked[:max_keywords])


def _legacy_search_intent_query(question: str) -> str:
    question = question.strip()
    if not question:
        return ""
    return f"{question} 정답의 근거가 되는 문서와 핵심 사실"


def llm_style_query(question: str) -> str:
    question = question.strip()
    keywords = extract_keywords(question)
    if not question:
        return ""
    return f"{keywords} 관련 문서에서 '{question}'의 정답 근거를 찾기"


def expanded_query(question: str) -> str:
    tokens = tokenize(question)
    expansions = []
    for token in tokens:
        expansions.extend(SYNONYM_EXPANSIONS.get(token, []))

    keywords = extract_keywords(question)
    expanded_terms = " ".join(dict.fromkeys(expansions))
    return " ".join(part for part in [question.strip(), keywords, expanded_terms] if part)


def structured_query(question: str) -> str:
    question = question.strip()
    keywords = extract_keywords(question)
    if not question:
        return ""
    return f"핵심어: {keywords} 질문: {question} 찾을 정보: 정답 근거 문서"
