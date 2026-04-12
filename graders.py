"""
Deterministic, rule-based graders for each task difficulty.
All graders return a float in [0.05, 0.95].
Same input always produces same score (deterministic).
Different inputs produce different scores (non-trivial).
"""

import re


def _count_words(text: str) -> int:
    return len(text.split())


def _count_paragraphs(text: str) -> int:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs)


def _has_title(text: str) -> bool:
    lines = text.strip().split("\n")
    for line in lines[:3]:
        if line.startswith("#"):
            return True
    return False


def _topic_relevance(text: str, topic: str) -> float:
    """Check what fraction of topic words appear in the article."""
    topic_words = set(w.lower() for w in topic.split() if len(w) > 3)
    if not topic_words:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for w in topic_words if w in text_lower)
    return matched / len(topic_words)


def _keywords_present(text: str, keywords: list) -> float:
    """Return fraction of required keywords found in text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def _word_count_in_range(text: str, min_words: int, max_words: int) -> float:
    """Score 1.0 if in range, partial credit for close misses."""
    wc = _count_words(text)
    if min_words <= wc <= max_words:
        return 1.0
    elif wc < min_words:
        return max(0.0, wc / min_words)
    else:
        # Over max — partial penalty
        overage = wc - max_words
        return max(0.5, 1.0 - (overage / max_words) * 0.5)


def _has_sections(text: str, sections: list) -> float:
    """Return fraction of required section headings found."""
    if not sections:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for s in sections if s.lower() in text_lower)
    return found / len(sections)


def _has_stats(text: str) -> float:
    """Check for numeric statistics or percentages."""
    # Match patterns like: 42%, $1.2 billion, 3.5 million, 2025, etc.
    stat_pattern = r"\b\d+[\.,]?\d*\s*(%|billion|million|trillion|thousand|percent)|\$\d+|\b\d{4}\b"
    matches = re.findall(stat_pattern, text, re.IGNORECASE)
    if len(matches) >= 2:
        return 1.0
    elif len(matches) == 1:
        return 0.5
    return 0.0


def _exact_quotes(text: str, required_count: int) -> float:
    """Check if exactly the required number of blockquotes are present."""
    count = sum(1 for line in text.split("\n") if line.strip().startswith(">"))
    return 1.0 if count == required_count else 0.0


def _exact_citations(text: str, required_count: int) -> float:
    """Check if exactly the required number of citations [Ref: X] are present."""
    matches = re.findall(r"\[Ref: \d+\]", text)
    return 1.0 if len(matches) == required_count else 0.0


def _title_length(text: str, exact_length: int) -> float:
    """Check if the main title (# ...) has exactly the required number of words."""
    for line in text.strip().split("\n"):
        if line.startswith("# "):
            title_text = line[2:].strip()
            return 1.0 if len(title_text.split()) == exact_length else 0.0
    return 0.0


def _starts_with(text: str, phrase: str) -> float:
    """Check if the first body paragraph begins exactly with the phrase."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        if line.startswith("#"):
            continue
        return 1.0 if line.startswith(phrase) else 0.0
    return 0.0


def _exact_word_frequency(text: str, target_word: str, exact_count: int) -> float:
    """Check if target_word appears exactly exact_count times."""
    # Match whole words, case-insensitive
    matches = re.findall(rf"\b{re.escape(target_word)}\b", text, re.IGNORECASE)
    return 1.0 if len(matches) == exact_count else 0.0


def _exact_nth_word(text: str, target_word: str, n: int) -> float:
    """Check if the N-th word (1-indexed) of the article body is exactly target_word."""
    # Strip markdown symbols and split into words
    cleaned = re.sub(r"[#>*_\[\]()]", "", text)
    words = [w.strip(".,;:!?\"'") for w in cleaned.split() if w.strip(".,;:!?\"'")]
    if len(words) < n:
        return 0.0
    return 1.0 if words[n - 1].lower() == target_word.lower() else 0.0


def grade_easy(article: str, topic: str, constraints: dict = None) -> float:
    """
    Easy grader: checks basic article quality.
    Criteria:
      - Word count >= 200          (20%)
      - Paragraph count >= 3       (20%)
      - Has a markdown title       (20%)
      - Topic relevance >= 0.5     (20%)
      - Starts with phrase         (20%)
    """
    if not article or not article.strip():
        return 0.05

    scores = []

    # Word count
    wc = _count_words(article)
    wc_score = min(1.0, wc / 200)
    scores.append(("word_count", wc_score, 0.20))

    # Paragraph count
    pc = _count_paragraphs(article)
    pc_score = min(1.0, pc / 3)
    scores.append(("paragraphs", pc_score, 0.20))

    # Has title
    title_score = 1.0 if _has_title(article) else 0.0
    scores.append(("has_title", title_score, 0.20))

    # Topic relevance
    relevance = _topic_relevance(article, topic)
    scores.append(("relevance", relevance, 0.20))

    # Starts with
    if constraints and "starting_phrase" in constraints:
        sw_score = _starts_with(article, constraints["starting_phrase"])
        scores.append(("starts_with", sw_score, 0.20))
    else:
        # Default if no constraint passed (distribute weight)
        scores.append(("starts_with", 1.0, 0.20))

    total = sum(score * weight for _, score, weight in scores)
    return round(min(0.95, max(0.05, total)), 4)


def grade_medium(article: str, topic: str, constraints: dict) -> float:
    """
    Medium grader: checks structure + constraint compliance.
    Criteria:
      - All easy criteria          (40%)
      - Keywords present           (30%)
      - Word count in range        (30%)
    """
    if not article or not article.strip():
        return 0.05

    # Easy base score
    easy_score = grade_easy(article, topic, constraints)

    # Keywords
    keywords = constraints.get("keywords", [])
    kw_score = _keywords_present(article, keywords)

    # Word count range
    min_w = constraints.get("min_words", 250)
    max_w = constraints.get("max_words", 500)
    range_score = _word_count_in_range(article, min_w, max_w)

    # Exact paragraphs
    req_paragraphs = constraints.get("exact_paragraphs", 0)
    if req_paragraphs > 0:
        pc = _count_paragraphs(article)
        para_score = 1.0 if pc == req_paragraphs else 0.0
    else:
        para_score = 1.0

    req_quotes = constraints.get("required_quotes", 0)
    if req_quotes > 0:
        quotes_score = _exact_quotes(article, req_quotes)
        total = (easy_score * 0.25) + (kw_score * 0.20) + (range_score * 0.20) + (quotes_score * 0.20) + (para_score * 0.15)
    else:
        total = (easy_score * 0.40) + (kw_score * 0.30) + (range_score * 0.30)

    return round(min(0.95, max(0.05, total)), 4)


def grade_hard(article: str, topic: str, constraints: dict) -> float:
    """
    Hard grader: checks investigative article structure, stats, and depth.
    Criteria:
      - Easy base criteria         (30%)
      - Required sections present  (25%)
      - Statistics/facts present   (20%)
      - Word count in range        (15%)
      - Topic relevance >= 0.7     (10%)
    """
    if not article or not article.strip():
        return 0.05

    # Easy base
    easy_score = grade_easy(article, topic, constraints)

    # Sections
    sections = constraints.get("sections", [])
    section_score = _has_sections(article, sections)

    # Stats
    stats_score = _has_stats(article)

    # Word count range
    min_w = constraints.get("min_words", 400)
    max_w = constraints.get("max_words", 700)
    range_score = _word_count_in_range(article, min_w, max_w)

    # Deep topic relevance
    relevance = _topic_relevance(article, topic)
    deep_relevance = 1.0 if relevance >= 0.7 else relevance / 0.7

    # Exact citations
    req_citations = constraints.get("exact_citations", 3)
    citations_score = _exact_citations(article, req_citations)

    # Title length
    req_title_len = constraints.get("title_length", 6)
    title_len_score = _title_length(article, req_title_len)

    # Word Frequency
    freq_data = constraints.get("exact_word_frequency", {})
    freq_score = 1.0
    if freq_data:
        f_scores = [_exact_word_frequency(article, w, c) for w, c in freq_data.items()]
        freq_score = sum(f_scores) / len(f_scores) if f_scores else 1.0

    # Exact N-th word check
    nth_data = constraints.get("exact_nth_word", {})
    nth_score = 1.0
    if nth_data:
        nth_score = _exact_nth_word(article, nth_data["word"], nth_data["n"])

    total = (
        easy_score * 0.10
        + section_score * 0.15
        + stats_score * 0.10
        + range_score * 0.15
        + deep_relevance * 0.05
        + citations_score * 0.15
        + title_len_score * 0.10
        + freq_score * 0.10
        + nth_score * 0.10
    )
    return round(min(0.95, max(0.05, total)), 4)


def grade(article: str, topic: str, difficulty: str, constraints: dict) -> float:
    """Unified grading entry point."""
    if difficulty == "easy":
        return grade_easy(article, topic, constraints)
    elif difficulty == "medium":
        return grade_medium(article, topic, constraints)
    elif difficulty == "hard":
        return grade_hard(article, topic, constraints)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
