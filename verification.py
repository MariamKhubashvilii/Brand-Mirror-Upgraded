import difflib
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def normalize_token(token: str) -> str:
    token = re.sub(r"[^a-z0-9]+", " ", token.lower()).strip()
    return re.sub(r"\s+", " ", token)


def simple_stem(token: str) -> str:
    token = normalize_token(token)
    if not token:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def keyword_present(text: str, keyword: str) -> bool:
    norm_text = set(simple_stem(t) for t in normalize_token(text).split())
    norm_kw = simple_stem(normalize_token(keyword))
    if not norm_kw:
        return False
    return norm_kw in norm_text


def compliance_report(draft: str, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rules = rules or {}
    text = draft or ""
    missing_keywords = []
    banned_words_found = []
    structural_issues = []

    target_keywords = rules.get("keywords", [])
    for keyword in target_keywords:
        if not keyword_present(text, keyword):
            missing_keywords.append(keyword)

    banned_words = rules.get("banned_words", [])
    for word in banned_words:
        pattern = re.compile(re.escape(word.lower()))
        if pattern.search(text.lower()):
            banned_words_found.append(word)

    min_words = rules.get("min_words")
    if min_words is not None:
        word_count = len(re.findall(r"\b\w+\b", text))
        if word_count < min_words:
            structural_issues.append(f"word_count below minimum: {word_count} < {min_words}")

    required_headings = rules.get("required_headings", [])
    headings = re.findall(r"^(#{1,6})\s+(.*)$", text, flags=re.MULTILINE)
    present_headings = [h[1].strip() for h in headings]
    for heading in required_headings:
        if heading not in present_headings:
            structural_issues.append(f"missing heading: {heading}")

    intro_length = rules.get("intro_length_max")
    if intro_length is not None:
        intro = re.split(r"\n#{1,6}\s+", text, maxsplit=1)[0].strip()
        intro_words = len(re.findall(r"\b\w+\b", intro))
        if intro_words > intro_length:
            structural_issues.append(f"intro too long: {intro_words} > {intro_length}")

    faq_required = rules.get("require_faq", False)
    if faq_required:
        if not re.search(r"^#+\s*faq\b", text, flags=re.IGNORECASE | re.MULTILINE):
            structural_issues.append("missing FAQ section")

    return {
        "missing_keywords": missing_keywords,
        "banned_words_found": banned_words_found,
        "structural_issues": structural_issues,
    }


def summarize_diff(before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    return "\n".join(diff[:80])
