import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verification import compliance_report
from llm import extract_banned_words, select_relevant_context


def test_compliance_report_flags_missing_keywords_and_banned_words():
    draft = "# Intro\n\nThis article explains how to buy shoes."
    rules = {
        "keywords": ["running shoes", "size guide"],
        "banned_words": ["innovative"],
        "min_words": 20,
        "required_headings": ["FAQ"],
    }

    report = compliance_report(draft, rules)

    assert "running shoes" in report["missing_keywords"]
    assert report["missing_keywords"] == ["running shoes", "size guide"]
    assert report["banned_words_found"] == []
    assert any("FAQ" in issue for issue in report["structural_issues"])


def test_extract_banned_words_and_select_relevant_context():
    brand_knowledge = "Brand voice: direct\nAudience: young runners\nBanned words: innovative, leverage\nCTA style: short"
    banned = extract_banned_words(brand_knowledge)
    relevant = select_relevant_context("running shoes", brand_knowledge)

    assert banned == ["innovative", "leverage"]
    assert "Audience: young runners" in relevant
