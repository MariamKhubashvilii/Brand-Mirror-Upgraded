import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import research_competitors, _looks_like_listicle_topic, _needs_retry_for_sparse_outline
from source_flow import build_competitor_results, finalize_competitor_results


class FakeResponse:
    def __init__(self, payload):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": payload})()})]


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload):
        self.chat = type("Chat", (), {"completions": FakeCompletions(payload)})()


def test_build_competitor_results_matches_html_pastes_to_original_slot_indexes():
    def scrape(url):
        return {"success": False, "error": "boom", "url": url, "title": url, "text": "", "sections": []}

    def parse_html(html, label):
        return {"success": True, "text": html, "sections": [], "title": label}

    results = build_competitor_results(
        ["", "https://two.com", "https://three.com"],
        ["", "<p>paste two</p>", ""],
        scrape,
        parse_html,
    )

    assert len(results) == 2
    assert results[0]["slot_index"] == 1
    assert results[0]["url"] == "https://two.com"
    assert results[0]["text"] == "<p>paste two</p>"
    assert results[1]["slot_index"] == 2
    assert results[1]["url"] == "https://three.com"


def test_finalize_competitor_results_keeps_all_sources_when_pastes_fill_failures():
    results = [
        {"url": "https://one.com", "success": False, "slot_index": 0},
        {"url": "https://two.com", "success": False, "slot_index": 1},
        {"url": "https://three.com", "success": True, "slot_index": 2},
    ]

    finalized, excluded = finalize_competitor_results(results, ["pasted one", "pasted two", ""], lambda text: len(text.split()) >= 1)

    assert len(finalized) == 3
    assert [item["url"] for item in finalized] == ["https://one.com", "https://two.com", "https://three.com"]
    assert excluded == []


def test_research_competitors_reports_all_sources():
    client = FakeClient('{"search_intent": "tutorial", "content_gaps": [], "unique_angles": [], "lsi_keywords": [], "questions_to_answer": [], "ai_visibility_recommendations": [], "competitor_weaknesses": [], "recommended_word_count": 1200, "schema_types": ["FAQ"]}')
    competitors = [
        {"url": "https://one.com", "text": "NumPy is fast.", "entities": ["NumPy"], "attributes": ["fast"]},
        {"url": "https://two.com", "text": "NumPy is lightweight.", "entities": ["NumPy"], "attributes": ["lightweight"]},
        {"url": "https://three.com", "text": "NumPy is beginner-friendly.", "entities": ["NumPy"], "attributes": ["beginner-friendly"]},
    ]

    result = research_competitors(client, "numpy tutorial", competitors, "")

    assert result["source_count"] == 3
    assert result["source_urls"] == ["https://one.com", "https://two.com", "https://three.com"]
    assert result["entity_frequency_table"]["NumPy"]["coverage"] == "3/3"


def test_listicle_detector_catches_library_and_tool_keywords():
    assert _looks_like_listicle_topic("Python libraries for data science") is True
    assert _looks_like_listicle_topic("best tools for SEO") is True
    assert _looks_like_listicle_topic("how to write a blog post") is False


def test_sparse_outline_detection_flags_comprehensive_topics():
    outline = {
        "outline_a": {"sections": [{"heading": "Intro", "level": "H2"}]},
        "outline_b": {"sections": [{"heading": "Intro", "level": "H2"}]},
    }

    assert _needs_retry_for_sparse_outline(outline, "Python libraries for data science", "include more libraries") is True
    assert _needs_retry_for_sparse_outline(outline, "how to write a blog post", "include more detail") is False
