import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extraction import build_competitor_context
from llm import build_entity_frequency_table, research_competitors


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


def test_build_competitor_context_preserves_section_structure():
    client = FakeClient('{"intro_summary": {"topics_mentioned": ["pricing"], "keywords_used": ["shipping"], "tone": "direct"}, "outline": [{"level": 2, "heading": "Pricing", "format": "table", "images": ["price chart"], "has_code": false, "notes": "Contains pricing details."}] }')

    item = {
        "url": "https://example.com",
        "title": "Example",
        "text": "Pricing details and shipping info",
        "sections": [
            {
                "heading": "Pricing",
                "level": 2,
                "text": "Our pricing is simple.",
                "format": "table",
                "images": ["price chart"],
                "has_code": False,
            }
        ],
    }

    context = build_competitor_context([item], client, "")

    assert len(context) == 1
    assert context[0]["outline"][0]["heading"] == "Pricing"
    assert context[0]["outline"][0]["format"] == "table"
    assert context[0]["outline"][0]["images"] == ["price chart"]
    assert context[0]["outline"][0]["has_code"] is False
    assert context[0]["intro_summary"]["tone"] == "direct"


def test_build_entity_frequency_table_counts_terms_exactly():
    competitors = [
        {"url": "https://one.com", "raw_text": "NumPy is fast and beginner-friendly. NumPy is used widely.", "entities": ["NumPy"], "attributes": ["beginner-friendly", "fast"]},
        {"url": "https://two.com", "raw_text": "NumPy is lightweight and scalable.", "entities": ["NumPy"], "attributes": ["lightweight", "scalable"]},
    ]

    table = build_entity_frequency_table(competitors)

    assert table["NumPy"]["total_mentions"] == 3
    assert table["NumPy"]["coverage"] == "2/2"
    assert table["beginner-friendly"]["total_mentions"] == 1
    assert table["beginner-friendly"]["coverage"] == "1/2"


def test_build_competitor_context_keeps_pasted_competitors_and_marks_outline_source():
    client = FakeClient('{"intro_summary": {"topics_mentioned": [], "keywords_used": [], "tone": "neutral"}, "outline": [{"level": 2, "heading": "Overview", "format": "paragraph", "images": [], "has_code": false, "notes": "Summary"}], "entities": [], "attributes": []}')
    item = {
        "url": "https://paste.example",
        "title": "Paste",
        "text": "This is pasted content with enough words to be processed for research.",
        "sections": [],
    }

    context = build_competitor_context([item], client, "")

    assert len(context) == 1
    assert context[0]["outline_source"] == "inferred_from_paste"
    assert context[0]["raw_text"] == item["text"]


def test_build_entity_frequency_table_uses_full_text_when_available():
    competitors = [{"url": "https://one.com", "full_text": "alpha " * 2500, "entities": ["alpha"]}]

    table = build_entity_frequency_table(competitors)

    assert table["alpha"]["total_mentions"] == 2500


def test_research_competitors_includes_entity_frequency_matrix():
    client = FakeClient('{"search_intent": "tutorial", "content_gaps": [], "unique_angles": [], "lsi_keywords": [], "questions_to_answer": [], "ai_visibility_recommendations": [], "competitor_weaknesses": [], "recommended_word_count": 1200, "schema_types": ["FAQ"]}')
    competitors = [
        {"url": "https://one.com", "text": "NumPy is fast and beginner-friendly.", "entities": ["NumPy"], "attributes": ["beginner-friendly"]},
        {"url": "https://two.com", "text": "NumPy is lightweight.", "entities": ["NumPy"], "attributes": ["lightweight"]},
    ]

    result = research_competitors(client, "numpy tutorial", competitors, "")

    assert result["entity_frequency_table"]["NumPy"]["total_mentions"] == 2
    assert result["entity_frequency_table"]["beginner-friendly"]["coverage"] == "1/2"
    assert "underused_but_important" in result
