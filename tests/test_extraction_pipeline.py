import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extraction import build_competitor_context


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
