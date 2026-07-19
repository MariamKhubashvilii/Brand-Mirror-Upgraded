import hashlib
import json
import logging
from typing import Any, Dict

from model_config import DEFAULT_EXTRACTION_MODEL

_CACHE: Dict[str, Dict[str, Any]] = {}


def _hash_sections(sections: list[dict]) -> str:
    payload = json.dumps(sections, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_competitor_summary(client: Any, url: str, item: dict, brand_knowledge: str = "") -> Dict[str, Any]:
    sections = item.get("sections") or []
    cache_key = f"{url}:{_hash_sections(sections)}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    outline = []
    for section in sections:
        outline.append({
            "level": section.get("level", 2),
            "heading": section.get("heading", ""),
            "format": section.get("format", "paragraph"),
            "images": section.get("images", []),
            "has_code": bool(section.get("has_code", False)),
        })

    if len(outline) != len(sections):
        logging.warning("Outline length mismatch for %s", url)

    prompt = f"""You are an extraction assistant. Enrich the section outline with a short note for each section.
Use only the provided section structure and text. If the source does not contain enough information, say 'not enough information in the source content'.
Return only valid JSON with fields:
{{
  "intro_summary": {{
    "topics_mentioned": ["topic 1"],
    "keywords_used": ["keyword 1"],
    "tone": "string"
  }},
  "outline": [
    {{
      "level": 2,
      "heading": "same heading text",
      "format": "table | list | paragraph | mixed",
      "images": ["alt text"],
      "has_code": false,
      "notes": "one sentence"
    }}
  ]
}}

Brand knowledge: {brand_knowledge}

Sections:
{json.dumps(sections, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=DEFAULT_EXTRACTION_MODEL,
        messages=[{"role": "system", "content": "You are a concise extraction assistant."}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1800,
    )
    payload = response.choices[0].message.content.strip()
    payload = payload.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(payload)

    if isinstance(parsed, dict):
        parsed_outline = parsed.get("outline") or []
        if isinstance(parsed_outline, list) and len(parsed_outline) == len(outline):
            for idx, entry in enumerate(parsed_outline):
                outline[idx].update({
                    "notes": entry.get("notes") or "",
                })
        parsed = {
            "intro_summary": parsed.get("intro_summary") or {"topics_mentioned": [], "keywords_used": [], "tone": "neutral"},
            "outline": outline,
        }

    _CACHE[cache_key] = parsed
    return parsed


def build_competitor_context(competitor_texts: list[dict], client: Any, brand_knowledge: str = "") -> list[dict]:
    summaries = []
    for item in competitor_texts:
        url = item.get("url") or ""
        if not item.get("sections"):
            continue
        summary = extract_competitor_summary(client, url, item, brand_knowledge)
        summaries.append({
            "url": url,
            "title": item.get("title") or url,
            "intro_summary": summary.get("intro_summary") if isinstance(summary, dict) else None,
            "outline": summary.get("outline") if isinstance(summary, dict) else [],
            "summary": summary,
            "raw_text": (item.get("text") or "")[:2000],
        })
    return summaries
