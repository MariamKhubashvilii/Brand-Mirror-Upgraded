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
    text = item.get("text") or ""
    has_sections = bool(sections)
    has_text = bool(text.strip())
    cache_key = f"{url}:{_hash_sections(sections)}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
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

    source_block = (
        "Sections:\n" + json.dumps(sections, ensure_ascii=False)
        if has_sections
        else "Raw text:\n" + text[:12000]
    )
    prompt = f"""You are an extraction assistant. Enrich the section outline with a short note for each section.
Use only the provided source content. If structured sections are available, use them. If not, infer a best-effort outline from the raw text and default the format to "paragraph" where unclear. If the source does not contain enough information, say 'not enough information in the source content'.
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
  ],
  "entities": ["entity 1", "entity 2"],
  "attributes": ["attribute 1", "attribute 2"]
}}

Brand knowledge: {brand_knowledge}

{source_block}
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
        if isinstance(parsed_outline, list):
            if has_sections and len(parsed_outline) == len(outline):
                for idx, entry in enumerate(parsed_outline):
                    outline[idx].update({
                        "notes": entry.get("notes") or "",
                    })
            elif not has_sections and parsed_outline:
                outline = parsed_outline
        elif not has_sections and has_text:
            outline = [{
                "level": 2,
                "heading": "Overview",
                "format": "paragraph",
                "images": [],
                "has_code": False,
                "notes": "not enough information in the source content",
            }]
        parsed = {
            "intro_summary": parsed.get("intro_summary") or {"topics_mentioned": [], "keywords_used": [], "tone": "neutral"},
            "outline": outline,
            "entities": parsed.get("entities") or [],
            "attributes": parsed.get("attributes") or [],
        }

    _CACHE[cache_key] = parsed
    return parsed


def build_competitor_context(competitor_texts: list[dict], client: Any, brand_knowledge: str = "") -> list[dict]:
    summaries = []
    for item in competitor_texts:
        url = item.get("url") or ""
        has_sections = bool(item.get("sections"))
        raw_text = item.get("text") or ""
        has_text = bool(raw_text.strip())
        if not has_sections and not has_text:
            summary = {
                "intro_summary": {"topics_mentioned": [], "keywords_used": [], "tone": "neutral"},
                "outline": [],
                "entities": [],
                "attributes": [],
            }
        else:
            summary = extract_competitor_summary(client, url, item, brand_knowledge)
        summaries.append({
            "url": url,
            "title": item.get("title") or url,
            "intro_summary": summary.get("intro_summary") if isinstance(summary, dict) else None,
            "outline": summary.get("outline") if isinstance(summary, dict) else [],
            "entities": summary.get("entities") if isinstance(summary, dict) else [],
            "attributes": summary.get("attributes") if isinstance(summary, dict) else [],
            "summary": summary,
            "outline_source": "structured_sections" if has_sections else "inferred_from_paste",
            "raw_text": raw_text,
            "raw_text_preview": raw_text[:2000],
            "full_text": raw_text,
        })
    return summaries
