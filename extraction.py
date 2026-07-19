import hashlib
import json
import re
from typing import Any, Dict, Optional

from model_config import DEFAULT_EXTRACTION_MODEL

_CACHE: Dict[str, Dict[str, Any]] = {}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_competitor_summary(client: Any, url: str, text: str, brand_knowledge: str = "") -> Dict[str, Any]:
    cache_key = f"{url}:{_hash_text(text)}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    prompt = f"""You are an extraction assistant. Convert the provided source text into a compact structured JSON summary.
Use only the provided text. If the source does not contain enough information, say 'not enough information in the source content'.
Return only valid JSON with fields:
{{
  \"headings_used\": [\"heading 1\", \"heading 2\"],
  \"key_claims\": [\"claim 1\", \"claim 2\"],
  \"stats\": [\"stat or number\"],
  \"unique_angles\": [\"angle 1\"],
  \"keywords_entities\": [\"keyword/entity 1\"],
  \"tone_notes\": [\"tone note 1\"],
  \"summary\": \"140-220 words of signal\"
}}

Brand knowledge: {brand_knowledge}

Source text:
{text}
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
    _CACHE[cache_key] = parsed
    return parsed


def build_competitor_context(competitor_texts: list[dict], client: Any, brand_knowledge: str = "") -> list[dict]:
    summaries = []
    for item in competitor_texts:
        text = item.get("text") or ""
        url = item.get("url") or ""
        if not text.strip():
            continue
        summary = extract_competitor_summary(client, url, text, brand_knowledge)
        summaries.append({
            "url": url,
            "title": item.get("title") or url,
            "summary": summary,
            "raw_text": text[:2000],
        })
    return summaries
