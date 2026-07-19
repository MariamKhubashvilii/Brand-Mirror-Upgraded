import json
from typing import Dict, List, Tuple

from sops import AI_VISIBILITY_GUIDE, SOPS
from .base import ArticleTypeHandler


STRUCTURE_COMPONENTS = {
    "heading": "H2 with the item's real name only.",
    "paragraph": "A 2–4 sentence explanation of what it is and why it is included.",
    "table": "A small spec or feature table with 2–6 useful rows.",
    "pros_cons": "Short, balanced pros and cons lists.",
    "bullets": "Three to five concise key-feature bullets.",
    "screenshot": "A placeholder note for an image or screenshot.",
    "quote": "An optional pull quote or callout stat, only when supported by research.",
}


class ListicleHandler(ArticleTypeHandler):
    def structure_options(self) -> List[Dict]:
        return [
            {"id": "compact", "label": "Compact", "components": ["heading", "paragraph", "bullets"]},
            {"id": "detailed", "label": "Detailed", "components": ["heading", "paragraph", "table", "pros_cons"]},
            {"id": "visual", "label": "Visual", "components": ["heading", "screenshot", "paragraph", "bullets"]},
        ]

    def build_skeleton_prompt(
        self, keyword: str, research: Dict, brand_context: str,
        directive: str, variant_label: str,
    ) -> Tuple[str, str]:
        system = f"""You are a senior content strategist building the SKELETON of a listicle, not a full outline.
Follow these SOPs: {SOPS}
Use only the supplied research; do not invent unsupported facts. Return only valid JSON.

A listicle skeleton has exactly three parts:
1. pre_list: 1–3 structural H2 sections before the list.
2. the_list: every item in the list. Each item has only name, why_included, and source.
3. post_list: 1–3 structural H2 sections after the list.

Every the_list name must be the item's real name only—never a question or generic phrase. Do not produce content briefs, keywords, key points, or question-style list-item headings. Brand context: {brand_context}"""
        user = f"""Keyword: {keyword}
Research: {json.dumps(research)}
Custom directive: {directive or 'None'}

Generate one skeleton, variant {variant_label}. Suggest a tone.
Return JSON:
{{
  "tone": "string", "tone_rationale": "string",
  "pre_list": [{{"heading": "string", "rationale": "string"}}],
  "the_list": [{{"name": "string", "why_included": "string", "source": "competitor|added"}}],
  "post_list": [{{"heading": "string", "rationale": "string"}}]
}}"""
        return system, user

    def build_expansion_prompt(
        self, keyword: str, skeleton: Dict, structure_template: List[str],
        research: Dict, brand_context: str, directive: str,
    ) -> Tuple[str, str]:
        invalid = [component for component in structure_template if component not in STRUCTURE_COMPONENTS]
        if invalid:
            raise ValueError(f"Unknown listicle structure component(s): {', '.join(invalid)}")
        component_notes = "\n".join(f"- {component}: {STRUCTURE_COMPONENTS[component]}" for component in structure_template)
        system = f"""You are a senior content strategist expanding an approved listicle skeleton into a full outline.
Follow these SOPs: {SOPS}
Apply these AI visibility principles: {AI_VISIBILITY_GUIDE}
Use only the supplied research and skeleton. Do not add, remove, rename, or reorder list items. Return only valid JSON.

Every item must follow this exact structure, in order:
{component_notes}

Every expanded section must include evidence_ids: the IDs from the research evidence_index that support its factual claims. Structural pre-list and post-list sections use the fuller outline schema: heading, level, type, key_points, keywords_to_use, from_competitor, from_brand, ai_visibility_note, rationale, word_count_target, content_brief, and evidence_ids.
Brand context: {brand_context}"""
        user = f"""Keyword: {keyword}
Approved skeleton: {json.dumps(skeleton)}
Structure template: {json.dumps(structure_template)}
Custom directive: {directive or 'None'}
Research: {json.dumps(research)}

Return JSON with tone, tone_rationale, target_word_count, pre_list_expanded, list_items_expanded, and post_list_expanded.
Each list_items_expanded entry must have name, source, why_included, heading, level="H2", type="listitem", format, content_brief, key_points, keywords_to_use, rationale, evidence_ids, and components (an object keyed by the chosen components)."""
        return system, user
