import json
from typing import Dict, List, Tuple

from sops import AI_VISIBILITY_GUIDE, SOPS
from .base import ArticleTypeHandler


STRUCTURE_COMPONENTS = {
    "heading": "H2 with the criterion's real name only (e.g. 'Performance', 'Pricing', 'Ease of Use').",
    "paragraph": "A 2–4 sentence assessment of the product against this criterion.",
    "score": "A rating for this specific criterion (e.g. X/10), with a one-sentence justification, only if supported by research.",
    "bullets": "Three to five concise supporting points for this criterion.",
    "pros_cons": "Short, balanced pros and cons specific to this criterion.",
    "quote": "An optional pull quote or callout stat, only when supported by research.",
}


class ReviewHandler(ArticleTypeHandler):
    def available_components(self) -> List[str]:
        return list(STRUCTURE_COMPONENTS.keys())

    def structure_options(self) -> List[Dict]:
        return [
            {"id": "compact", "label": "Compact", "components": ["heading", "paragraph", "score"]},
            {"id": "detailed", "label": "Detailed", "components": ["heading", "paragraph", "score", "pros_cons"]},
            {"id": "visual", "label": "Visual", "components": ["heading", "paragraph", "bullets", "score"]},
        ]

    def build_skeleton_prompt(
        self, keyword: str, research: Dict, brand_context: str,
        directive: str, variant_label: str,
    ) -> Tuple[str, str]:
        system = f"""You are a senior content strategist building the SKELETON of a review article, not a full outline.
Follow these SOPs: {SOPS}
Use the supplied research as evidence; do not invent unsupported facts. You may add generally established, clearly relevant evaluation criteria the competitors missed, but label them "added" and do not make unverified detailed claims about them. Return only valid JSON.

A review skeleton has exactly three parts:
1. pre_list: 1–2 structural H2 sections before the criteria breakdown — an intro that teases the overall verdict up front (what it is, who it's for, and the headline verdict), before the detailed evaluation.
2. the_list: every criterion the product/service/tool should be evaluated against (e.g. performance, pricing, ease of use, support, features), one H2 each. Each item has only name, why_included, and source.
3. post_list: 1–3 structural H2 sections after the criteria breakdown — a pros/cons summary, an alternatives/comparison section, and a final verdict or overall score section.

Every the_list name must be the criterion's real name only—never a question or generic phrase. Do not produce content briefs, keywords, key points, or question-style headings for criteria. Brand context: {brand_context}"""
        user = f"""Keyword: {keyword}
Research: {json.dumps(research)}
Custom directive: {directive or 'None'}

Cover every criterion genuinely important for evaluating this product/service — retain strong competitor-covered criteria first, then add valid missing criteria; do not pad with weak, redundant, or off-topic entries.

Generate one skeleton, variant {variant_label}. Suggest a tone.
Recommend a component sequence for the criterion sections based on the research and search intent. Choose only from heading, paragraph, score, bullets, pros_cons, quote.
Return JSON:
{{
  "tone": "string", "tone_rationale": "string",
  "recommended_structure": ["heading", "paragraph", "score"], "structure_rationale": "string",
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
            raise ValueError(f"Unknown review structure component(s): {', '.join(invalid)}")
        component_notes = "\n".join(f"- {component}: {STRUCTURE_COMPONENTS[component]}" for component in structure_template)
        system = f"""You are a senior content strategist expanding an approved review skeleton into a full outline.
Follow these SOPs: {SOPS}
Apply these AI visibility principles: {AI_VISIBILITY_GUIDE}
Use only the supplied research and skeleton. Do not add, remove, rename, or reorder criteria. Return only valid JSON.

Every criterion must follow this exact structure, in order:
{component_notes}

Every expanded section must include evidence_ids: the IDs from the research evidence_index that support its factual claims. Structural pre-list and post-list sections use the fuller outline schema: heading, level, type, key_points, keywords_to_use, from_competitor, from_brand, ai_visibility_note, rationale, word_count_target, content_brief, and evidence_ids. The final post-list verdict section should give a clear overall recommendation or score.
Brand context: {brand_context}"""
        user = f"""Keyword: {keyword}
Approved skeleton: {json.dumps(skeleton)}
Structure template: {json.dumps(structure_template)}
Custom directive: {directive or 'None'}
Research: {json.dumps(research)}

Return JSON with tone, tone_rationale, target_word_count, pre_list_expanded, list_items_expanded, and post_list_expanded.
Each list_items_expanded entry must have name, source, why_included, heading, level="H2", type="listitem", format, content_brief, key_points, keywords_to_use, rationale, evidence_ids, and components (an object keyed by the chosen components)."""
        return system, user
