import json
from typing import Dict, List, Tuple

from sops import AI_VISIBILITY_GUIDE, SOPS
from .base import ArticleTypeHandler


STRUCTURE_COMPONENTS = {
    "heading": "H2 with the step's real, action-oriented name only (e.g. 'Install the CLI', not 'Step 1').",
    "paragraph": "A 2–5 sentence explanation of what to do in this step and why it matters.",
    "code_block": "A placeholder note for a fenced code block or command, only when this step involves code or commands.",
    "screenshot": "A placeholder note for an image or screenshot illustrating this step.",
    "tip": "An optional short callout with a pro tip or warning for this step, only when genuinely useful.",
    "bullets": "Three to five concise sub-points or checklist items for this step.",
}


class TutorialHandler(ArticleTypeHandler):
    def available_components(self) -> List[str]:
        return list(STRUCTURE_COMPONENTS.keys())

    def structure_options(self) -> List[Dict]:
        return [
            {"id": "compact", "label": "Compact", "components": ["heading", "paragraph"]},
            {"id": "detailed", "label": "Detailed", "components": ["heading", "paragraph", "code_block", "tip"]},
            {"id": "visual", "label": "Visual", "components": ["heading", "screenshot", "paragraph", "bullets"]},
        ]

    def build_skeleton_prompt(
        self, keyword: str, research: Dict, brand_context: str,
        directive: str, variant_label: str,
    ) -> Tuple[str, str]:
        system = f"""You are a senior content strategist building the SKELETON of a tutorial/how-to article, not a full outline.
Follow these SOPs: {SOPS}
Use the supplied research as evidence; do not invent unsupported facts. You may add generally established steps competitors missed, but label them "added" and do not make unverified detailed claims about them. Return only valid JSON.

A tutorial skeleton has exactly three parts:
1. pre_list: 1–3 structural H2 sections before the steps — an intro covering what the reader will accomplish and why it matters, and a prerequisites/requirements section listing what's needed before starting.
2. the_list: every numbered step required to complete the process, in the exact order they must be performed. Each item has only name, why_included, and source.
3. post_list: 1–3 structural H2 sections after the steps — a troubleshooting/common-mistakes section, and a wrap-up covering next steps.

Every the_list name must be the step's real, action-oriented title only—never a question or generic phrase like "Step 1". Do not produce content briefs, keywords, key points, or question-style headings for steps. Brand context: {brand_context}"""
        user = f"""Keyword: {keyword}
Research: {json.dumps(research)}
Custom directive: {directive or 'None'}

Include every step genuinely necessary to complete the process correctly, in the correct order — retain strong competitor-covered steps first, then add valid missing steps; do not pad with unnecessary or redundant steps, and do not omit a step required for the process to work.

Generate one skeleton, variant {variant_label}. Suggest a tone.
Recommend a component sequence for the step sections based on the research and search intent. Choose only from heading, paragraph, code_block, screenshot, tip, bullets.
Return JSON:
{{
  "tone": "string", "tone_rationale": "string",
  "recommended_structure": ["heading", "paragraph"], "structure_rationale": "string",
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
            raise ValueError(f"Unknown tutorial structure component(s): {', '.join(invalid)}")
        component_notes = "\n".join(f"- {component}: {STRUCTURE_COMPONENTS[component]}" for component in structure_template)
        system = f"""You are a senior content strategist expanding an approved tutorial skeleton into a full outline.
Follow these SOPs: {SOPS}
Apply these AI visibility principles: {AI_VISIBILITY_GUIDE}
Use only the supplied research and skeleton. Do not add, remove, rename, or reorder steps. Return only valid JSON.

Every step must follow this exact structure, in order:
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
