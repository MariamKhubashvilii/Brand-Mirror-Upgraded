import json
import re
from typing import Any, Dict, List, Optional

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency in tests
    openai = None

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency in tests
    anthropic = None

from model_config import DEFAULT_JSON_MODEL, DEFAULT_PROSE_MODEL, QUALITY_CHECK_MODEL
from article_types.registry import get_handler
from sops import SOPS, AI_VISIBILITY_GUIDE, SOP_BANNED_WORDS
from verification import compliance_report, summarize_diff
from prompts import (
    ARTICLE_TYPES,
    ARTICLE_TYPE_DESCRIPTIONS,
    build_guardrail_prompt,
    build_evidence_synthesis_prompts,
    build_research_competitors_system_prompt,
    build_research_competitors_user_prompt,
    build_score_existing_article_system_prompt,
    build_score_existing_article_user_prompt,
    build_outline_system_prompt,
    build_outline_user_prompt,
    build_outline_retry_system_prompt,
    build_outline_retry_user_prompt,
    build_draft_sections_prompt,
    build_draft_revision_prompt,
    build_final_article_prompt,
    build_landing_page_analysis_prompt,
    build_landing_page_suggestions_prompt,
    build_compliance_revision_prompt,
    build_brand_voice_check_prompt,
)


def _estimate_competitor_list_items(competitor_texts: list[dict]) -> int:
    """Heuristic: estimate how many list items (e.g. libraries, tools, products) the
    largest competitor covers, based on numbered headings or H3 subheading counts."""
    counts = []
    for c in competitor_texts:
        outline = []
        if isinstance(c, dict):
            outline = c.get("outline") or (c.get("summary") or {}).get("outline") or []
        numbered = [o for o in outline if re.match(r"^\s*\d+[\.\)]", str(o.get("heading", "")))]
        if numbered:
            counts.append(len(numbered))
            continue
        h3s = [o for o in outline if o.get("level") == 3]
        if h3s:
            counts.append(len(h3s))
    return max(counts) if counts else 0


def chat(client, system, user, temperature=0.5, max_tokens=4000, model: Optional[str] = None):
    if not client:
        raise RuntimeError("No client available")
    resp = client.chat.completions.create(
        model=model or DEFAULT_JSON_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def chat_claude(claude_client, system, user, max_tokens=8000, model: Optional[str] = None):
    if not claude_client:
        raise RuntimeError("No Claude client available")
    resp = claude_client.messages.create(
        model=model or DEFAULT_PROSE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return resp.content[0].text.strip()


def chat_claude_json(claude_client, system, user, max_tokens=8000):
    raw = chat_claude(claude_client, system, user, max_tokens=max_tokens)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def chat_json(client, system, user, temperature=0.4, max_tokens=8000, model: Optional[str] = None):
    if not client:
        raise RuntimeError("No client available")
    resp = client.chat.completions.create(
        model=model or DEFAULT_JSON_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    return json.loads(raw)


def extract_banned_words(brand_knowledge: str) -> List[str]:
    text = brand_knowledge.lower()
    if "banned words" not in text:
        return []
    match = re.search(r"banned words\s*[:\-]\s*(.+)", text)
    if not match:
        return []
    parts = [p.strip() for p in match.group(1).split(",") if p.strip()]
    return [p.strip(" .") for p in parts]


def select_relevant_context(topic: str, brand_knowledge: str, sop_text: str = SOPS) -> str:
    topic_tokens = {t for t in re.split(r"[^a-z0-9]+", topic.lower()) if t}
    if not topic_tokens:
        return brand_knowledge.strip() or sop_text.strip()

    relevant_lines = []
    for source_name, source_text in [("brand", brand_knowledge), ("sop", sop_text)]:
        for line in source_text.splitlines():
            line_lower = line.lower()
            if not line.strip():
                continue
            if any(token and token in line_lower for token in topic_tokens):
                relevant_lines.append(line.strip())
            elif source_name == "brand" and any(marker in line_lower for marker in ["voice", "audience", "usp", "cta", "banned", "brand"]):
                relevant_lines.append(line.strip())
    return "\n".join(dict.fromkeys(relevant_lines))


def build_entity_frequency_table(competitors: List[dict]) -> Dict[str, dict]:
    terms = []
    for competitor in competitors:
        summary = competitor.get("summary") if isinstance(competitor.get("summary"), dict) else {}
        terms.extend(competitor.get("entities") or summary.get("entities") or [])
        terms.extend(competitor.get("attributes") or summary.get("attributes") or [])
    unique_terms = sorted({term.strip() for term in terms if term and str(term).strip()}, key=str.lower)
    total_competitors = len(competitors)
    rows: Dict[str, dict] = {}
    for term in unique_terms:
        per_competitor = {}
        total_mentions = 0
        coverage = 0
        pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
        for competitor in competitors:
            raw_text = competitor.get("full_text") or competitor.get("raw_text") or competitor.get("text") or ""
            count = len(pattern.findall(raw_text))
            per_competitor[competitor.get("url") or "competitor"] = count
            total_mentions += count
            if count > 0:
                coverage += 1
        rows[term] = {
            "total_mentions": total_mentions,
            "per_competitor": per_competitor,
            "coverage": f"{coverage}/{total_competitors}",
        }
    return dict(sorted(rows.items(), key=lambda item: (-item[1]["total_mentions"], item[0].lower())))


def judge_underused_terms(keyword: str, frequency_table: Dict[str, dict], client=None) -> List[dict]:
    if not client:
        return []
    compact_table = []
    for term, values in frequency_table.items():
        compact_table.append({
            "term": term,
            "total_mentions": values.get("total_mentions", 0),
            "coverage": values.get("coverage", "0/0"),
        })
    system = "You are a careful SEO analyst. Flag terms that are underused by competitors but important to the topic. Return only valid JSON."
    user = f"""Keyword/topic: {keyword}
Frequency table (term, total_mentions, coverage):
{json.dumps(compact_table[:80])}

Pick a short list of terms where coverage is low but the term is conceptually central to the topic based on general knowledge of the space. Explain briefly why each is important."""
    payload = chat_json(client, system, user, max_tokens=2500)
    return payload.get("underused_but_important", [])


def _build_guardrail_prompt() -> str:
    return build_guardrail_prompt()


def _looks_like_listicle_topic(keyword: str, directive: str = "") -> bool:
    text = f"{keyword}\n{directive}".lower()
    if not text.strip():
        return False
    listicle_markers = [
        " best ", " top ", " alternatives to ", " options ", " tools ", " libraries ", " apps ", " platforms ", " resources ",
        " for beginners", " for seo", " for data science", " for y", " for x", "vs", " versus ", " compare ", " comparison ",
    ]
    if any(marker in text for marker in listicle_markers):
        return True
    if re.search(r"\b(?:libraries|tools|apps|platforms|resources|options|alternatives)\b", text):
        return True
    return False


def _min_listicle_items(research: dict) -> int:
    competitor_max = (research or {}).get("max_competitor_list_items", 0) or 0
    return max(competitor_max + 2, 22)


def _needs_retry_for_sparse_outline(outlines: dict, keyword: str, directive: str = "") -> bool:
    if not isinstance(outlines, dict):
        return False
    for key in ("outline_a", "outline_b"):
        outline = outlines.get(key)
        if _outline_sections_are_sparse(outline, keyword, directive):
            return True
    return False


def _apply_compliance_loop(text: str, rules: Dict[str, Any], *, keyword: str, tone: str, brand_context: str, directive: str, writer_fn, client, claude_client, max_tokens: int = 4000) -> tuple[str, Dict[str, Any]]:
    draft = text
    report = compliance_report(draft, rules)
    for _ in range(2):
        if not report["missing_keywords"] and not report["banned_words_found"] and not report["structural_issues"]:
            return draft, report
        issues = []
        issues.extend([f"missing keyword: {kw}" for kw in report["missing_keywords"]])
        issues.extend([f"banned word found: {bw}" for bw in report["banned_words_found"]])
        issues.extend(report["structural_issues"])
        draft = writer_fn(
            client=client,
            claude_client=claude_client,
            keyword=keyword,
            tone=tone,
            brand_context=brand_context,
            directive=directive,
            draft=draft,
            issues=issues,
            max_tokens=max_tokens,
        )
        report = compliance_report(draft, rules)
    return draft, report


def _revise_with_targeted_prompt(client, claude_client, keyword: str, tone: str, brand_context: str, directive: str, draft: str, issues: List[str], max_tokens: int = 4000) -> str:
    # Deliberately the cheap/fast model, not chat_claude — this is a targeted mechanical
    # fix-up pass (missing keywords, banned words, structure), not creative drafting.
    system, user = build_compliance_revision_prompt(keyword, tone, brand_context, directive, issues, draft)
    return chat(client, system, user, max_tokens=max_tokens, model=QUALITY_CHECK_MODEL)


def _build_compliance_rules(keyword: str, outline: Optional[Dict], research: Optional[Dict], brand_knowledge: str) -> Dict[str, Any]:
    """Populate the rules dict compliance_report() expects, from the article's own
    topic, research, and brand context — there's no fixed rule set to point at."""
    research = research or {}
    outline = outline or {}
    lsi_terms = [
        term.get("keyword", term) if isinstance(term, dict) else term
        for term in (research.get("lsi_keywords") or [])
    ][:6]
    keywords = list(dict.fromkeys([keyword, *[term for term in lsi_terms if term]]))
    banned_words = list(dict.fromkeys([*SOP_BANNED_WORDS, *extract_banned_words(brand_knowledge)]))
    structural_headings = [
        section.get("heading") for section in outline.get("sections", [])
        if isinstance(section, dict) and section.get("type") != "listitem" and section.get("heading")
    ]
    target_word_count = outline.get("target_word_count") or 1200
    return {
        "keywords": keywords,
        "banned_words": banned_words,
        "min_words": max(300, int(target_word_count * 0.6)),
        "required_headings": structural_headings,
        "require_faq": any("faq" in str(heading).lower() for heading in structural_headings),
    }


def check_brand_voice(client, keyword: str, draft: str, brand_context: str, directive: str = "") -> Dict[str, Any]:
    """Cheap-model pass comparing the draft against brand voice rules, with light edits applied where clearly needed."""
    system, user = build_brand_voice_check_prompt(keyword, brand_context, draft, directive)
    result = chat_json(client, system, user, max_tokens=6000, model=QUALITY_CHECK_MODEL)
    if not isinstance(result, dict):
        raise ValueError("Brand voice check returned invalid JSON")
    return {
        "feedback": result.get("feedback", []),
        "revised_draft": result.get("revised_draft") or draft,
    }


def run_quality_check(client, claude_client, keyword: str, draft: str, outline: dict,
                       research: dict, brand_knowledge: str, directive: str = "") -> Dict[str, Any]:
    """Opt-in check-and-fix pass over a finished draft: deterministic SOP/structural
    compliance first, then a brand-voice pass. Both run on the cheap structured-task
    model — never the Sonnet call used for main drafting. Never runs automatically."""
    brand_context = select_relevant_context(keyword, brand_knowledge)
    tone = outline.get("tone", "") if outline else ""
    rules = _build_compliance_rules(keyword, outline, research, brand_knowledge)

    after_compliance, compliance = _apply_compliance_loop(
        draft, rules, keyword=keyword, tone=tone, brand_context=brand_context,
        directive=directive, writer_fn=_revise_with_targeted_prompt,
        client=client, claude_client=claude_client, max_tokens=6000,
    )

    voice = check_brand_voice(client, keyword, after_compliance, brand_context, directive)
    after_voice = voice["revised_draft"]

    return {
        "original": draft,
        "after_compliance": after_compliance,
        "after_voice": after_voice,
        "compliance_report": compliance,
        "voice_feedback": voice["feedback"],
        "diff_compliance": summarize_diff(draft, after_compliance),
        "diff_voice": summarize_diff(after_compliance, after_voice),
        "rules_applied": rules,
    }

# ── Article: research ────────────────────────────────────────────────────────
def synthesize_competitor_evidence(client, keyword: str, competitor_texts: list[dict], brand_knowledge: str) -> dict:
    """Build a source-linked, cross-competitor dossier before strategic research."""
    evidence_packs = []
    next_evidence_id = 1
    for competitor in competitor_texts:
        source_url = competitor.get("url") or competitor.get("title") or "unknown"
        entries = competitor.get("evidence") or (competitor.get("summary") or {}).get("evidence") or []
        normalized = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            statement, snippet = entry.get("statement"), entry.get("snippet")
            if not statement or not snippet:
                continue
            evidence_id = f"ev_{next_evidence_id:04d}"
            next_evidence_id += 1
            normalized.append({
                "id": evidence_id, "kind": entry.get("kind", "claim"),
                "subject": entry.get("subject", ""), "statement": statement,
                "snippet": snippet, "section": entry.get("section", "Overview"),
            })
        evidence_packs.append({"source_url": source_url, "evidence": normalized})

    brand_context = select_relevant_context(keyword, brand_knowledge)
    system, user = build_evidence_synthesis_prompts(keyword, evidence_packs, brand_context)
    synthesis = chat_json(client, system, user, max_tokens=8000)
    if not isinstance(synthesis, dict):
        raise ValueError("Evidence synthesis returned invalid JSON")
    # Keep the exact source-backed records even if the model omits or rewrites its index.
    source_index = {
        entry["id"]: {
            "source_url": pack["source_url"], "section": entry["section"],
            "snippet": entry["snippet"], "kind": entry["kind"], "statement": entry["statement"],
        }
        for pack in evidence_packs for entry in pack["evidence"]
    }
    synthesis["evidence_index"] = source_index
    synthesis["source_evidence_packs"] = evidence_packs
    return synthesis


def research_competitors(client, keyword, competitor_texts: list[dict], brand_knowledge: str) -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    comps = []
    for i, c in enumerate(competitor_texts, start=1):
        if isinstance(c, dict) and isinstance(c.get("summary"), dict):
            summary = c["summary"]
            outline = c.get("outline") or summary.get("outline") or []
            intro_summary = c.get("intro_summary") or summary.get("intro_summary") or {}
            comps.append(
                f"COMPETITOR {i} ({c.get('url', 'unknown')}):\n"
                f"INTRO: {json.dumps(intro_summary)}\n"
                f"OUTLINE: {json.dumps(outline)}"
            )
        else:
            source_text = (c.get("text") or "")[:3000]
            comps.append(f"COMPETITOR {i} ({c.get('url', 'unknown')}):\n{source_text}")
    competitor_payload = "\n\n".join(comps)
    frequency_table = build_entity_frequency_table(competitor_texts)
    underused = judge_underused_terms(keyword, frequency_table, client)
    evidence_synthesis = synthesize_competitor_evidence(client, keyword, competitor_texts, brand_knowledge)
    system = build_research_competitors_system_prompt(relevant_brand_context, competitor_payload)
    user = build_research_competitors_user_prompt(
        keyword, relevant_brand_context, competitor_payload, evidence_synthesis
    )
    payload = chat_json(client, system, user)
    payload["entity_frequency_table"] = frequency_table
    payload["underused_but_important"] = underused
    payload["evidence_synthesis"] = evidence_synthesis
    payload["source_count"] = len(competitor_texts)
    payload["source_urls"] = [c.get("url") or c.get("title") or "unknown" for c in competitor_texts]
    if payload.get("detected_article_type") not in ARTICLE_TYPES:
        payload["detected_article_type"] = "other"
    payload["max_competitor_list_items"] = _estimate_competitor_list_items(competitor_texts)
    return payload

# ── Article: score existing ──────────────────────────────────────────────────
def score_existing_article(client, article_text: str, keyword: str, brand_knowledge: str, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    system = build_score_existing_article_system_prompt(relevant_brand_context)
    user = build_score_existing_article_user_prompt(keyword, relevant_brand_context, article_text, directive)
    return chat_json(client, system, user)

# ── Article: generate two outlines ──────────────────────────────────────────
def _outline_sections_are_sparse(outline: dict, keyword: str, directive: str = "", research: dict = None, effective_type: str = None) -> bool:
    if not isinstance(outline, dict):
        return False
    is_listicle = effective_type == "listicle" or _looks_like_listicle_topic(keyword, directive)
    if not is_listicle:
        return False
    sections = outline.get("sections") or []
    if not isinstance(sections, list):
        return True
    threshold = 2
    if research is not None:
        threshold = max(2, _min_listicle_items(research) // 2)
    if len(sections) <= threshold:
        return True
    # Section count alone can look fine even when the model produced a generic FAQ/Q&A
    # structure instead of one H2 per list item. Catch that shape too.
    question_headings = sum(
        1 for s in sections if isinstance(s, dict) and str(s.get("heading", "")).strip().endswith("?")
    )
    listitem_typed = sum(1 for s in sections if isinstance(s, dict) and s.get("type") == "listitem")
    if listitem_typed < max(2, threshold) and question_headings >= max(3, len(sections) // 2):
        return True
    return False


def _outline_failure_reason(outline: dict, research: dict = None) -> str:
    """Classify why an outline was flagged sparse, so the retry prompt can target the actual problem."""
    sections = outline.get("sections") or [] if isinstance(outline, dict) else []
    threshold = max(2, _min_listicle_items(research) // 2) if research is not None else 2
    if not isinstance(sections, list) or len(sections) <= threshold:
        return "too_few_sections"
    return "faq_style_headings"


def generate_outlines(client, keyword: str, research: dict, brand_knowledge: str, directive: str = "", article_type: str = None) -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    effective_type = article_type or (research or {}).get("detected_article_type") or "other"
    if effective_type not in ARTICLE_TYPES:
        effective_type = "listicle" if _looks_like_listicle_topic(keyword, directive) else "other"
    is_listicle = effective_type == "listicle" or _looks_like_listicle_topic(keyword, directive)

    type_specific_block = ""
    section_schema_block = ""
    if is_listicle:
        min_items = _min_listicle_items(research)
        competitor_max = (research or {}).get("max_competitor_list_items", 0) or 0
        type_specific_block = f"""

LISTICLE-SPECIFIC REQUIREMENTS — this overrides the general section-count guidance above wherever they conflict:
- The largest competitor covers roughly {competitor_max or 'an unclear number of'} list items (libraries, tools, products, options, etc. — whatever the keyword's list is of).
- Your outline must include AT LEAST {min_items} distinct items, each as its own H2 section. Never group multiple items into one section, and never bury an item inside another section's key_points — one item, one H2.
- First, evaluate every item any competitor includes. Keep it only if it is genuinely relevant, current, and not discontinued, off-topic, or a weak filler entry.
- Then research and add additional genuinely relevant items competitors missed, continuing until you reach or exceed the minimum count above using only high-quality, relevant entries. Do not pad with irrelevant or low-quality items just to hit the number. If you genuinely cannot find enough relevant items to reach the minimum, include as many as are truly valid and say so plainly in tone_rationale — do not silently fall short without explanation.
- Structure the outline in this order:
  1. One intro H2 (why this topic/list matters, how the list is organized).
  2. Optionally, one orientation H2 *before* the list starts if it helps the reader use the list well (e.g. how to evaluate/choose between the items, or the categories the list is organized by). Only include this if it's genuinely useful, not as filler.
  3. One H2 per list item (the bulk of the outline), grouped into logical categories if that aids skimmability.
  4. After the list, one or more H2 sections covering angles that matter for this keyword but aren't list items themselves — anything that improves completeness or reader UX (e.g. how the items work together, common pitfalls, integration notes, decision guidance). Include as many of these as are genuinely useful, not just one for the sake of it.
  5. A final "Final Thoughts" / conclusion H2.

CRITICAL — TWO DIFFERENT SECTION SHAPES: with {min_items}+ items, using the full detailed section schema for
every single item makes the response too large to fit and forces you to silently cut the item count short.
To prevent that, use TWO different shapes for "sections" entries:
- "structural" sections (intro, orientation, post-list, conclusion — usually only 3-6 of these total) use the
  FULL schema shown below, with all fields.
- "listitem" sections (one per item in the list — the bulk of the outline) use the COMPACT schema instead:
  {{
    "heading": "the item's name",
    "level": "H2",
    "type": "listitem",
    "format": "paragraph | table | image+text | code+text | mixed — pick whichever best fits how this item should be presented",
    "content_brief": "2-3 sentences: what it is, why it belongs in this list, and the one or two things worth calling out about it",
    "key_points": ["short point 1", "short point 2", "short point 3"],
    "keywords_to_use": ["kw1", "kw2"],
    "rationale": "under 15 words on why this made the cut"
  }}
  Do NOT include entities, from_competitor, from_brand, ai_visibility_note, example_sentences, or
  h3_subsections on listitem sections — leave those fields out entirely for items to keep the response
  compact enough to cover all {min_items}+ items. Save the full schema's richness for the handful of
  structural sections, where it's cheap because there are only a few of them.
"""
        section_schema_block = """
NOTE: for this listicle, "sections" will contain a MIX of "structural" (full schema) and "listitem" (compact
schema, described above) entries. Use the full schema below only for structural sections. Use the compact
listitem schema for every list item — do not use the full schema for items, or you will run out of room
before covering enough items."""

    system = build_outline_system_prompt(relevant_brand_context, effective_type, is_listicle, type_specific_block, section_schema_block)

    def build_user(variant_label: str, contrast_note: str = "") -> str:
        return build_outline_user_prompt(keyword, relevant_brand_context, research, directive, variant_label, contrast_note, is_listicle)

    # gpt-4o-mini has a hard 16384-token output ceiling — requesting more than that raises an API error
    # instead of yielding a longer response, so neither budget below may exceed it. Each outline now gets
    # its own call and its own full budget, instead of splitting one budget across both outlines.
    base_tokens = 16000 if is_listicle else 8000
    retry_tokens = 16384 if is_listicle else 10000

    def generate_one(variant_label: str, contrast_note: str = "") -> dict:
        user = build_user(variant_label, contrast_note)
        outline = chat_json(client, system, user, max_tokens=base_tokens)
        if not isinstance(outline, dict):
            raise ValueError(f"Outline {variant_label} generation returned invalid JSON")

        if _outline_sections_are_sparse(outline, keyword, directive, research=research, effective_type=effective_type):
            retry_system = build_outline_retry_system_prompt(system)
            retry_user = build_outline_retry_user_prompt(user)
            outline = chat_json(client, retry_system, retry_user, max_tokens=retry_tokens)
            if not isinstance(outline, dict):
                raise ValueError(f"Outline {variant_label} retry returned invalid JSON")
        return outline

    outline_a = generate_one("A")
    tone_a = outline_a.get("tone", "")
    tone_a_rationale = outline_a.get("tone_rationale", "")
    contrast_note = (
        f" Outline B must use a genuinely different tone and structural emphasis than Outline A, which used "
        f"tone '{tone_a}' ({tone_a_rationale}). Do not reuse or lightly reword that tone — pick a distinct angle."
        if tone_a else ""
    )
    outline_b = generate_one("B", contrast_note)

    return {"outline_a": outline_a, "outline_b": outline_b}


# ── Article: listicle skeleton → structure → expansion ─────────────────────
def generate_skeletons(client, keyword: str, research: dict, brand_knowledge: str,
                       directive: str = "", article_type: str = "listicle") -> dict:
    """Generate two lightweight, editable skeleton variants for a supported type."""
    handler = get_handler(article_type)
    brand_context = select_relevant_context(keyword, brand_knowledge)
    # Only listicles need the "cover at least N competitor items" minimum — for other
    # types (comparison options, tutorial steps, guide subtopics, review criteria) the
    # right count is whatever is genuinely relevant, not a competitor-derived floor.
    minimum_items = _min_listicle_items(research) if article_type == "listicle" else 1

    def generate_one(label: str, contrast_note: str = "") -> dict:
        system, user = handler.build_skeleton_prompt(
            keyword, research, brand_context, f"{directive}\n{contrast_note}".strip(), label
        )
        skeleton = chat_json(client, system, user, max_tokens=8000)
        if not isinstance(skeleton, dict) or not isinstance(skeleton.get("the_list"), list):
            raise ValueError(f"Skeleton {label} generation returned invalid JSON")
        if len(skeleton["the_list"]) < minimum_items:
            retry_directive = (
                f"{directive}\nIMPORTANT: The previous skeleton had too few list items. "
                f"Return at least {minimum_items} distinct, relevant items; each must be a separate the_list entry. "
                "Do not replace the list with FAQ questions or generic categories."
            )
            system, user = handler.build_skeleton_prompt(keyword, research, brand_context, retry_directive, label)
            skeleton = chat_json(client, system, user, max_tokens=12000)
            if not isinstance(skeleton, dict) or not isinstance(skeleton.get("the_list"), list):
                raise ValueError(f"Skeleton {label} retry returned invalid JSON")
        return skeleton

    skeleton_a = generate_one("A")
    contrast = "Use a genuinely different structural emphasis and tone from variant A."
    skeleton_b = generate_one("B", contrast)
    return {"skeleton_a": skeleton_a, "skeleton_b": skeleton_b, "structure_options": handler.structure_options()}


def expand_outline(client, keyword: str, skeleton: dict, structure_template: list[str],
                   research: dict, brand_knowledge: str, directive: str = "",
                   article_type: str = "listicle") -> dict:
    """Expand an approved skeleton and normalize it to the app's outline schema."""
    handler = get_handler(article_type)
    brand_context = select_relevant_context(keyword, brand_knowledge)
    system, user = handler.build_expansion_prompt(
        keyword, skeleton, structure_template, research, brand_context, directive
    )
    expanded = chat_json(client, system, user, max_tokens=16000)
    if not isinstance(expanded, dict):
        raise ValueError("Outline expansion returned invalid JSON")

    def structural_sections(entries: list[dict], section_type: str) -> list[dict]:
        return [
            {
                "heading": entry.get("heading", "Untitled section"), "level": entry.get("level", "H2"),
                "type": entry.get("type", section_type), "key_points": entry.get("key_points", []),
                "keywords_to_use": entry.get("keywords_to_use", []),
                "from_competitor": entry.get("from_competitor", ""), "from_brand": entry.get("from_brand", ""),
                "ai_visibility_note": entry.get("ai_visibility_note", ""), "rationale": entry.get("rationale", ""),
                "word_count_target": entry.get("word_count_target", 150),
                "content_brief": entry.get("content_brief", ""), "evidence_ids": entry.get("evidence_ids", []),
            }
            for entry in entries if isinstance(entry, dict)
        ]

    approved_items = skeleton.get("the_list", [])
    raw_items = expanded.get("list_items_expanded", [])
    by_name = {item.get("name"): item for item in raw_items if isinstance(item, dict)}
    list_sections = []
    for approved in approved_items:
        if not isinstance(approved, dict):
            continue
        item = by_name.get(approved.get("name"), {})
        list_sections.append({
            "heading": approved.get("name", "Untitled item"), "level": "H2", "type": "listitem",
            "format": item.get("format", "mixed"), "content_brief": item.get("content_brief", ""),
            "key_points": item.get("key_points", []), "keywords_to_use": item.get("keywords_to_use", []),
            "rationale": item.get("rationale", approved.get("why_included", "")),
            "why_included": approved.get("why_included", ""), "source": approved.get("source", "added"),
            "components": item.get("components", {}), "evidence_ids": item.get("evidence_ids", []),
        })
    return {
        "tone": expanded.get("tone", skeleton.get("tone", "")),
        "tone_rationale": expanded.get("tone_rationale", skeleton.get("tone_rationale", "")),
        "target_word_count": expanded.get("target_word_count", 1400),
        "sections": structural_sections(expanded.get("pre_list_expanded", []), "intro")
        + list_sections
        + structural_sections(expanded.get("post_list_expanded", []), "conclusion"),
        "skeleton": skeleton,
        "structure_template": structure_template,
    }



# ── Article: draft selected sections ────────────────────────────────────────
def draft_sections(client, claude_client, keyword: str, outline: dict, selected_headings: list[str],
                   brand_knowledge: str, research: dict, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    research_summary = {k: research[k] for k in ['search_intent', 'lsi_keywords', 'questions_to_answer', 'ai_visibility_recommendations'] if k in research}
    drafted_sections = []
    section_state = {"summaries": [], "keywords_used": []}
    evidence_index = (research.get("evidence_synthesis") or {}).get("evidence_index", {})
    for section in [s for s in outline["sections"] if s["heading"] in selected_headings]:
        section_keywords = [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in section.get("keywords_to_use", [])]
        section_state["keywords_used"].extend(section_keywords)
        section_research = dict(research_summary)
        section_research["source_evidence"] = [
            evidence_index[evidence_id] for evidence_id in section.get("evidence_ids", [])
            if evidence_id in evidence_index
        ]
        system, prompt = build_draft_sections_prompt(keyword, outline, section_research, relevant_brand_context, section, directive, section_state)
        content = chat_claude(claude_client, system, prompt, max_tokens=6000)
        drafted_sections.append({
            "heading": section["heading"],
            "content": content,
            "sop_notes": ["SOP 1", "SOP 13"],
            "ai_visibility_notes": ["direct answer up front"],
        })
        section_state["summaries"].append(f"{section['heading']}: {content[:120].replace(chr(10), ' ')}")
    return {"drafted_sections": drafted_sections}


def revise_drafted_sections(claude_client, keyword: str, outline: dict, drafted_sections: list[dict],
                            user_edits: dict, brand_knowledge: str, research: dict,
                            directive: str) -> dict:
    """Revise the sample sections in place before committing to a full article."""
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    evidence_index = (research.get("evidence_synthesis") or {}).get("evidence_index", {})
    sections_by_heading = {section.get("heading"): section for section in outline.get("sections", [])}
    revised = []
    for drafted in drafted_sections:
        heading = drafted.get("heading", "")
        section = sections_by_heading.get(heading, {"heading": heading})
        source_evidence = [
            evidence_index[evidence_id] for evidence_id in section.get("evidence_ids", [])
            if evidence_id in evidence_index
        ]
        current_draft = user_edits.get(heading, drafted.get("content", ""))
        system, user = build_draft_revision_prompt(
            keyword, outline.get("tone", ""), relevant_brand_context, section,
            source_evidence, current_draft, directive,
        )
        revised.append({**drafted, "content": chat_claude(claude_client, system, user, max_tokens=6000)})
    return {"drafted_sections": revised}

# ── Article: final version ───────────────────────────────────────────────────
def generate_final_article(client, claude_client, keyword: str, outline: dict, drafted_sections: list[dict],
                            user_edits: dict, brand_knowledge: str, research: dict, directive: str = "") -> str:
    edits_block = "\n".join(
        f"Section '{h}': User changed to: {t}"
        for h, t in user_edits.items() if t.strip()
    )
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    research_summary = {k: research[k] for k in ['lsi_keywords', 'questions_to_answer', 'ai_visibility_recommendations', 'content_gaps'] if k in research}
    system, user = build_final_article_prompt(keyword, outline, drafted_sections, edits_block, relevant_brand_context, research_summary, directive)
    return chat_claude(claude_client, system, user, max_tokens=8000)

# ── Landing Page: analyze sections ──────────────────────────────────────────
def analyze_landing_page_sections(client, page_text: str, page_sections: list[dict],
                                   keyword: str, brand_knowledge: str) -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    system, user = build_landing_page_analysis_prompt(keyword, relevant_brand_context, page_text)
    return chat_json(client, system, user)

# ── Landing Page: generate section suggestions ───────────────────────────────
def generate_lp_suggestions(client, keyword: str, page_text: str, selected_sections: list[str],
                              all_sections: list[dict], brand_knowledge: str,
                              research: dict, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    selected = [s for s in all_sections if s["name"] in selected_sections]
    system, user = build_landing_page_suggestions_prompt(keyword, relevant_brand_context, page_text, selected, research, directive)
    return chat_json(client, system, user, max_tokens=4000)
