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

from model_config import DEFAULT_JSON_MODEL, DEFAULT_PROSE_MODEL
from sops import SOPS, AI_VISIBILITY_GUIDE
from verification import compliance_report

# ── Article type classification ─────────────────────────────────────────────
ARTICLE_TYPES = ["listicle", "comparison", "tutorial/how-to", "guide/informational", "review", "other"]

ARTICLE_TYPE_DESCRIPTIONS = {
    "listicle": "A list of distinct items/options/tools/libraries/products under one topic (e.g. 'Best X', 'Top X for Y', 'X libraries for Y', 'Alternatives to X').",
    "comparison": "A head-to-head comparison of two or more named alternatives (e.g. 'X vs Y').",
    "tutorial/how-to": "Step-by-step instructional content teaching a specific process or skill.",
    "guide/informational": "Broad explanatory or conceptual content not centered on a list or a head-to-head comparison.",
    "review": "An evaluation of a single product, service, or tool.",
    "other": "Doesn't clearly fit the above.",
}


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
    return "Use only the provided text and sources. If something is not covered by the source content, say 'not enough information in the source content' rather than inventing details."


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
    system = f"""You are a careful editor. Revise the draft to fix only the listed issues. Keep the prose concise and grounded in the source text.
{_build_guardrail_prompt()}"""
    user = f"""Keyword: {keyword}
Tone: {tone}
Brand context: {brand_context}
Custom directive: {directive or 'None'}
Issues to fix:
- {chr(10).join(issues)}

Current draft:
{draft}

Return only the revised draft."""
    return chat_claude(claude_client, system, user, max_tokens=max_tokens)

# ── Article: research ────────────────────────────────────────────────────────
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
    system = f"""You are a senior SEO strategist. Analyze competitor content and produce a structured research report.
Use these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}

You must also classify the article type. Choose exactly one from this list, based on the keyword and
how competitors have structured their content:
{json.dumps(ARTICLE_TYPE_DESCRIPTIONS, indent=2)}

Return only valid JSON."""
    user = f"""Keyword: {keyword}
Relevant Brand Context: {relevant_brand_context}

Competitor Content:
{competitor_payload}

Return JSON with this structure:
{{
  "keyword": "string",
  "search_intent": "string — what the user actually wants",
  "competitor_patterns": ["pattern1", "pattern2", ...],
  "content_gaps": ["gap1", "gap2", ...],
  "unique_angles": ["angle1", "angle2", ...],
  "lsi_keywords": ["kw1", "kw2", ...],
  "questions_to_answer": ["q1", "q2", ...],
  "ai_visibility_recommendations": ["rec1", "rec2", ...],
  "competitor_weaknesses": ["weakness1", "weakness2", ...],
  "recommended_word_count": 1200,
  "schema_types": ["FAQ", "HowTo", "..."],
  "detected_article_type": "one of: listicle, comparison, tutorial/how-to, guide/informational, review, other",
  "article_type_rationale": "one sentence explaining why this type fits"
}}"""
    payload = chat_json(client, system, user)
    payload["entity_frequency_table"] = frequency_table
    payload["underused_but_important"] = underused
    payload["source_count"] = len(competitor_texts)
    payload["source_urls"] = [c.get("url") or c.get("title") or "unknown" for c in competitor_texts]
    if payload.get("detected_article_type") not in ARTICLE_TYPES:
        payload["detected_article_type"] = "other"
    payload["max_competitor_list_items"] = _estimate_competitor_list_items(competitor_texts)
    return payload

# ── Article: score existing ──────────────────────────────────────────────────
def score_existing_article(client, article_text: str, keyword: str, brand_knowledge: str, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    system = f"""You are an expert SEO editor. Score this article against these SOPs:
{SOPS}
And these AI visibility principles:
{AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}
Return only valid JSON."""
    user = f"""Keyword: {keyword}
Relevant Brand Context: {relevant_brand_context}
Custom Directive: {directive or 'None'}

Article:
{article_text[:5000]}

Return JSON:
{{
  "overall_score": 0-100,
  "sop_scores": {{
    "conciseness": 0-100,
    "active_voice": 0-100,
    "structure": 0-100,
    "directness": 0-100,
    "skimmability": 0-100,
    "cta_presence": 0-100
  }},
  "ai_visibility_score": 0-100,
  "suggestions": [
    {{
      "sop_number": 1,
      "issue": "string",
      "current_text": "short excerpt",
      "suggested_fix": "rewritten version",
      "priority": "high|medium|low"
    }}
  ],
  "strengths": ["str1", "str2"],
  "summary": "2-3 sentence overall assessment"
}}"""
    return chat_json(client, system, user)

# ── Article: generate two outlines ──────────────────────────────────────────
def _outline_sections_are_sparse(outline: dict, keyword: str, directive: str = "", research: dict = None, effective_type: str = None) -> bool:
    if not isinstance(outline, dict):
        return False
    is_listicle = effective_type == "listicle" or _looks_like_listicle_topic(keyword, directive)
    if not is_listicle:
        return False
    threshold = 2
    if research is not None:
        threshold = max(2, _min_listicle_items(research) // 2)
    sections = outline.get("sections") or []
    return isinstance(sections, list) and len(sections) <= threshold


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

    system = f"""You are a senior content strategist. Generate one article outline.
Follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}
Return only valid JSON.

Detected article type for this piece: {effective_type}

Section count is not fixed. Use as many H2 sections as the topic genuinely needs, not a target number.
A thorough outline is often 6-10 H2 sections with 1-3 H3 subsections where relevant, but that's a typical
range, not a ceiling. If the topic legitimately needs more, use more.

Listicle handling: if the keyword or directive suggests a set of items, tools, options, libraries, apps,
platforms, resources, or alternatives, treat it as a comprehensive list topic. Do not cap the list at what
competitors happen to cover. Research the full set of valid options and make each substantial item its own
H2 section. Do not compress several items into one section or bury them inside a single key_points list.
{type_specific_block}
If the custom directive asks to include more items, more libraries/tools/options, or to exceed competitor
coverage, each additional item must become its own H2 section — do not compress multiple items into the
key_points or content_brief of a single section.

Be critical of the competitor research, not deferential to it. Where competitor articles are bloated with
filler, tangents, or repeated points, do not mirror that bloat. For each candidate section or list item,
judge on its own merits whether it's actually optimal for AI search visibility, SEO, and the reader, then
keep, merge, or cut it. Competitor coverage is a signal to weigh, not a template to copy.

Base the structure on the competitor research and search intent, filtered through your own editorial
judgment, not on any default template.

If a custom directive is given below, it overrides any instruction above it that it conflicts with. Treat
it as the final word on structure, scope, section count, and list length for this outline.

If this is a list/comparison/comprehensive topic, prioritize breadth and coverage over brevity, and make sure
the outline contains enough H2 sections to cover the full set of distinct items rather than a single summary
section.
{section_schema_block}
"""

    def build_user(variant_label: str, contrast_note: str = "") -> str:
        return f"""Keyword: {keyword}
Relevant Brand Context: {relevant_brand_context}
Research: {json.dumps(research)}
Custom Directive for Outline stage (overrides the system instructions above wherever they conflict): {directive or 'None'}

Generate ONE outline — "Outline {variant_label}". Suggest a tone based on the research.{contrast_note}
If the topic is list/comparison/comprehensive, make sure each major item or tool has its own H2 section, not a bullet list inside one section.{' Use the compact listitem schema for items and the full schema only for structural sections, as instructed above.' if is_listicle else ''}
Return JSON:
{{
  "tone": "string",
  "tone_rationale": "why this tone fits based on research",
  "target_word_count": 1400,
  "sections": [
    {{
      "heading": "string",
      "level": "H2",
      "type": "intro|body|faq|cta|conclusion",
      "key_points": ["point1", "point2", "point3", "point4", "point5"],
      "keywords_to_use": [
        {{"keyword": "kw1", "source": "competitor 1 — used in their H2", "why": "high frequency, matches search intent"}},
        {{"keyword": "kw2", "source": "competitor 2 — found in body copy", "why": "LSI term, adds semantic coverage"}}
      ],
      "entities": ["specific brands, tools, studies, stats, or names to mention"],
      "from_competitor": "what we took from competitor research",
      "from_brand": "what comes from brand knowledge/voice",
      "ai_visibility_note": "specific AI visibility tactic for this section",
      "rationale": "why this section exists, and why it earned its place over anything cut",
      "word_count_target": 150,
      "content_brief": "2-3 sentences describing exactly what this section should say and feel like — written so a human writer can follow it without guessing",
      "example_sentences": ["An actual example sentence in the brand voice", "Another one if needed"],
      "h3_subsections": [
        {{
          "heading": "string",
          "key_points": ["point1", "point2", "point3"],
          "keywords_to_use": [
            {{"keyword": "kw1", "source": "competitor 1 — used in meta", "why": "directly relevant to subtopic"}},
            {{"keyword": "kw2", "source": "brand knowledge", "why": "aligns with brand USP"}}
          ],
          "entities": ["specific entities for this subsection"],
          "content_brief": "what this subsection covers",
          "word_count_target": 80
        }}
      ]
    }}
    // ^ this full schema is for STRUCTURAL sections only (intro, orientation, post-list, conclusion).
    // For listitem sections in a listicle, use the compact schema instead:
    // {{"heading": "...", "level": "H2", "type": "listitem", "format": "...", "content_brief": "...", "key_points": [...], "keywords_to_use": [...], "rationale": "..."}}
  ]
}}"""

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
            retry_system = f"""{system}

IMPORTANT: The previous response was too sparse for a comprehensive list/comparison topic. Return a much more detailed outline with many H2 sections. Each major item, tool, library, app, or option should receive its own H2 section, using the compact listitem schema described above — do NOT use the full schema for items, that is what caused the previous response to fall short. Do not compress multiple items into one section."""
            retry_user = f"""{user}

The previous answer was too sparse for this topic. Expand the outline substantially. Make each significant item its own H2 section using the compact listitem schema (heading, level, type, format, content_brief, key_points, keywords_to_use, rationale only — no entities, from_competitor, from_brand, ai_visibility_note, example_sentences, or h3_subsections on items). Reserve the full schema only for structural sections. Do not produce only a handful of sections."""
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



# ── Article: draft selected sections ────────────────────────────────────────
def draft_sections(client, claude_client, keyword: str, outline: dict, selected_headings: list[str],
                   brand_knowledge: str, research: dict, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    research_summary = {k: research[k] for k in ['search_intent', 'lsi_keywords', 'questions_to_answer', 'ai_visibility_recommendations'] if k in research}
    drafted_sections = []
    section_state = {"summaries": [], "keywords_used": []}
    for section in [s for s in outline["sections"] if s["heading"] in selected_headings]:
        section_keywords = [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in section.get("keywords_to_use", [])]
        section_state["keywords_used"].extend(section_keywords)
        prompt = f"""You are an expert content writer. Write one article section in markdown.
Follow these SOPs: {SOPS}
Follow these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}

Keyword: {keyword}
Tone: {outline['tone']}
Research Summary: {json.dumps(research_summary)}
Relevant Brand Context: {relevant_brand_context}
Compact state from earlier sections:
- Summaries: {json.dumps(section_state['summaries'])}
- Keywords already used: {json.dumps(section_state['keywords_used'])}

Section heading: {section['heading']}
Section brief: {section.get('content_brief', '')}
Key points: {json.dumps(section.get('key_points', []))}
Keywords to use: {json.dumps(section.get('keywords_to_use', []))}
Custom directive: {directive or 'None'}

Write the section in markdown and keep it focused on the brief. Use only the provided information."""
        content = chat_claude(claude_client, prompt, "Write the requested section only.", max_tokens=6000)
        drafted_sections.append({
            "heading": section["heading"],
            "content": content,
            "sop_notes": ["SOP 1", "SOP 13"],
            "ai_visibility_notes": ["direct answer up front"],
        })
        section_state["summaries"].append(f"{section['heading']}: {content[:120].replace(chr(10), ' ')}")
    return {"drafted_sections": drafted_sections}

# ── Article: final version ───────────────────────────────────────────────────
def generate_final_article(client, claude_client, keyword: str, outline: dict, drafted_sections: list[dict],
                            user_edits: dict, brand_knowledge: str, research: dict, directive: str = "") -> str:
    edits_block = "\n".join(
        f"Section '{h}': User changed to: {t}"
        for h, t in user_edits.items() if t.strip()
    )
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    system = f"""You are an expert content writer producing a final polished article.
Follow these SOPs: {SOPS}
Follow these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}
Write in clean markdown. No preamble.

VOICE EXAMPLES ..."""
    research_summary = {k: research[k] for k in ['lsi_keywords', 'questions_to_answer', 'ai_visibility_recommendations', 'content_gaps'] if k in research}
    user = f"""Keyword: {keyword}
Tone: {outline['tone']}
Relevant Brand Context: {relevant_brand_context}
Full Outline: {json.dumps(outline['sections'])}
Pre-drafted Sections: {json.dumps(drafted_sections)}
User Edits/Feedback on Drafted Sections:
{edits_block or 'None'}
Custom Directive for Final stage: {directive or 'None'}
Research: {json.dumps(research_summary)}

Write the complete final article in markdown. Apply all SOPs and AI visibility principles throughout.
Respect user edits — they reflect the preferred style and content choices."""
    return chat_claude(claude_client, system, user, max_tokens=8000)

# ── Landing Page: analyze sections ──────────────────────────────────────────
def analyze_landing_page_sections(client, page_text: str, page_sections: list[dict],
                                   keyword: str, brand_knowledge: str) -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    system = f"""You are a senior conversion copywriter and SEO strategist.
Analyze a landing page and identify its sections. Return only valid JSON."""
    user = f"""Keyword: {keyword}
Relevant Brand Context: {relevant_brand_context}

Page content:
{page_text[:5000]}

Identify and name each logical section of this landing page (hero, features, description, FAQ, CTA, testimonials, etc.).
Return JSON:
{{
  "detected_sections": [
    {{
      "name": "string",
      "type": "hero|features|description|faq|cta|testimonials|other",
      "current_text_snippet": "first 100 chars of this section",
      "editable_recommendation": "yes|no",
      "reason": "why editable or not"
    }}
  ],
  "overall_assessment": "2-3 sentences on the page's current state"
}}"""
    return chat_json(client, system, user)

# ── Landing Page: generate section suggestions ───────────────────────────────
def generate_lp_suggestions(client, keyword: str, page_text: str, selected_sections: list[str],
                              all_sections: list[dict], brand_knowledge: str,
                              research: dict, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    selected = [s for s in all_sections if s["name"] in selected_sections]
    system = f"""You are a senior conversion copywriter and SEO strategist.
Follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}
Return only valid JSON."""
    user = f"""Keyword: {keyword}
Relevant Brand Context: {relevant_brand_context}
Research: {json.dumps(research) if research else 'Not provided'}
Custom Directive: {directive or 'None'}

Full page for context (do NOT repeat fixed sections):
{page_text[:3000]}

Sections selected for improvement:
{json.dumps(selected)}

For each selected section, give 2 copy variations.
Return JSON:
{{
  "section_suggestions": [
    {{
      "section_name": "string",
      "variation_a": {{
        "copy": "full rewritten copy in markdown",
        "rationale": "why these choices",
        "sop_applied": ["SOP 13", "SOP 30"],
        "ai_visibility_tactic": "string"
      }},
      "variation_b": {{
        "copy": "full rewritten copy in markdown",
        "rationale": "why these choices",
        "sop_applied": ["SOP 13", "SOP 30"],
        "ai_visibility_tactic": "string"
      }}
    }}
  ]
}}"""
    return chat_json(client, system, user, max_tokens=4000)