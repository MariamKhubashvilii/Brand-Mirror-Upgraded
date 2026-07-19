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


def _build_guardrail_prompt() -> str:
    return "Use only the provided text and sources. If something is not covered by the source content, say 'not enough information in the source content' rather than inventing details."


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
            comps.append(f"COMPETITOR {i} ({c.get('url', 'unknown')}):\n{json.dumps(c['summary'])}")
        else:
            source_text = (c.get("text") or "")[:3000]
            comps.append(f"COMPETITOR {i} ({c.get('url', 'unknown')}):\n{source_text}")
    competitor_payload = "\n\n".join(comps)
    system = f"""You are a senior SEO strategist. Analyze competitor content and produce a structured research report.
Use these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}
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
  "schema_types": ["FAQ", "HowTo", "..."]
}}"""
    return chat_json(client, system, user)

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
def generate_outlines(client, keyword: str, research: dict, brand_knowledge: str, directive: str = "") -> dict:
    relevant_brand_context = select_relevant_context(keyword, brand_knowledge)
    system = f"""You are a senior content strategist. Generate two distinct article outlines with different tones.
Follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
{_build_guardrail_prompt()}
Return only valid JSON.

Section count is not fixed. Use as many H2 sections as the topic genuinely needs, not a target number.
A thorough outline is often 6-10 H2 sections with 1-3 H3 subsections where relevant, but that's a typical
range, not a ceiling. If the topic legitimately needs more, use more.

Listicle handling: if the keyword is list-format (best X, top X, X vs Y vs Z, alternatives to X, etc.),
do not cap the list at what competitors happen to cover. Research the full set of options, tools, or
approaches that are genuinely valid for the topic, including ones no competitor mentions. Then apply a
critical filter before including anything: drop items that don't fit the brand, are discontinued or
outdated, or wouldn't hold up to a knowledgeable reader checking them. The goal is a list that is more
complete and more accurate than any single competitor's, not a superset padded with weak entries.

Be critical of the competitor research, not deferential to it. Where competitor articles are bloated with
filler, tangents, or repeated points, do not mirror that bloat. For each candidate section or list item,
judge on its own merits whether it's actually optimal for AI search visibility, SEO, and the reader, then
keep, merge, or cut it. Competitor coverage is a signal to weigh, not a template to copy.

Base the structure on the competitor research and search intent, filtered through your own editorial
judgment, not on any default template.

If a custom directive is given below, it overrides any instruction above it that it conflicts with. Treat
it as the final word on structure, scope, section count, and list length for this outline."""
    user = f"""Keyword: {keyword}
Relevant Brand Context: {relevant_brand_context}
Research: {json.dumps(research)}
Custom Directive for Outline stage (overrides the system instructions above wherever they conflict): {directive or 'None'}

Generate two outlines. For each, suggest a distinct tone based on the research.
Return JSON:
{{
  "outline_a": {{
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
    ]
  }},
  "outline_b": {{
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
    ]
  }}
}}"""
    return chat_json(client, system, user, max_tokens=8000)

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