import openai
import json
from sops import SOPS, AI_VISIBILITY_GUIDE

def chat(client, system, user, temperature=0.5, max_tokens=4000):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def chat_json(client, system, user, temperature=0.4, max_tokens=4000):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    return json.loads(raw)

# ── Article: research ────────────────────────────────────────────────────────
def research_competitors(client, keyword, competitor_texts: list[dict], brand_knowledge: str) -> dict:
    comps = "\n\n".join(
        f"COMPETITOR {i+1} ({c['url']}):\n{c['text'][:3000]}"
        for i, c in enumerate(competitor_texts)
    )
    system = f"""You are a senior SEO strategist. Your job is to analyze competitor content and produce a structured research report.
Use these AI visibility principles: {AI_VISIBILITY_GUIDE}
Return only valid JSON."""
    user = f"""Keyword: {keyword}
Brand Knowledge: {brand_knowledge}

Competitor Content:
{comps}

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
    system = f"""You are an expert SEO editor. Score this article against these SOPs:
{SOPS}
And these AI visibility principles:
{AI_VISIBILITY_GUIDE}
Return only valid JSON."""
    user = f"""Keyword: {keyword}
Brand Knowledge: {brand_knowledge}
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
    system = f"""You are a senior content strategist. Generate two distinct article outlines with different tones.
Follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
Return only valid JSON.
Include as many sections as the topic genuinely needs — do not limit yourself to 2 or 3 headings.
A thorough outline typically has 6-10 H2 sections, each with 1-3 H3 subsections where relevant.
Base the structure entirely on the competitor research and search intent, not on any default template."""
    user = f"""Keyword: {keyword}
Brand Knowledge: {brand_knowledge}
Research: {json.dumps(research)}
Custom Directive for Outline stage: {directive or 'None'}

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
        "rationale": "why this section exists",
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
        "rationale": "why this section exists",
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
    return chat_json(client, system, user)

# ── Article: draft selected sections ────────────────────────────────────────
def draft_sections(client, keyword: str, outline: dict, selected_headings: list[str],
                   brand_knowledge: str, research: dict, directive: str = "") -> dict:
    sel_sections = [s for s in outline["sections"] if s["heading"] in selected_headings]
    system = f"""You are an expert SEO content writer. Write the selected article sections.
Strictly follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
Return only valid JSON."""
    research_summary = {k: research[k] for k in ['search_intent','lsi_keywords','questions_to_answer','ai_visibility_recommendations'] if k in research}

    user = f"""Keyword: {keyword}
Tone: {outline['tone']}
Brand Knowledge: {brand_knowledge}
Research Summary: {json.dumps(research_summary)}

Custom Directive for Drafting stage: {directive or 'None'}

Sections to write:
{json.dumps(sel_sections)}

Return JSON:
{{
  "drafted_sections": [
    {{
      "heading": "string",
      "content": "full markdown content for this section",
      "sop_notes": ["which SOPs were applied and how"],
      "ai_visibility_notes": ["which AI visibility tactics were used"]
    }}
  ]
}}"""
    return chat_json(client, system, user, max_tokens=4000)

# ── Article: final version ───────────────────────────────────────────────────
def generate_final_article(client, keyword: str, outline: dict, drafted_sections: list[dict],
                            user_edits: dict, brand_knowledge: str, research: dict, directive: str = "") -> str:
    edits_block = "\n".join(
        f"Section '{h}': User changed to: {t}"
        for h, t in user_edits.items() if t.strip()
    )
    system = f"""You are an expert SEO content writer producing a final polished article.
Strictly follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
Write in clean markdown. No preamble."""
    research_summary = {k: research[k] for k in ['lsi_keywords','questions_to_answer','ai_visibility_recommendations','content_gaps'] if k in research}
    user = f"""Keyword: {keyword}
Tone: {outline['tone']}
Brand Knowledge: {brand_knowledge}
Full Outline: {json.dumps(outline['sections'])}
Pre-drafted Sections: {json.dumps(drafted_sections)}
User Edits/Feedback on Drafted Sections:
{edits_block or 'None'}
Custom Directive for Final stage: {directive or 'None'}
Research: {json.dumps(research_summary)}

Write the complete final article in markdown. Apply all SOPs and AI visibility principles throughout.
Respect user edits — they reflect the preferred style and content choices."""
    return chat(client, system, user, temperature=0.6, max_tokens=4000)

# ── Landing Page: analyze sections ──────────────────────────────────────────
def analyze_landing_page_sections(client, page_text: str, page_sections: list[dict],
                                   keyword: str, brand_knowledge: str) -> dict:
    system = f"""You are a senior conversion copywriter and SEO strategist.
Analyze a landing page and identify its sections. Return only valid JSON."""
    user = f"""Keyword: {keyword}
Brand Knowledge: {brand_knowledge}

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
    selected = [s for s in all_sections if s["name"] in selected_sections]
    system = f"""You are a senior conversion copywriter and SEO strategist.
Follow these SOPs: {SOPS}
And these AI visibility principles: {AI_VISIBILITY_GUIDE}
Return only valid JSON."""
    user = f"""Keyword: {keyword}
Brand Knowledge: {brand_knowledge}
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
