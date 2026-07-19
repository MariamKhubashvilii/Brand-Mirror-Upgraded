import openai
import anthropic
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

def chat_claude(claude_client, system, user, max_tokens=8000):
    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return resp.content[0].text.strip()

def chat_claude_json(claude_client, system, user, max_tokens=8000):
    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    raw = resp.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def chat_json(client, system, user, temperature=0.4, max_tokens=8000):
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
Brand Knowledge: {brand_knowledge}
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
    sel_sections = [s for s in outline["sections"] if s["heading"] in selected_headings]
    research_summary = {k: research[k] for k in ['search_intent','lsi_keywords','questions_to_answer','ai_visibility_recommendations'] if k in research}
    system = f"""You are an expert content writer. Write the selected article sections.
Follow these SOPs: {SOPS}
Follow these AI visibility principles: {AI_VISIBILITY_GUIDE}

Return only valid JSON."""
    user = f"""Keyword: {keyword}
Tone: {outline['tone']}
Brand Knowledge: {brand_knowledge}
Research Summary: {json.dumps(research_summary)}

Custom Directive for Drafting stage (overrides the SOPs above wherever they conflict): {directive or 'None'}

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
    return chat_claude_json(claude_client, system, user, max_tokens=8000)

# ── Article: final version ───────────────────────────────────────────────────
def generate_final_article(client, claude_client, keyword: str, outline: dict, drafted_sections: list[dict],
                            user_edits: dict, brand_knowledge: str, research: dict, directive: str = "") -> str:
    edits_block = "\n".join(
        f"Section '{h}': User changed to: {t}"
        for h, t in user_edits.items() if t.strip()
    )
    system = f"""You are an expert content writer producing a final polished article.
Follow these SOPs: {SOPS}
Follow these AI visibility principles: {AI_VISIBILITY_GUIDE}
Write in clean markdown. No preamble.

VOICE EXAMPLES - match this tone and style exactly:

---
You have decided to learn data science. Now you are staring at two names that keep coming up: DataCamp and Coursera. Both teach data skills. Both have Python courses. Both have thousands of students. So which one is actually worth your time?

The honest answer is it depends on how you learn. This guide breaks down what each platform does well, where they fall short, and who they are actually built for.
---
DataCamp puts you in the code from day one. There are no long video lectures to sit through. You read a short explanation, then write actual code in the browser to move forward. That is the whole model.

It works well if you learn by doing. It works less well if you like understanding the theory before you touch anything.
---
DataCamp runs on a subscription, around $25/month if you pay annually. That gets you access to everything. Coursera works differently. Individual courses are free to audit, but certificates cost money, usually $49 per course or $59/month for Coursera Plus.

If you want one platform and a clear learning path, DataCamp is simpler. If you want to pick specific courses or earn university-backed certificates, Coursera makes more sense.
---

NEVER do this:
- Bold random phrases mid-sentence
- Use em dashes. Use a comma or a period instead.
- Use words like: significantly, comprehensive, leverage, dive deep, innovative, seamless, robust, transformative, cutting-edge, powerful, elevate, unlock
- Start sentences with "When selecting", "It is important to", "In conclusion", "In today's"
- State the obvious
- Repeat the same idea in different words"""
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
    return chat_claude(claude_client, system, user, max_tokens=8000)

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