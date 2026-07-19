import re
import streamlit as st
import openai
import anthropic
import json
from scraper import scrape_url, parse_pasted_html
from extraction import build_competitor_context

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None
from llm import (
    research_competitors, score_existing_article, generate_outlines,
    draft_sections, generate_final_article,
    analyze_landing_page_sections, generate_lp_suggestions
)

st.set_page_config(page_title="SEO Writer", page_icon="✍️", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; font-size: 13px; }
h1,h2,h3,.syne { font-family: 'Syne', sans-serif !important; }
.stApp { background: #f5f7fa; color: #1a1a2e; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1300px; }

/* Cards */
.card { background: #ffffff; border: 1px solid #d0d8e8; border-radius: 3px; padding: 1.2rem 1.4rem; margin: 0.6rem 0; box-shadow: 0 1px 3px rgba(30,60,120,0.07); }
.card.accent { border-color: #2a7abf; background: #eef5fc; }
.card.warn { border-color: #7b3fa0; background: #f8f0fc; }
.card.ok { border-color: #1a7a6e; background: #edfaf7; }

/* Tags */
.tag { display:inline-block; border:1px solid #ccc; color:#666; font-size:0.7rem; padding:0.1rem 0.45rem; border-radius:2px; margin:0.1rem; }
.tag.green { border-color:#2a7abf; color:#2a7abf; }
.tag.orange { border-color:#7b3fa0; color:#7b3fa0; }

/* Labels */
.lbl { font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:#6a7a99; margin-bottom:0.3rem; }

/* Score */
.score-big { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; }
.score-big.hi { color:#1a7a6e; }
.score-big.mid { color:#2a7abf; }
.score-big.lo { color:#9b3060; }

/* Step badge */
.step-badge { background:#1a2a4a; color:#f0f4ff; font-family:'Syne',sans-serif; font-weight:700;
  font-size:0.7rem; padding:0.15rem 0.5rem; border-radius:2px; margin-right:0.4rem; }

hr { border-color:#d0d8e8; }

/* Inputs */
.stTextArea textarea { background:#ffffff !important; color:#1a1a2e !important; border-color:#c0cce0 !important; font-family:'DM Mono',monospace !important; font-size:12px !important; }
.stTextInput input { background:#ffffff !important; color:#1a1a2e !important; border-color:#c0cce0 !important; }
.stSidebar { background: #eaeef5 !important; }

/* Buttons — primary */
.stButton > button[kind="primary"] {
  background: #2a7abf !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 3px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
}
.stButton > button[kind="primary"]:hover {
  background: #1a5a9a !important;
  color: #ffffff !important;
}

/* Buttons — secondary */
.stButton > button[kind="secondary"] {
  background: #ffffff !important;
  color: #2a7abf !important;
  border: 1px solid #2a7abf !important;
  border-radius: 3px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
}
.stButton > button[kind="secondary"]:hover {
  background: #eef5fc !important;
  color: #1a5a9a !important;
}

/* Disabled buttons */
.stButton > button:disabled {
  background: #c0cce0 !important;
  color: #8899bb !important;
  border: none !important;
}

p, li, div { color: #1a1a2e; }
.stRadio label, .stCheckbox label { color: #1a1a2e !important; }
.stTextInput label, .stTextArea label, .stSelectbox label { color: #3a4a6a !important; }
caption, .caption, small { color: #6a7a99 !important; }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def sc(v):
    if v >= 70: return "hi"
    if v >= 45: return "mid"
    return "lo"

def get_client():
    return openai.OpenAI(api_key=st.session_state.api_key)

def get_claude_client():
    return anthropic.Anthropic(api_key=st.session_state.anthropic_key)

def init_state(keys):
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state({
    "api_key": "", "anthropic_key": "", "brand_knowledge": "",
    "mode": "article",
    # article
    "keyword": "", "comp_urls": ["","",""],
    "own_url": "", "comp_results": [],
    "research": None, "existing_score": None,
    "existing_suggestions": [], "confirmed_suggestions": {},
    "outlines": None, "chosen_outline": None,
    "drafted": None, "user_edits": {},
    "final_article": "",
    "directive_outline": "", "directive_draft": "", "directive_final": "",
    "outline_feedback": "",
    # landing page
    "lp_url": "", "lp_result": None,
    "lp_sections": None, "lp_selected": [],
    "lp_suggestions": None, "lp_directive": "",
    "lp_research": None,
    # scrape pastes
    "comp_pastes": ["","",""],
    "comp_html_pastes": ["","","],
    "own_html_paste": "",
    "lp_comp_pastes": ["", ""],
    "lp_comp_html_pastes": ["", ""],
    "lp_html_paste": "",
})


def is_usable_paste(text: str, min_words: int = 100) -> bool:
    if not text or not text.strip():
        return False
    return len(re.findall(r"\b\w+\b", text)) >= min_words


def build_research_payload(results, client, brand_knowledge):
    usable = [r for r in results if r.get("text")]
    return build_competitor_context(usable, client, brand_knowledge)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='syne' style='font-size:1.2rem;font-weight:800;color:#1a1a1a;'>SEO WRITER</div>", unsafe_allow_html=True)
    st.markdown("<div class='lbl' style='margin-bottom:1rem;'>AI-Powered Content Tool</div>", unsafe_allow_html=True)

    st.session_state.api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password", placeholder="sk-...")
    st.markdown("---")

    st.session_state.anthropic_key = st.text_input("Anthropic API Key", value=st.session_state.anthropic_key, type="password", placeholder="sk-ant-...")
    st.markdown("---")

    mode = st.radio("Mode", ["Article", "Landing Page"], index=0 if st.session_state.mode == "article" else 1)
    st.session_state.mode = mode.lower().replace(" ", "_")
    st.markdown("---")

    st.markdown("<div class='lbl'>Brand Knowledge</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#777;margin-bottom:0.3rem;'>Voice, audience, USPs, banned words, CTAs</div>", unsafe_allow_html=True)
    st.session_state.brand_knowledge = st.text_area(
        "Brand Knowledge",
        value=st.session_state.brand_knowledge,
        height=220,
        label_visibility="collapsed",
        placeholder="e.g.\nBrand voice: casual, direct, never corporate\nAudience: young men 18-30 into streetwear\nUSPs: below-retail prices, fast shipping\nBanned words: innovative, leverage, utilize\nCTA style: short, action-first ('Shop now', 'Get yours')"
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────
if st.session_state.mode == "article":
    st.markdown("<h1 class='syne' style='font-size:2.2rem;font-weight:800;margin-bottom:0;'>Article Writer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:0.8rem;margin-bottom:1.5rem;'>Research → Outline → Draft → Final</p>", unsafe_allow_html=True)

    # ── STEP 1: Inputs ──────────────────────────────────────────────────────
    st.markdown("<span class='step-badge'>1</span><span class='syne' style='font-size:1rem;font-weight:700;'>Setup</span>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1], gap="large")
    with c1:
        st.session_state.keyword = st.text_input("Target Keyword", value=st.session_state.keyword, placeholder="e.g. custom bumper stickers")
        st.markdown("<div class='lbl' style='margin-top:0.6rem;'>Your page URL (leave blank for new article)</div>", unsafe_allow_html=True)
        st.session_state.own_url = st.text_input("Your URL", value=st.session_state.own_url, placeholder="https://yoursite.com/article (blank = new)", label_visibility="collapsed")
        with st.expander("Already have HTML for your page? Paste it here to skip scraping", expanded=False):
            st.session_state.own_html_paste = st.text_area(
                "HTML for your page",
                value=st.session_state.own_html_paste,
                height=120,
                key="own_html_paste_input",
                label_visibility="collapsed",
                placeholder="Paste full HTML or an Inspect element snippet here"
            )
    with c2:
        st.markdown("<div class='lbl'>Competitor URLs (up to 3)</div>", unsafe_allow_html=True)
        for i in range(3):
            st.session_state.comp_urls[i] = st.text_input(f"Competitor {i+1}", value=st.session_state.comp_urls[i], placeholder=f"https://competitor{i+1}.com/...", label_visibility="collapsed", key=f"curl_{i}")
            with st.expander(f"Already have HTML for Competitor {i+1}?", expanded=False):
                st.session_state.comp_html_pastes[i] = st.text_area(
                    f"HTML for Competitor {i+1}",
                    value=st.session_state.comp_html_pastes[i],
                    height=120,
                    key=f"comp_html_{i}",
                    label_visibility="collapsed",
                    placeholder="Paste full HTML or an Inspect element snippet here"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Scrape & Research", type="primary", disabled=not (st.session_state.api_key and st.session_state.keyword)):
        urls = [u.strip() for u in st.session_state.comp_urls if u.strip()]
        if not urls:
            st.warning("Add at least one competitor URL.")
        else:
            results = []
            failed = []
            with st.spinner("Scraping competitor pages..."):
                for i, url in enumerate(urls):
                    html_paste = st.session_state.comp_html_pastes[i].strip() if i < len(st.session_state.comp_html_pastes) else ""
                    if html_paste:
                        r = parse_pasted_html(html_paste, label=url)
                        r["url"] = url
                    else:
                        r = scrape_url(url)
                    results.append(r)
                    if not r["success"]:
                        failed.append(r)
            st.session_state.comp_results = results

            if failed:
                st.markdown("---")
                st.markdown("<div class='lbl' style='color:#cc4400;'>Could not open these pages — paste content manually below</div>", unsafe_allow_html=True)
                for f in failed:
                    st.markdown(f"<div class='card warn'>⚠ <b>{f['url']}</b><br><span style='color:#888;font-size:0.8rem;'>{f['error']}</span></div>", unsafe_allow_html=True)
                    idx = [r["url"] for r in results].index(f["url"])
                    st.session_state.comp_pastes[idx] = st.text_area(
                        f"Paste content for {f['url']}",
                        value=st.session_state.comp_pastes[idx],
                        height=150, key=f"paste_{idx}",
                        label_visibility="collapsed",
                        placeholder="Paste the page text here..."
                    )

            # Merge pastes into failed results
            for i, r in enumerate(st.session_state.comp_results):
                if not r["success"] and st.session_state.comp_pastes[i].strip():
                    st.session_state.comp_results[i]["text"] = st.session_state.comp_pastes[i]
                    st.session_state.comp_results[i]["success"] = True

            usable = [r for r in st.session_state.comp_results if r.get("text")]
            if usable:
                with st.spinner("Running deep research..."):
                    compressed = build_research_payload(usable, get_client(), st.session_state.brand_knowledge)
                    st.session_state.research = research_competitors(
                        get_client(), st.session_state.keyword, compressed, st.session_state.brand_knowledge
                    )
                    st.session_state.outlines = None
                    st.session_state.drafted = None
                    st.session_state.final_article = ""
                st.success("Research complete.")

    # ── Show failed pastes persistently and allow re-run ─────────────────
    if st.session_state.comp_results:
        failed_after = [r for r in st.session_state.comp_results if not r["success"]]
        if failed_after:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='card warn'>", unsafe_allow_html=True)
            st.markdown("<div class='lbl' style='color:#cc4400;'>These pages could not be scraped. Paste their content below, then re-run research.</div>", unsafe_allow_html=True)
            for r in failed_after:
                idx = st.session_state.comp_results.index(r)
                st.markdown(f"<div style='font-size:0.8rem;color:#cc4400;margin:0.4rem 0 0.3rem;'>&#9888; {r['url']} — {r['error']}</div>", unsafe_allow_html=True)
                paste = st.text_area(
                    f"Paste for {r['url']}",
                    value=st.session_state.comp_pastes[idx],
                    height=140,
                    key=f"paste_show_{idx}",
                    label_visibility="collapsed",
                    placeholder="Paste the page text here..."
                )
                st.session_state.comp_pastes[idx] = paste
            st.markdown("</div>", unsafe_allow_html=True)

            any_paste_filled = any(
                st.session_state.comp_pastes[st.session_state.comp_results.index(r)].strip()
                for r in failed_after
            )
            if any_paste_filled:
                if st.button("Re-run Research with Pasted Content", type="primary", key="rerun_research"):
                    for i, r in enumerate(st.session_state.comp_results):
                        if not r["success"] and i < len(st.session_state.comp_pastes) and st.session_state.comp_pastes[i].strip():
                            paste_text = st.session_state.comp_pastes[i].strip()
                            if is_usable_paste(paste_text):
                                st.session_state.comp_results[i]["text"] = paste_text
                                st.session_state.comp_results[i]["success"] = True
                            else:
                                st.session_state.comp_results[i]["error"] = "Pasted text is too short. Add at least 100 words."
                    usable = [r for r in st.session_state.comp_results if r.get("text")]
                    if usable:
                        with st.spinner("Running deep research with pasted content..."):
                            compressed = build_research_payload(usable, get_client(), st.session_state.brand_knowledge)
                            st.session_state.research = research_competitors(
                                get_client(), st.session_state.keyword, compressed, st.session_state.brand_knowledge
                            )
                            st.session_state.outlines = None
                            st.session_state.drafted = None
                            st.session_state.final_article = ""
                        st.success("Research complete.")
                        st.rerun()

    # ── STEP 2: Research display ────────────────────────────────────────────
    if st.session_state.research:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='step-badge'>2</span><span class='syne' style='font-size:1rem;font-weight:700;'>Research Results</span>", unsafe_allow_html=True)
        r = st.session_state.research
        rc1, rc2, rc3 = st.columns(3, gap="medium")
        with rc1:
            st.markdown(f"<div class='card'><div class='lbl'>Search Intent</div><div style='font-size:0.85rem;color:#333;'>{r.get('search_intent','')}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><div class='lbl'>Content Gaps</div>{''.join(f'<div style=\"font-size:0.82rem;color:#333;padding:0.2rem 0;\">→ {g}</div>' for g in r.get('content_gaps',[]))}</div>", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"<div class='card'><div class='lbl'>LSI Keywords</div>{''.join(f'<span class=\"tag\">{k}</span>' for k in r.get('lsi_keywords',[]))}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><div class='lbl'>Questions to Answer</div>{''.join(f'<div style=\"font-size:0.82rem;color:#333;padding:0.15rem 0;\">• {q}</div>' for q in r.get('questions_to_answer',[]))}</div>", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"<div class='card'><div class='lbl'>AI Visibility Recs</div>{''.join(f'<div style=\"font-size:0.8rem;color:#5a7a00;padding:0.15rem 0;\">✦ {a}</div>' for a in r.get('ai_visibility_recommendations',[]))}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><div class='lbl'>Recommended Schema</div>{''.join(f'<span class=\"tag green\">{s}</span>' for s in r.get('schema_types',[]))}</div>", unsafe_allow_html=True)

        entity_rows = []
        for term, data in r.get('entity_frequency_table', {}).items():
            entity_rows.append({
                'entity/attribute': term,
                'total_mentions': data.get('total_mentions', 0),
                'coverage': data.get('coverage', '0/0'),
                'per_competitor': ', '.join(f'{k}: {v}' for k, v in sorted(data.get('per_competitor', {}).items()))
            })
        if entity_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Entity / attribute frequency matrix"):
                if pd is not None:
                    st.dataframe(pd.DataFrame(entity_rows), width="stretch", hide_index=True)
                else:
                    for row in entity_rows:
                        st.write(row)

        underused = r.get('underused_but_important', [])
        if underused:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Underused but important"):
                for item in underused:
                    st.markdown(f"- **{item.get('term', '')}** ({item.get('coverage', '')}): {item.get('why', '')}")

        # ── STEP 3A: Update existing article ───────────────────────────────
        if st.session_state.own_url.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<span class='step-badge'>3</span><span class='syne' style='font-size:1rem;font-weight:700;'>Score Existing Article</span>", unsafe_allow_html=True)

            st.session_state.directive_outline = st.text_input(
                "Custom directive (optional)", value=st.session_state.directive_outline,
                placeholder="e.g. keep the tone super simple, almost like a FAQ",
                key="dir_score"
            )

            if st.button("Score My Article", disabled=not st.session_state.api_key):
                with st.spinner("Scraping your page..."):
                    own_html = st.session_state.own_html_paste.strip()
                    if own_html:
                        own = parse_pasted_html(own_html, label=st.session_state.own_url)
                        own["url"] = st.session_state.own_url
                    else:
                        own = scrape_url(st.session_state.own_url)
                if not own["success"]:
                    st.error(f"Could not scrape your page: {own['error']}")
                    pasted = st.text_area("Paste your article content here", value="", height=180, placeholder="Paste at least 100 words of article text...")
                    if pasted.strip() and is_usable_paste(pasted):
                        with st.spinner("Scoring against SOPs..."):
                            result = score_existing_article(
                                get_client(), pasted, st.session_state.keyword,
                                st.session_state.brand_knowledge, st.session_state.directive_outline
                            )
                            st.session_state.existing_score = result
                            st.session_state.confirmed_suggestions = {}
                    elif pasted.strip():
                        st.warning("That pasted content is too short. Add at least 100 words before scoring.")
                else:
                    with st.spinner("Scoring against SOPs..."):
                        result = score_existing_article(
                            get_client(), own["text"], st.session_state.keyword,
                            st.session_state.brand_knowledge, st.session_state.directive_outline
                        )
                        st.session_state.existing_score = result
                        st.session_state.confirmed_suggestions = {}

            if st.session_state.existing_score:
                es = st.session_state.existing_score
                sc1, sc2, sc3 = st.columns(3, gap="medium")
                with sc1:
                    v = es.get("overall_score", 0)
                    st.markdown(f"<div class='card'><div class='lbl'>Overall Score</div><div class='score-big {sc(v)}'>{v}/100</div><div style='font-size:0.8rem;color:#555;margin-top:0.5rem;'>{es.get('summary','')}</div></div>", unsafe_allow_html=True)
                with sc2:
                    sop_s = es.get("sop_scores", {})
                    def sop_color(v2):
                        if v2 >= 70: return "#b8f000"
                        if v2 >= 45: return "#ffa040"
                        return "#ff4f4f"
                    rows = "".join(
                        f"<div style='display:flex;justify-content:space-between;padding:0.2rem 0;border-bottom:1px solid #222;'>"
                        f"<span style='color:#555;font-size:0.8rem;'>{k.replace('_',' ').title()}</span>"
                        f"<span style='color:{sop_color(v2)};font-weight:700;font-size:0.82rem;'>{v2}</span></div>"
                        for k, v2 in sop_s.items()
                    )
                    st.markdown(f"<div class='card'><div class='lbl'>SOP Breakdown</div>{rows}</div>", unsafe_allow_html=True)
                with sc3:
                    aiv = es.get("ai_visibility_score", 0)
                    strengths = "".join(f"<div style='font-size:0.8rem;color:#5a7a00;padding:0.1rem 0;'>✓ {s}</div>" for s in es.get("strengths",[]))
                    st.markdown(f"<div class='card'><div class='lbl'>AI Visibility</div><div class='score-big {sc(aiv)}'>{aiv}/100</div>{strengths}</div>", unsafe_allow_html=True)

                st.markdown("<br><div class='lbl'>Suggestions — Confirm or Deny Each</div>", unsafe_allow_html=True)
                for i, sug in enumerate(es.get("suggestions", [])):
                    priority_color = "#ff4f4f" if sug["priority"] == "high" else "#ffa040" if sug["priority"] == "medium" else "#888"
                    confirmed = st.session_state.confirmed_suggestions.get(i)
                    card_cls = "card ok" if confirmed == True else "card warn" if confirmed == False else "card"
                    st.markdown(f"""<div class='{card_cls}'>
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>
  <div><span class='tag' style='border-color:{priority_color};color:{priority_color};'>{sug['priority'].upper()}</span> <span style='font-size:0.8rem;font-weight:600;'>SOP {sug.get('sop_number','')}</span></div>
</div>
<div class='lbl'>Issue</div><div style='font-size:0.82rem;color:#ccc;margin-bottom:0.4rem;'>{sug['issue']}</div>
<div class='lbl'>Current</div><div style='font-size:0.8rem;color:#888;font-style:italic;margin-bottom:0.4rem;'>"{sug.get('current_text','')}"</div>
<div class='lbl'>Suggested Fix</div><div style='font-size:0.82rem;color:#5a7a00;'>{sug['suggested_fix']}</div>
</div>""", unsafe_allow_html=True)
                    col_y, col_n, _ = st.columns([1,1,4])
                    with col_y:
                        if st.button("✓ Confirm", key=f"confirm_{i}"):
                            st.session_state.confirmed_suggestions[i] = True
                            st.rerun()
                    with col_n:
                        if st.button("✗ Deny", key=f"deny_{i}"):
                            st.session_state.confirmed_suggestions[i] = False
                            st.rerun()

        # ── STEP 3B: New article — outlines ────────────────────────────────
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<span class='step-badge'>3</span><span class='syne' style='font-size:1rem;font-weight:700;'>Generate Outlines</span>", unsafe_allow_html=True)
            st.session_state.directive_outline = st.text_input(
                "Outline directive (optional)", value=st.session_state.directive_outline,
                placeholder="e.g. keep both outlines very beginner-friendly",
                key="dir_outline"
            )
            if st.button("Generate 2 Outlines", disabled=not st.session_state.api_key):
                with st.spinner("Generating outlines..."):
                    st.session_state.outlines = generate_outlines(
                        get_client(), st.session_state.keyword,
                        st.session_state.research, st.session_state.brand_knowledge,
                        st.session_state.directive_outline
                    )
                    st.session_state.chosen_outline = None
                    st.session_state.drafted = None
                    st.session_state.final_article = ""

            if st.session_state.outlines:
                outlines = st.session_state.outlines
                oa, ob = outlines.get("outline_a"), outlines.get("outline_b")

                for label, ol in [("A", oa), ("B", ob)]:
                    if not ol: continue
                    chosen = st.session_state.chosen_outline == label
                    card_cls = "card accent" if chosen else "card"
                    st.markdown(f"<div class='{card_cls}'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='syne' style='font-size:1rem;font-weight:700;'>Outline {label} — {ol.get('tone','')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:0.8rem;color:#666;margin-bottom:0.8rem;'>{ol.get('tone_rationale','')}</div>", unsafe_allow_html=True)

                    for sec in ol.get("sections", []):
                        lvl_color = "#b8f000" if sec["level"] == "H2" else "#888"
                        kws = "".join(
                            f'<span class="tag" title="{k.get("source","") } — {k.get("why","")}">{k.get("keyword", k) if isinstance(k, dict) else k}</span>'
                            for k in sec.get("keywords_to_use", [])
                        )
                        st.markdown(f"""<div style='border-left:2px solid {lvl_color};padding-left:0.8rem;margin:0.5rem 0;'>
<div style='font-size:0.88rem;font-weight:600;color:#1a1a1a;'>{sec['heading']} <span style='color:#aaa;font-size:0.7rem;'>{sec['level']}</span></div>
<div style='font-size:0.78rem;color:#666;margin:0.2rem 0;'>{sec.get('rationale','')}</div>
<div style='margin:0.2rem 0;'>{kws}</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem;margin-top:0.3rem;'>
  <div><div class='lbl'>From Competitors</div><div style='font-size:0.75rem;color:#555;'>{sec.get('from_competitor','')}</div></div>
  <div><div class='lbl'>From Brand</div><div style='font-size:0.75rem;color:#555;'>{sec.get('from_brand','')}</div></div>
  <div><div class='lbl'>AI Visibility</div><div style='font-size:0.75rem;color:#5a7a00;'>{sec.get('ai_visibility_note','')}</div></div>
</div></div>""", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                    if st.button(f"Choose Outline {label}", key=f"choose_{label}"):
                        st.session_state.chosen_outline = label
                        st.session_state.drafted = None
                        st.session_state.final_article = ""
                        st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                st.session_state.outline_feedback = st.text_area(
                    "Feedback on outlines (optional)",
                    value=st.session_state.outline_feedback,
                    placeholder="e.g. make outline A shorter, add a comparison section, remove the FAQ from B",
                    height=80,
                    key="outline_fb"
                )
                if st.button("🔄 Regenerate Outlines", disabled=not st.session_state.api_key):
                    with st.spinner("Regenerating outlines..."):
                        st.session_state.outlines = generate_outlines(
                            get_client(), st.session_state.keyword,
                            st.session_state.research, st.session_state.brand_knowledge,
                            directive=st.session_state.outline_feedback
                        )
                        st.session_state.chosen_outline = None
                        st.session_state.drafted = None
                        st.session_state.final_article = ""
                        st.rerun()

                # ── STEP 4: Draft selected sections ────────────────────────
                if st.session_state.chosen_outline:
                    chosen_key = f"outline_{st.session_state.chosen_outline.lower()}"
                    chosen_ol = outlines.get(chosen_key, {})
                    all_headings = [s["heading"] for s in chosen_ol.get("sections", [])]

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<span class='step-badge'>4</span><span class='syne' style='font-size:1rem;font-weight:700;'>Draft Sections</span>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:0.8rem;color:#666;margin-bottom:0.6rem;'>Outline {st.session_state.chosen_outline} selected. Pick which sections to preview first.</div>", unsafe_allow_html=True)

                    selected_to_draft = st.multiselect(
                        "Sections to draft", all_headings, default=all_headings[:2], key="sections_to_draft"
                    )
                    st.session_state.directive_draft = st.text_input(
                        "Draft directive (optional)", value=st.session_state.directive_draft,
                        placeholder="e.g. make the intro extra punchy, very short sentences",
                        key="dir_draft"
                    )

                    if st.button("Draft Selected Sections", disabled=not (st.session_state.api_key and selected_to_draft)):
                        with st.spinner("Writing sections..."):
                            st.session_state.drafted = draft_sections(
                                get_client(), get_claude_client(), st.session_state.keyword, chosen_ol,
                                selected_to_draft, st.session_state.brand_knowledge,
                                st.session_state.research, st.session_state.directive_draft
                            )
                            st.session_state.user_edits = {}

                    if st.session_state.drafted:
                        st.markdown("<br><div class='lbl'>Drafted Sections — Edit inline if needed</div>", unsafe_allow_html=True)
                        for ds in st.session_state.drafted.get("drafted_sections", []):
                            st.markdown(f"<div class='card'><div class='syne' style='font-size:0.95rem;font-weight:700;margin-bottom:0.5rem;'>{ds['heading']}</div>", unsafe_allow_html=True)
                            sop_tags = "".join(f'<span class="tag green">{n}</span>' for n in ds.get("sop_notes",[]))
                            ai_tags = "".join(f'<span class="tag orange">{n}</span>' for n in ds.get("ai_visibility_notes",[]))
                            if sop_tags or ai_tags:
                                st.markdown(f"<div style='margin-bottom:0.5rem;'>{sop_tags}{ai_tags}</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            edited = st.text_area(
                                f"Edit: {ds['heading']}", value=ds["content"],
                                height=180, key=f"edit_{ds['heading']}", label_visibility="collapsed"
                            )
                            st.session_state.user_edits[ds["heading"]] = edited

                        # ── STEP 5: Final article ───────────────────────────
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("<span class='step-badge'>5</span><span class='syne' style='font-size:1rem;font-weight:700;'>Generate Final Article</span>", unsafe_allow_html=True)
                        st.session_state.directive_final = st.text_input(
                            "Final directive (optional)", value=st.session_state.directive_final,
                            placeholder="e.g. add more bucket brigades, make it warmer overall",
                            key="dir_final"
                        )
                        if st.button("✍️ Write Full Article", type="primary", disabled=not st.session_state.api_key):
                            with st.spinner("Writing full article..."):
                                st.session_state.final_article = generate_final_article(
                                    get_client(), get_claude_client(), st.session_state.keyword, chosen_ol,
                                    st.session_state.drafted.get("drafted_sections", []),
                                    st.session_state.user_edits, st.session_state.brand_knowledge,
                                    st.session_state.research, st.session_state.directive_final
                                )

                        if st.session_state.final_article:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("<div class='card accent'>", unsafe_allow_html=True)
                            st.markdown("<div class='syne' style='font-size:1rem;font-weight:700;margin-bottom:0.8rem;'>Final Article</div>", unsafe_allow_html=True)
                            final_edit = st.text_area(
                                "Final Article", value=st.session_state.final_article,
                                height=600, label_visibility="collapsed"
                            )
                            st.session_state.final_article = final_edit
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.download_button(
                                "⬇ Download as .md",
                                data=st.session_state.final_article,
                                file_name=f"{st.session_state.keyword.replace(' ','_')}.md",
                                mime="text/markdown"
                            )

# ── LANDING PAGE MODE ─────────────────────────────────────────────────────────
else:
    st.markdown("<h1 class='syne' style='font-size:2.2rem;font-weight:800;margin-bottom:0;'>Landing Page Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:0.8rem;margin-bottom:1.5rem;'>Scrape → Detect Sections → Select → Optimize</p>", unsafe_allow_html=True)

    lp1, lp2 = st.columns([1,1], gap="large")
    with lp1:
        st.session_state.lp_url = st.text_input("Landing Page URL", value=st.session_state.lp_url, placeholder="https://yoursite.com/product-page")
        with st.expander("Already have HTML for this landing page?", expanded=False):
            st.session_state.lp_html_paste = st.text_area(
                "HTML for landing page",
                value=st.session_state.lp_html_paste,
                height=120,
                key="lp_html_paste_input",
                label_visibility="collapsed",
                placeholder="Paste full HTML or an Inspect element snippet here"
            )
        st.session_state.keyword = st.text_input("Target Keyword", value=st.session_state.keyword, placeholder="e.g. custom bumper stickers", key="lp_kw")
    with lp2:
        st.markdown("<div class='lbl'>Optional: competitor URLs for context</div>", unsafe_allow_html=True)
        for i in range(2):
            st.session_state.comp_urls[i] = st.text_input(f"Competitor {i+1}", value=st.session_state.comp_urls[i], placeholder=f"https://...", label_visibility="collapsed", key=f"lp_curl_{i}")
            with st.expander(f"Already have HTML for Competitor {i+1}?", expanded=False):
                st.session_state.lp_comp_html_pastes[i] = st.text_area(
                    f"HTML for competitor {i+1}",
                    value=st.session_state.lp_comp_html_pastes[i],
                    height=120,
                    key=f"lp_comp_html_{i}",
                    label_visibility="collapsed",
                    placeholder="Paste full HTML or an Inspect element snippet here"
                )

    if st.button("🔍 Scrape & Analyze Page", type="primary", disabled=not (st.session_state.api_key and st.session_state.lp_url)):
        with st.spinner("Scraping your landing page..."):
            lp_html = st.session_state.lp_html_paste.strip()
            if lp_html:
                result = parse_pasted_html(lp_html, label=st.session_state.lp_url)
                result["url"] = st.session_state.lp_url
            else:
                result = scrape_url(st.session_state.lp_url)
        if not result["success"]:
            st.error(f"Could not scrape page: {result['error']}")
            pasted = st.text_area("Paste your landing page content here", value="", height=180, placeholder="Paste at least 100 words of page content...")
            if pasted.strip() and is_usable_paste(pasted):
                result = {"success": True, "text": pasted, "sections": [], "title": st.session_state.lp_url}
            elif pasted.strip():
                st.warning("That pasted content is too short. Add at least 100 words before analyzing.")
        if result.get("success"):
            st.session_state.lp_result = result
            comp_urls = [u.strip() for u in st.session_state.comp_urls[:2] if u.strip()]
            comp_texts = []
            if comp_urls:
                with st.spinner("Scraping competitor pages for context..."):
                    for i, u in enumerate(comp_urls):
                        html_paste = st.session_state.lp_comp_html_pastes[i].strip() if i < len(st.session_state.lp_comp_html_pastes) else ""
                        if html_paste:
                            cr = parse_pasted_html(html_paste, label=u)
                            cr["url"] = u
                        else:
                            cr = scrape_url(u)
                        if cr["success"]:
                            comp_texts.append(cr)
                        else:
                            idx = i
                            paste = st.text_area(f"Paste competitor content for {u}", value=st.session_state.lp_comp_pastes[idx], height=120, key=f"lp_comp_paste_{idx}")
                            st.session_state.lp_comp_pastes[idx] = paste
                            if is_usable_paste(paste):
                                comp_texts.append({"url": u, "text": paste, "title": u, "sections": []})
            if comp_texts:
                with st.spinner("Running competitor research..."):
                    compressed = build_research_payload(comp_texts, get_client(), st.session_state.brand_knowledge)
                    st.session_state.lp_research = research_competitors(
                        get_client(), st.session_state.keyword, compressed, st.session_state.brand_knowledge
                    )
            with st.spinner("Detecting page sections..."):
                st.session_state.lp_sections = analyze_landing_page_sections(
                    get_client(), result["text"], result["sections"],
                    st.session_state.keyword, st.session_state.brand_knowledge
                )
            st.session_state.lp_selected = []
            st.session_state.lp_suggestions = None

    if st.session_state.lp_sections:
        secs = st.session_state.lp_sections
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='step-badge'>2</span><span class='syne' style='font-size:1rem;font-weight:700;'>Detected Sections</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='card' style='margin-bottom:1rem;'><div class='lbl'>Overall Assessment</div><div style='font-size:0.85rem;color:#333;'>{secs.get('overall_assessment','')}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='lbl'>Select sections to optimize (uncheck fixed ones like features)</div>", unsafe_allow_html=True)
        selected = []
        detected = secs.get("detected_sections", [])
        for sec in detected:
            rec = sec.get("editable_recommendation","yes") == "yes"
            type_color = "#b8f000" if rec else "#555"
            checked = st.checkbox(
                f"**{sec['name']}** — {sec['type']}",
                value=rec,
                key=f"lp_sec_{sec['name']}"
            )
            st.markdown(f"<div style='font-size:0.75rem;color:#888;margin:-0.6rem 0 0.4rem 1.8rem;'>{sec.get('reason','')} | <i>{sec.get('current_text_snippet','')[:80]}...</i></div>", unsafe_allow_html=True)
            if checked:
                selected.append(sec["name"])
        st.session_state.lp_selected = selected

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='step-badge'>3</span><span class='syne' style='font-size:1rem;font-weight:700;'>Generate Suggestions</span>", unsafe_allow_html=True)
        st.session_state.lp_directive = st.text_input(
            "Custom directive (optional)", value=st.session_state.lp_directive,
            placeholder="e.g. avoid repeating words used in the features section",
            key="lp_dir"
        )
        if st.button("Generate Copy Suggestions", type="primary",
                     disabled=not (st.session_state.api_key and st.session_state.lp_selected)):
            with st.spinner("Writing optimized copy..."):
                st.session_state.lp_suggestions = generate_lp_suggestions(
                    get_client(), st.session_state.keyword,
                    st.session_state.lp_result["text"],
                    st.session_state.lp_selected,
                    detected, st.session_state.brand_knowledge,
                    st.session_state.lp_research or {},
                    st.session_state.lp_directive
                )

    if st.session_state.lp_suggestions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='step-badge'>4</span><span class='syne' style='font-size:1rem;font-weight:700;'>Copy Suggestions</span>", unsafe_allow_html=True)
        for sugg in st.session_state.lp_suggestions.get("section_suggestions", []):
            st.markdown(f"<div class='syne' style='font-size:1rem;font-weight:700;margin:1.2rem 0 0.5rem;'>{sugg['section_name']}</div>", unsafe_allow_html=True)
            va, vb = sugg.get("variation_a", {}), sugg.get("variation_b", {})
            col_a, col_b = st.columns(2, gap="medium")
            for col, var, label in [(col_a, va, "A"), (col_b, vb, "B")]:
                with col:
                    sop_tags = "".join(f'<span class="tag green">{s}</span>' for s in var.get("sop_applied",[]))
                    st.markdown(f"<div class='card'><div class='lbl'>Variation {label}</div>", unsafe_allow_html=True)
                    st.text_area(f"Copy {label} — {sugg['section_name']}", value=var.get("copy",""),
                                 height=160, key=f"lp_copy_{sugg['section_name']}_{label}", label_visibility="collapsed")
                    st.markdown(f"<div style='font-size:0.78rem;color:#555;margin-top:0.4rem;'>{var.get('rationale','')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:0.3rem;'>{sop_tags}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:0.75rem;color:#5a7a00;margin-top:0.3rem;'>✦ {var.get('ai_visibility_tactic','')}</div></div>", unsafe_allow_html=True)