# Brand Mirror

An AI-powered content tool that researches competitors, builds structured outlines, and drafts full articles and landing page copy — tuned to your brand voice.

Built with Streamlit, GPT-4o, and Claude Sonnet.

---

## What it does

### Article Writer
A four-step pipeline for producing SEO-ready articles from scratch or improving existing ones.

1. **Research** — Scrapes up to 3 competitor URLs, extracts structured content, and runs a competitor analysis using GPT-4o. Surfaces content gaps, LSI keywords, search intent, entity recommendations, and AI visibility tactics.
2. **Score existing article** — If you have a current page, paste it in and get a detailed SOP audit with scored dimensions (conciseness, active voice, structure, skimmability, CTA presence) and prioritized fix suggestions.
3. **Outline** — Generates two distinct outlines with different tones, each with H2 and H3 sections, keyword placement rationale, content briefs, and example sentences per section. You pick one, give feedback, and confirm.
4. **Draft and final** — Claude Sonnet drafts selected sections in your brand voice. You review and edit inline. Claude then assembles the full final article in markdown, respecting your edits. Download as `.md`.

### Landing Page Optimizer
Scrapes your landing page, detects logical sections (hero, features, FAQ, CTA, testimonials), and generates two copy variations per selected section — with rationale, applied SOPs, and AI visibility tactics.

---

## Architecture

| Task | Model | Why |
|---|---|---|
| Competitor research | GPT-4o | Fast structured JSON, reliable at analysis |
| Outline generation | GPT-4o | Strong at planning and taxonomy |
| Section drafting | Claude Sonnet | Better prose, voice consistency, longer outputs |
| Final article | Claude Sonnet | Best at sustained tone across 1000+ words |
| LP section analysis | GPT-4o | Structured detection task |
| LP copy suggestions | GPT-4o | Variation generation with JSON schema |

---

## SOPs and voice

Content is generated against a 34-rule editorial SOP covering:
- Active voice, one idea per sentence, direct answers first
- No em dashes, no banned words (leverage, seamless, robust, etc.)
- Bucket brigades, expert quotes, internal linking recommendations
- PAS formula for intros, featured snippet optimization

Also incorporates an AI Visibility Guide covering entity association, semantic HTML, schema markup types, factual consistency, and engagement signals.

Brand voice is configurable per session in the sidebar. Pass in tone, audience, USPs, banned words, and CTA style.

---

## Setup

### Requirements

```
streamlit>=1.35.0
openai>=1.30.0
anthropic>=0.40.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### API keys

Enter your OpenAI and Anthropic API keys in the sidebar at runtime. For Streamlit Cloud deployment, add them to your app secrets:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

Then update `get_client()` and `get_claude_client()` in `app.py` to pull from `st.secrets` instead of session state.

---

## File structure

```
├── app.py          # Streamlit UI and session state
├── llm.py          # All LLM calls and prompt logic
├── scraper.py      # URL scraping and content extraction
├── sops.py         # Editorial SOPs and AI visibility guide
└── requirements.txt
```

---

## Deployment

Deploy to [Streamlit Cloud](https://streamlit.io/cloud) by pushing to a GitHub repo and connecting it. Free tier is sufficient. Add API keys via the Streamlit secrets manager in the dashboard — never hardcode them.

---

## Notes

- Scraper truncates pages to ~6000 words to stay within model context limits
- All LLM outputs are JSON-validated before rendering; malformed responses are caught and surfaced as errors
- Session state persists within a single browser session; refreshing clears all data
- The tool does not store any content, keys, or brand data between sessions
