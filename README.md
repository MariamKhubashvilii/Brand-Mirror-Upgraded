# Brand Mirror

An AI-powered content tool that turns competitor research into evidence-grounded article plans, sample drafts, and full articles—tuned to your brand voice.

Built with Streamlit, GPT-4o, and Claude Sonnet.

---

## What it does

### Article Writer
A research-first workflow for producing SEO-ready articles from scratch or improving existing ones.

1. **Research** — Scrapes up to 3 competitor URLs. If any page cannot be scraped, research pauses and asks you to paste usable page text; it never silently researches only a partial source set. Each source is converted into a source-linked evidence pack (claims, features, list items, limitations, statistics, quotes, and snippets), then GPT performs a cross-source synthesis before strategic research begins.
2. **Score existing article** — If you have a current page, paste it in and get a detailed SOP audit with scored dimensions (conciseness, active voice, structure, skimmability, CTA presence) and prioritized fix suggestions.
3. **Plan** — For listicles, generate two editable skeletons. Review the complete H2 sequence (pre-list, every list item, and post-list), add/remove/rename sections, select whole H2s to use as sample drafts, and accept or override a research-based component suggestion. Other article types use two full outline variants and feedback-driven regeneration.
4. **Sample draft and revise** — Draft selected complete H2 sections—such as the intro, a guidance section, or a specific list item—with an optional direction. Edit inline and run one or more revision prompts until the writing style is right.
5. **Final article** — Generate the complete markdown article from the approved plan, source-backed evidence, sample drafts, and your edits. Download it as `.md`.

### Landing Page Optimizer
Scrapes your landing page, detects logical sections (hero, features, FAQ, CTA, testimonials), and generates two copy variations per selected section — with rationale, applied SOPs, and AI visibility tactics.

---

## Architecture

| Task | Model | Why |
|---|---|---|
| Source evidence extraction | GPT-4o mini | Structured, source-linked evidence capture |
| Cross-source research synthesis | GPT-4o mini | Compares all extracted evidence before planning |
| Competitor research | GPT-4o mini | Fast structured JSON, reliable at analysis |
| Outline/skeleton generation | GPT-4o mini | Strong at planning and taxonomy |
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
├── llm.py          # LLM orchestration and workflow logic
├── prompts.py      # Shared prompt builders
├── article_types/  # Pluggable article-type handlers (listicle today)
├── extraction.py   # Per-source outline and evidence extraction
├── scraper.py      # URL scraping and page parsing
├── source_flow.py  # Scrape/paste fallback handling
├── sops.py         # Editorial SOPs and AI visibility guide
└── requirements.txt
```

---

## Deployment

Deploy to [Streamlit Cloud](https://streamlit.io/cloud) by pushing to a GitHub repo and connecting it. Free tier is sufficient. Add API keys via the Streamlit secrets manager in the dashboard — never hardcode them.

---

## Notes

- Source evidence is attached to expanded listicle sections by ID and supplied to the relevant sample-drafting and revision calls.
- Raw pasted content is capped during extraction to stay within model context limits.
- All LLM outputs are JSON-validated before rendering; malformed responses are caught and surfaced as errors
- Session state persists within a single browser session; refreshing clears all data
- The tool does not store any content, keys, or brand data between sessions
