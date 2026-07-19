import re
import requests
from bs4 import BeautifulSoup


def parse_pasted_html(html: str, label=None) -> dict:
    """Parse pasted HTML or Inspect-style snippets into the same shape as scrape_url()."""
    if not html or not html.strip():
        return {
            "success": False,
            "url": label or None,
            "title": label or None,
            "text": None,
            "sections": [],
            "error": "No HTML provided.",
        }

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else (label or "Pasted HTML")

        sections = []
        current = {"heading": "intro", "level": 0, "text": "", "format": "paragraph", "images": [], "has_code": False}
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "img", "pre"]):
            tag = el.name
            text = el.get_text(" ", strip=True)
            if not text and tag not in ["img", "pre"]:
                continue
            if tag in ["h1", "h2", "h3", "h4"]:
                if current.get("text", "").strip():
                    sections.append(current)
                level = int(tag[1])
                current = {"heading": text, "level": level, "text": "", "format": "paragraph", "images": [], "has_code": False}
            elif tag == "img":
                alt = el.get("alt") or ""
                src = el.get("src") or ""
                if alt or src:
                    current.setdefault("images", []).append(alt or src)
                current["format"] = "mixed"
            elif tag in ["ul", "ol"]:
                current["format"] = "list"
                current["text"] += " " + text
            elif tag == "table":
                current["format"] = "table"
                current["text"] += " " + text
            elif tag == "pre":
                current["has_code"] = True
                current["format"] = "mixed"
                current["text"] += " " + text
            else:
                current["text"] += " " + text
                if tag == "p":
                    current["format"] = "paragraph"
        if current.get("text", "").strip():
            sections.append(current)

        full_text = " ".join(f"[{s['heading']}] {s['text']}" for s in sections)
        words = full_text.split()
        if len(words) > 6000:
            full_text = " ".join(words[:6000]) + "...[truncated]"

        if not _looks_substantive(full_text, sections):
            word_count = len(re.findall(r"\b\w+\b", full_text))
            has_structure = bool(sections) and any((s.get("text") or "").strip() for s in sections)
            if word_count < 20 and not has_structure:
                return {
                    "success": False,
                    "url": label or None,
                    "title": title,
                    "text": None,
                    "sections": [],
                    "error": "That HTML didn't have enough readable content — try pasting the visible article text instead.",
                }

        if not full_text.strip():
            return {
                "success": False,
                "url": label or None,
                "title": title,
                "text": None,
                "sections": [],
                "error": "That HTML didn't have enough readable content — try pasting the visible article text instead.",
            }

        return {"success": True, "url": label or None, "title": title, "text": full_text, "sections": sections, "error": None}
    except Exception as e:
        return {
            "success": False,
            "url": label or None,
            "title": label or None,
            "text": None,
            "sections": [],
            "error": str(e),
        }

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MIN_WORDS = 80
MIN_TEXT_RATIO = 0.12


def _looks_substantive(text: str, sections: list[dict]) -> bool:
    words = len(re.findall(r"\b\w+\b", text))
    if words < MIN_WORDS:
        return False
    if not sections:
        return False
    text_chars = len(text)
    if text_chars and len(" ".join(s.get("text", "") for s in sections)) / max(text_chars, 1) < MIN_TEXT_RATIO:
        return False
    return True


def scrape_url(url: str) -> dict:
    """Returns dict with keys: success, url, title, text, sections, error"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract structured sections by headings
        sections = []
        current = {"heading": "intro", "level": 0, "text": "", "format": "paragraph", "images": [], "has_code": False}
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "img", "pre"]):
            tag = el.name
            text = el.get_text(" ", strip=True)
            if not text and tag not in ["img", "pre"]:
                continue
            if tag in ["h1", "h2", "h3", "h4"]:
                if current.get("text", "").strip():
                    sections.append(current)
                level = int(tag[1])
                current = {"heading": text, "level": level, "text": "", "format": "paragraph", "images": [], "has_code": False}
            elif tag == "img":
                alt = el.get("alt") or ""
                src = el.get("src") or ""
                if alt or src:
                    current.setdefault("images", []).append(alt or src)
                current["format"] = "mixed"
            elif tag in ["ul", "ol"]:
                current["format"] = "list"
                current["text"] += " " + text
            elif tag == "table":
                current["format"] = "table"
                current["text"] += " " + text
            elif tag == "pre":
                current["has_code"] = True
                current["format"] = "mixed"
                current["text"] += " " + text
            else:
                current["text"] += " " + text
                if tag == "p":
                    current["format"] = "paragraph"
        if current.get("text", "").strip():
            sections.append(current)

        full_text = " ".join(f"[{s['heading']}] {s['text']}" for s in sections)
        words = full_text.split()
        if len(words) > 6000:
            full_text = " ".join(words[:6000]) + "...[truncated]"

        if not _looks_substantive(full_text, sections):
            return {
                "success": False,
                "url": url,
                "title": title,
                "text": None,
                "sections": [],
                "error": "Content too thin or mostly navigation. Paste content manually.",
            }

        return {"success": True, "url": url, "title": title, "text": full_text, "sections": sections, "error": None}

    except requests.exceptions.Timeout:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": "Timeout — page took too long to respond."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": "Connection error — could not reach this URL."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": f"HTTP {e.response.status_code} — page returned an error."}
    except Exception as e:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": str(e)}
