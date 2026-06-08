import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_url(url: str) -> dict:
    """Returns dict with keys: success, url, title, text, sections, error"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url

        # Extract structured sections by headings
        sections = []
        current = {"heading": "intro", "level": 0, "text": ""}
        for el in soup.find_all(["h1","h2","h3","h4","p","li","td"]):
            tag = el.name
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if tag in ["h1","h2","h3","h4"]:
                if current["text"].strip():
                    sections.append(current)
                level = int(tag[1])
                current = {"heading": text, "level": level, "text": ""}
            else:
                current["text"] += " " + text
        if current["text"].strip():
            sections.append(current)

        full_text = " ".join(
            f"[{s['heading']}] {s['text']}" for s in sections
        )
        # Truncate to ~6000 words
        words = full_text.split()
        if len(words) > 6000:
            full_text = " ".join(words[:6000]) + "...[truncated]"

        return {"success": True, "url": url, "title": title, "text": full_text, "sections": sections, "error": None}

    except requests.exceptions.Timeout:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": "Timeout — page took too long to respond."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": "Connection error — could not reach this URL."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": f"HTTP {e.response.status_code} — page returned an error."}
    except Exception as e:
        return {"success": False, "url": url, "title": None, "text": None, "sections": [], "error": str(e)}
