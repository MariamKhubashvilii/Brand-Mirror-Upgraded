from typing import Callable, List, Optional, Tuple


def build_competitor_results(
    comp_urls: List[str],
    comp_html_pastes: List[str],
    scrape_fn: Callable[[str], dict],
    parse_html_fn: Callable[[str, str], dict],
    comp_text_pastes: Optional[List[str]] = None,
    is_usable_paste_fn: Optional[Callable[[str], bool]] = None,
    paste_shortfall_fn: Optional[Callable[[str], str]] = None,
):
    comp_text_pastes = comp_text_pastes or []
    is_usable = is_usable_paste_fn or (lambda text: bool(text.strip()))
    results = []
    for slot_index, url in enumerate(comp_urls):
        if not url.strip():
            continue
        text_paste = comp_text_pastes[slot_index].strip() if slot_index < len(comp_text_pastes) else ""
        html_paste = comp_html_pastes[slot_index].strip() if slot_index < len(comp_html_pastes) else ""
        if text_paste:
            # A plain-text paste is a deliberate opt-out of scraping/HTML parsing for this
            # slot — if it's too short, that's a rejection to fix, not a cue to fall back
            # to a scrape the user didn't ask for.
            if is_usable(text_paste):
                r = {"success": True, "text": text_paste, "title": url, "sections": [], "url": url}
            else:
                message = paste_shortfall_fn(text_paste) if paste_shortfall_fn else "Pasted text is too short."
                r = {"success": False, "error": message, "url": url}
        elif html_paste:
            r = parse_html_fn(html_paste, url)
            r["url"] = url
        else:
            r = scrape_fn(url)
        r["slot_index"] = slot_index
        if r.get("url") is None:
            r["url"] = url
        results.append(r)
    return results


def finalize_competitor_results(results: List[dict], paste_values: List[str], is_usable_paste_fn: Callable[[str], bool]) -> Tuple[List[dict], List[dict]]:
    finalized = []
    excluded = []
    for idx, r in enumerate(results):
        if r.get("success"):
            finalized.append(r)
            continue
        paste_text = paste_values[idx].strip() if idx < len(paste_values) else ""
        if paste_text and is_usable_paste_fn(paste_text):
            updated = dict(r)
            updated["text"] = paste_text
            updated["success"] = True
            finalized.append(updated)
        else:
            excluded.append({
                "url": r.get("url") or f"competitor {idx + 1}",
                "error": r.get("error", "No usable content"),
                "slot_index": r.get("slot_index", idx),
            })
    return finalized, excluded
