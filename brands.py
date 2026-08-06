"""Small library of saved brand profiles.

Each profile is a plain .txt file in brands/, written in the same freeform
conventions as the sidebar's Brand Knowledge textarea (Brand voice, Audience,
USPs, Banned words, CTA style, ...). Add a new brand by dropping another
.txt file in brands/ — no code changes needed.
"""
import os
from typing import Dict

BRANDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brands")


def _display_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return stem.replace("_", " ").replace("-", " ").title()


def list_brand_profiles(brands_dir: str = BRANDS_DIR) -> Dict[str, str]:
    """Return {display_name: file_path} for every brand profile, sorted by name."""
    if not os.path.isdir(brands_dir):
        return {}
    profiles = {
        _display_name(filename): os.path.join(brands_dir, filename)
        for filename in os.listdir(brands_dir)
        if filename.endswith(".txt")
    }
    return dict(sorted(profiles.items()))


def load_brand_profile(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def merge_brand_knowledge(profile_text: str, typed_text: str) -> str:
    """Combine a saved brand profile with the user's typed Brand Knowledge text.

    Profile first, then the typed text appended if non-empty — so a saved brand
    profile always merges cleanly with whatever's in the textarea rather than
    overwriting it.
    """
    return "\n\n".join(part.strip() for part in (profile_text, typed_text) if part and part.strip())
