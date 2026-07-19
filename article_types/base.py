from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class ArticleTypeHandler(ABC):
    """Contract for article-type-specific outline generation."""

    @abstractmethod
    def build_skeleton_prompt(
        self, keyword: str, research: Dict, brand_context: str,
        directive: str, variant_label: str,
    ) -> Tuple[str, str]:
        """Return system and user prompts for the skeleton stage."""

    @abstractmethod
    def structure_options(self) -> List[Dict]:
        """Return fixed, user-selectable item structure templates."""

    @abstractmethod
    def build_expansion_prompt(
        self, keyword: str, skeleton: Dict, structure_template: List[str],
        research: Dict, brand_context: str, directive: str,
    ) -> Tuple[str, str]:
        """Return system and user prompts for the expansion stage."""
