"""Pluggable article-type generation handlers."""

from .registry import get_handler, has_handler, ARTICLE_TYPE_HANDLERS

__all__ = ["get_handler", "has_handler", "ARTICLE_TYPE_HANDLERS"]
