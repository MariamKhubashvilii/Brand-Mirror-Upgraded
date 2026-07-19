"""Pluggable article-type generation handlers."""

from .registry import get_handler

__all__ = ["get_handler"]
