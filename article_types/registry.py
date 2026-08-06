from .base import ArticleTypeHandler
from .listicle import ListicleHandler
from .comparison import ComparisonHandler
from .tutorial import TutorialHandler
from .guide import GuideHandler
from .review import ReviewHandler


ARTICLE_TYPE_HANDLERS = {
    "listicle": ListicleHandler(),
    "comparison": ComparisonHandler(),
    "tutorial/how-to": TutorialHandler(),
    "guide/informational": GuideHandler(),
    "review": ReviewHandler(),
}


def get_handler(article_type: str) -> ArticleTypeHandler:
    try:
        return ARTICLE_TYPE_HANDLERS[article_type]
    except KeyError as error:
        raise NotImplementedError(f"No handler yet for article type: {article_type}") from error


def has_handler(article_type: str) -> bool:
    return article_type in ARTICLE_TYPE_HANDLERS
