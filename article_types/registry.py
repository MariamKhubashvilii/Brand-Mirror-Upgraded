from .base import ArticleTypeHandler
from .listicle import ListicleHandler


ARTICLE_TYPE_HANDLERS = {"listicle": ListicleHandler()}


def get_handler(article_type: str) -> ArticleTypeHandler:
    try:
        return ARTICLE_TYPE_HANDLERS[article_type]
    except KeyError as error:
        raise NotImplementedError(f"No handler yet for article type: {article_type}") from error
