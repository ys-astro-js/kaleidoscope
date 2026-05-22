from typing import Annotated

from fastapi import Depends

from app.context import AppContext

_app_context: AppContext | None = None


def set_app_context(context: AppContext) -> None:
    global _app_context
    _app_context = context


def get_app_context() -> AppContext:
    if _app_context is None:
        raise RuntimeError("App context has not been initialized")
    return _app_context


AppContextDep = Annotated[AppContext, Depends(get_app_context)]
