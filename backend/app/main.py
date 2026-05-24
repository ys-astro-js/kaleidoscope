import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import database
from app.context import AppContext, create_app_context
from app.dependencies import set_app_context
from app.routers import feedback, media, tracks

context = create_app_context()
settings = context.settings
conn = context.conn
vectors = context.vectors
service = context.service


def create_app(app_context: AppContext = context) -> FastAPI:
    set_app_context(app_context)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        resume_pending_tracks(app_context)
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(tracks.router)
    app.include_router(feedback.router)
    app.include_router(media.router)
    return app


def resume_pending_tracks(app_context: AppContext) -> None:
    asyncio.create_task(asyncio.to_thread(app_context.service.recompute_layout))
    for row in database.list_tracks(app_context.conn):
        if row["status"] in {"queued", "processing"}:
            asyncio.create_task(asyncio.to_thread(app_context.service.process_track, row["id"]))


app = create_app(context)
