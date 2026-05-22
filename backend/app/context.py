import sqlite3
from dataclasses import dataclass

from app import database
from app.config import Settings, ensure_data_dirs, get_settings
from app.service import TrackService
from app.vector_store import VectorStore


@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    vectors: VectorStore
    service: TrackService


def create_app_context(settings: Settings | None = None) -> AppContext:
    app_settings = settings or get_settings()
    ensure_data_dirs(app_settings)
    conn = database.connect(app_settings.sqlite_path)
    database.init_db(conn)
    vectors = VectorStore(app_settings.lancedb_dir)
    service = TrackService(app_settings, conn, vectors)
    return AppContext(
        settings=app_settings,
        conn=conn,
        vectors=vectors,
        service=service,
    )
