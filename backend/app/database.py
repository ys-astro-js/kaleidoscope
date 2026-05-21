import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.models import TrackStatus


def connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            artist TEXT,
            audio_path TEXT NOT NULL,
            art_path TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            x REAL,
            y REAL,
            z REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def insert_track(
    conn: sqlite3.Connection,
    *,
    track_id: str,
    filename: str,
    title: str,
    audio_path: Path,
    art_path: Path,
) -> None:
    conn.execute(
        """
        INSERT INTO tracks (id, filename, title, audio_path, art_path, status)
        VALUES (?, ?, ?, ?, ?, 'queued')
        """,
        (track_id, filename, title, str(audio_path), str(art_path)),
    )
    conn.commit()


def update_track(
    conn: sqlite3.Connection,
    track_id: str,
    *,
    status: TrackStatus | None = None,
    artist: str | None = None,
    art_path: Path | None = None,
    error: str | None = None,
    coords: tuple[float, float, float] | None = None,
) -> None:
    fields: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
    values: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if artist is not None:
        fields.append("artist = ?")
        values.append(artist)
    if art_path is not None:
        fields.append("art_path = ?")
        values.append(str(art_path))
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if coords is not None:
        fields.extend(["x = ?", "y = ?", "z = ?"])
        values.extend(coords)

    values.append(track_id)
    conn.execute(f"UPDATE tracks SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def list_tracks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM tracks ORDER BY created_at ASC, id ASC"))


def get_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()


def ready_tracks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM tracks WHERE status = 'ready' ORDER BY created_at ASC"))


def set_track_coords(conn: sqlite3.Connection, coords_by_id: dict[str, tuple[float, float, float]]) -> None:
    conn.executemany(
        "UPDATE tracks SET x = ?, y = ?, z = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        [(x, y, z, track_id) for track_id, (x, y, z) in coords_by_id.items()],
    )
    conn.commit()


def delete_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    row = get_track(conn, track_id)
    if row is None:
        return None
    conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
    conn.commit()
    return row


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]

