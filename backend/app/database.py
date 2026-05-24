import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.models import FeedbackLabel, TrackStatus

FEEDBACK_WEIGHTS_ID = "global"
DEFAULT_FEEDBACK_WEIGHT_VALUES = {
    "global_weight": 0.34375,
    "segment_weight": 0.15625,
    "vocals_global_weight": 0.171875,
    "vocals_segment_weight": 0.078125,
    "instrumental_global_weight": 0.171875,
    "instrumental_segment_weight": 0.078125,
}


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
            album TEXT,
            audio_path TEXT NOT NULL,
            art_path TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            x REAL,
            y REAL,
            z REAL,
            cluster INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_track_column(conn, "cluster", "INTEGER")
    _ensure_track_column(conn, "album", "TEXT")
    _ensure_track_column(conn, "vocals_path", "TEXT")
    _ensure_track_column(conn, "instrumental_path", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            candidate_track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            label TEXT NOT NULL CHECK (label IN ('similar', 'not_similar')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (query_track_id <> candidate_track_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_weights (
            id TEXT PRIMARY KEY,
            global_weight REAL NOT NULL,
            segment_weight REAL NOT NULL,
            vocals_global_weight REAL NOT NULL,
            vocals_segment_weight REAL NOT NULL,
            instrumental_global_weight REAL NOT NULL,
            instrumental_segment_weight REAL NOT NULL,
            event_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _normalize_feedback_weights_schema(conn)
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
    album: str | None = None,
    art_path: Path | None = None,
    vocals_path: Path | None = None,
    instrumental_path: Path | None = None,
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
    if album is not None:
        fields.append("album = ?")
        values.append(album)
    if art_path is not None:
        fields.append("art_path = ?")
        values.append(str(art_path))
    if vocals_path is not None:
        fields.append("vocals_path = ?")
        values.append(str(vocals_path))
    if instrumental_path is not None:
        fields.append("instrumental_path = ?")
        values.append(str(instrumental_path))
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
    set_track_layout(
        conn,
        {track_id: (x, y, z, None) for track_id, (x, y, z) in coords_by_id.items()},
    )


def set_track_layout(
    conn: sqlite3.Connection,
    layout_by_id: dict[str, tuple[float, float, float, int | None]],
) -> None:
    conn.executemany(
        """
        UPDATE tracks
        SET x = ?, y = ?, z = ?, cluster = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        [(x, y, z, cluster, track_id) for track_id, (x, y, z, cluster) in layout_by_id.items()],
    )
    conn.commit()


def delete_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    row = get_track(conn, track_id)
    if row is None:
        return None
    delete_feedback_for_track(conn, track_id, commit=False)
    conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
    conn.commit()
    return row


def insert_feedback_event(
    conn: sqlite3.Connection,
    *,
    query_track_id: str,
    candidate_track_id: str,
    label: FeedbackLabel,
) -> None:
    conn.execute(
        """
        INSERT INTO feedback_events (query_track_id, candidate_track_id, label)
        VALUES (?, ?, ?)
        """,
        (query_track_id, candidate_track_id, label),
    )
    conn.commit()


def list_feedback_events(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT query_track_id, candidate_track_id, label
            FROM feedback_events
            ORDER BY id ASC
            """
        )
    )


def delete_feedback_for_track(
    conn: sqlite3.Connection,
    track_id: str,
    *,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        DELETE FROM feedback_events
        WHERE query_track_id = ? OR candidate_track_id = ?
        """,
        (track_id, track_id),
    )
    if commit:
        conn.commit()


def get_feedback_weights(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT
            global_weight,
            segment_weight,
            vocals_global_weight,
            vocals_segment_weight,
            instrumental_global_weight,
            instrumental_segment_weight,
            event_count
        FROM feedback_weights
        WHERE id = ?
        """,
        (FEEDBACK_WEIGHTS_ID,),
    ).fetchone()


def set_feedback_weights(
    conn: sqlite3.Connection,
    *,
    global_weight: float,
    segment_weight: float,
    vocals_global_weight: float,
    vocals_segment_weight: float,
    instrumental_global_weight: float,
    instrumental_segment_weight: float,
    event_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO feedback_weights (
            id,
            global_weight,
            segment_weight,
            vocals_global_weight,
            vocals_segment_weight,
            instrumental_global_weight,
            instrumental_segment_weight,
            event_count,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            global_weight = excluded.global_weight,
            segment_weight = excluded.segment_weight,
            vocals_global_weight = excluded.vocals_global_weight,
            vocals_segment_weight = excluded.vocals_segment_weight,
            instrumental_global_weight = excluded.instrumental_global_weight,
            instrumental_segment_weight = excluded.instrumental_segment_weight,
            event_count = excluded.event_count,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            FEEDBACK_WEIGHTS_ID,
            global_weight,
            segment_weight,
            vocals_global_weight,
            vocals_segment_weight,
            instrumental_global_weight,
            instrumental_segment_weight,
            event_count,
        ),
    )
    conn.commit()


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _ensure_track_column(conn: sqlite3.Connection, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(tracks)")}
    if column not in columns:
        conn.execute(f"ALTER TABLE tracks ADD COLUMN {column} {definition}")


def _normalize_feedback_weights_schema(conn: sqlite3.Connection) -> None:
    expected_columns = {
        "id",
        "global_weight",
        "segment_weight",
        "vocals_global_weight",
        "vocals_segment_weight",
        "instrumental_global_weight",
        "instrumental_segment_weight",
        "event_count",
        "updated_at",
    }
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(feedback_weights)")}
    if columns == expected_columns:
        return

    preserved_rows = [dict(row) for row in conn.execute("SELECT * FROM feedback_weights")]
    conn.execute("ALTER TABLE feedback_weights RENAME TO feedback_weights_legacy")
    conn.execute(
        """
        CREATE TABLE feedback_weights (
            id TEXT PRIMARY KEY,
            global_weight REAL NOT NULL,
            segment_weight REAL NOT NULL,
            vocals_global_weight REAL NOT NULL,
            vocals_segment_weight REAL NOT NULL,
            instrumental_global_weight REAL NOT NULL,
            instrumental_segment_weight REAL NOT NULL,
            event_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO feedback_weights (
            id,
            global_weight,
            segment_weight,
            vocals_global_weight,
            vocals_segment_weight,
            instrumental_global_weight,
            instrumental_segment_weight,
            event_count,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["id"],
                row.get("global_weight", DEFAULT_FEEDBACK_WEIGHT_VALUES["global_weight"]),
                row.get("segment_weight", DEFAULT_FEEDBACK_WEIGHT_VALUES["segment_weight"]),
                row.get(
                    "vocals_global_weight",
                    DEFAULT_FEEDBACK_WEIGHT_VALUES["vocals_global_weight"],
                ),
                row.get(
                    "vocals_segment_weight",
                    DEFAULT_FEEDBACK_WEIGHT_VALUES["vocals_segment_weight"],
                ),
                row.get(
                    "instrumental_global_weight",
                    DEFAULT_FEEDBACK_WEIGHT_VALUES["instrumental_global_weight"],
                ),
                row.get(
                    "instrumental_segment_weight",
                    DEFAULT_FEEDBACK_WEIGHT_VALUES["instrumental_segment_weight"],
                ),
                row.get("event_count", 0),
                row.get("updated_at", "CURRENT_TIMESTAMP"),
            )
            for row in preserved_rows
        ],
    )
    conn.execute("DROP TABLE feedback_weights_legacy")
