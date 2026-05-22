from pathlib import Path

from app import database

JAPANESE_FILENAME = "\u6771\u4eac\u4e8b\u5909 - \u7fa4\u9752\u65e5\u548c.mp3"
JAPANESE_TITLE = "\u6771\u4eac\u4e8b\u5909 - \u7fa4\u9752\u65e5\u548c"


def test_insert_track_preserves_multibyte_filename(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)

    database.insert_track(
        conn,
        track_id="track-id",
        filename=JAPANESE_FILENAME,
        title=JAPANESE_TITLE,
        audio_path=tmp_path / "track-id.mp3",
        art_path=tmp_path / "track-id.png",
    )

    row = database.get_track(conn, "track-id")

    assert row is not None
    assert row["filename"] == JAPANESE_FILENAME
    assert row["title"] == JAPANESE_TITLE
    assert row["album"] is None


def test_init_db_adds_album_to_existing_tracks_table(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    conn.execute(
        """
        CREATE TABLE tracks (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            artist TEXT,
            audio_path TEXT NOT NULL,
            art_path TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.commit()

    database.init_db(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(tracks)")}
    assert "album" in columns


def test_feedback_events_are_removed_with_track(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    for track_id in ("query", "candidate"):
        database.insert_track(
            conn,
            track_id=track_id,
            filename=f"{track_id}.mp3",
            title=track_id,
            audio_path=tmp_path / f"{track_id}.mp3",
            art_path=tmp_path / f"{track_id}.png",
        )

    database.insert_feedback_event(
        conn,
        query_track_id="query",
        candidate_track_id="candidate",
        label="similar",
    )

    assert len(database.list_feedback_events(conn)) == 1

    database.delete_track(conn, "candidate")

    assert database.list_feedback_events(conn) == []


def test_feedback_weights_roundtrip(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)

    database.set_feedback_weights(
        conn,
        global_weight=0.2,
        segment_weight=0.3,
        chroma_weight=0.5,
        event_count=12,
    )

    row = database.get_feedback_weights(conn)

    assert row is not None
    assert row["global_weight"] == 0.2
    assert row["segment_weight"] == 0.3
    assert row["chroma_weight"] == 0.5
    assert row["event_count"] == 12


def test_track_layout_stores_cluster_label(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    database.insert_track(
        conn,
        track_id="track-id",
        filename="track-id.mp3",
        title="track-id",
        audio_path=tmp_path / "track-id.mp3",
        art_path=tmp_path / "track-id.png",
    )

    database.set_track_layout(conn, {"track-id": (1.0, 2.0, 3.0, 4)})
    row = database.get_track(conn, "track-id")

    assert row is not None
    assert row["x"] == 1.0
    assert row["y"] == 2.0
    assert row["z"] == 3.0
    assert row["cluster"] == 4
