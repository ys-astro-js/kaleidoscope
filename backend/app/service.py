import sqlite3
from pathlib import Path

from app import database
from app.audio import extract_metadata, normalize_audio_for_model, write_art_or_placeholder
from app.config import Settings
from app.embedding import get_embedder
from app.layout import compute_layout
from app.models import Track
from app.vector_store import VectorStore


class TrackService:
    def __init__(self, settings: Settings, conn: sqlite3.Connection, vectors: VectorStore) -> None:
        self.settings = settings
        self.conn = conn
        self.vectors = vectors

    def list_tracks(self) -> list[Track]:
        vectors = self.vectors.all_vectors()
        rows = database.rows_to_dicts(database.list_tracks(self.conn))
        tracks: list[Track] = []
        for row in rows:
            similar = []
            vector = vectors.get(row["id"])
            if row["status"] == "ready" and vector is not None:
                similar = self.vectors.similar(vector, exclude_id=row["id"], limit=3)
            tracks.append(
                Track(
                    id=row["id"],
                    filename=row["filename"],
                    title=row["title"],
                    artist=row["artist"],
                    status=row["status"],
                    error=row["error"],
                    x=row["x"],
                    y=row["y"],
                    z=row["z"],
                    similar=similar,
                )
            )
        return tracks

    def process_track(self, track_id: str) -> None:
        row = database.get_track(self.conn, track_id)
        if row is None:
            return

        try:
            database.update_track(self.conn, track_id, status="processing")
            artist, art_bytes = extract_metadata(Path(row["audio_path"]))
            art_path = write_art_or_placeholder(track_id, art_bytes, self.settings.art_dir)
            model_audio_path = normalize_audio_for_model(
                Path(row["audio_path"]),
                self.settings.audio_dir / f"{track_id}.model.wav",
                self.settings.sample_rate,
            )
            embedder = get_embedder(self.settings.model_id, self.settings.sample_rate)
            vector = embedder.embed_file(str(model_audio_path))
            self.vectors.upsert(track_id, vector)
            database.update_track(
                self.conn,
                track_id,
                status="ready",
                artist=artist,
                art_path=art_path,
            )
            self.recompute_layout()
        except Exception as exc:
            database.update_track(self.conn, track_id, status="error", error=str(exc))

    def recompute_layout(self) -> None:
        ready_ids = {row["id"] for row in database.ready_tracks(self.conn)}
        vectors = {
            track_id: vector
            for track_id, vector in self.vectors.all_vectors().items()
            if track_id in ready_ids
        }
        database.set_track_coords(self.conn, compute_layout(vectors))
