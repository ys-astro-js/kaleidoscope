from pathlib import Path


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._table = None

    def upsert(self, track_id: str, vector: list[float]) -> None:
        table = self._open_table()
        if table is None:
            self._table = self._connect().create_table(
                "tracks",
                data=[{"id": track_id, "vector": vector}],
            )
            return

        self.delete(track_id)
        self._open_table().add([{"id": track_id, "vector": vector}])

    def delete(self, track_id: str) -> None:
        table = self._open_table()
        if table is None:
            return
        try:
            table.delete(f"id = '{track_id}'")
        except Exception:
            return

    def all_vectors(self) -> dict[str, list[float]]:
        table = self._open_table()
        if table is None:
            return {}
        rows = table.to_arrow().to_pylist()
        return {row["id"]: row["vector"] for row in rows}

    def similar(self, vector: list[float], *, exclude_id: str, limit: int = 3) -> list[str]:
        table = self._open_table()
        if table is None:
            return []
        rows = table.search(vector).metric("cosine").limit(limit + 1).to_list()
        return [row["id"] for row in rows if row["id"] != exclude_id][:limit]

    def _connect(self):
        import lancedb

        self.path.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self.path))

    def _open_table(self):
        if self._table is not None:
            return self._table

        db = self._connect()
        if "tracks" in db.table_names():
            self._table = db.open_table("tracks")
            return self._table

        return None
