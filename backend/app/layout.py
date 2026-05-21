import math

import numpy as np


def compute_layout(vectors_by_id: dict[str, list[float]]) -> dict[str, tuple[float, float, float]]:
    ids = list(vectors_by_id.keys())
    count = len(ids)
    if count == 0:
        return {}
    if count <= 5:
        return _small_layout(ids)

    import umap

    vectors = np.asarray([vectors_by_id[track_id] for track_id in ids], dtype=np.float32)
    try:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=max(2, min(15, count - 1)),
            metric="cosine",
            init="random",
            random_state=42,
        )
        coords = reducer.fit_transform(vectors)
    except Exception:
        return _small_layout(ids)
    coords = _scale(coords)
    return {track_id: tuple(float(value) for value in coords[idx]) for idx, track_id in enumerate(ids)}


def _small_layout(ids: list[str]) -> dict[str, tuple[float, float, float]]:
    if len(ids) == 1:
        return {ids[0]: (0.0, 0.0, 0.0)}
    count = len(ids)
    if count == 2:
        return {ids[0]: (-1.8, 0.0, 0.0), ids[1]: (1.8, 0.0, 0.0)}

    radius = 2.2
    return {
        track_id: (
            math.cos((idx / count) * math.tau) * radius,
            math.sin((idx / count) * math.tau) * radius,
            ((idx % 2) - 0.5) * 0.8,
        )
        for idx, track_id in enumerate(ids)
    }


def _scale(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    max_abs = float(np.max(np.abs(centered)))
    if max_abs == 0:
        return centered
    return centered / max_abs * 5.0
