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
    coords = _scale(coords, count)
    coords = _spread_apart(coords, target_distance=_target_min_distance(count))
    return {
        track_id: (float(coords[idx][0]), float(coords[idx][1]), float(coords[idx][2]))
        for idx, track_id in enumerate(ids)
    }


def compute_similarity_layout(
    similarity_by_id: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float, float]]:
    ids = list(similarity_by_id.keys())
    count = len(ids)
    if count == 0:
        return {}
    if count == 1:
        return {ids[0]: (0.0, 0.0, 0.0)}

    distances = _distance_matrix(similarity_by_id, ids)
    coords: np.ndarray | None = None

    if count > 5:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=max(2, min(15, count - 1)),
                metric="precomputed",
                init="random",
                random_state=42,
            )
            coords = reducer.fit_transform(distances)
        except Exception:
            coords = None

    if coords is None:
        coords = _classical_mds(distances)
    if coords is None:
        return _small_layout(ids)

    coords = _scale(coords, count)
    coords = _spread_apart(coords, target_distance=_target_min_distance(count))
    return {
        track_id: (float(coords[idx][0]), float(coords[idx][1]), float(coords[idx][2]))
        for idx, track_id in enumerate(ids)
    }


def _small_layout(ids: list[str]) -> dict[str, tuple[float, float, float]]:
    if len(ids) == 1:
        return {ids[0]: (0.0, 0.0, 0.0)}
    count = len(ids)
    if count == 2:
        return {ids[0]: (-1.8, 0.0, 0.0), ids[1]: (1.8, 0.0, 0.0)}

    radius = 3.0
    return {
        track_id: (
            math.cos((idx / count) * math.tau) * radius,
            math.sin((idx / count) * math.tau) * radius,
            ((idx % 2) - 0.5) * 1.1,
        )
        for idx, track_id in enumerate(ids)
    }


def _scale(coords: np.ndarray, count: int) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    max_abs = float(np.max(np.abs(centered)))
    if max_abs == 0:
        return centered
    radius = min(14.0, max(8.0, 4.5 + math.sqrt(count)))
    return centered / max_abs * radius


def _target_min_distance(count: int) -> float:
    return min(1.65, max(1.2, 7.5 / math.sqrt(count)))


def _distance_matrix(
    similarity_by_id: dict[str, dict[str, float]],
    ids: list[str],
) -> np.ndarray:
    distances = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for first, first_id in enumerate(ids):
        for second in range(first + 1, len(ids)):
            second_id = ids[second]
            similarity = _pair_similarity(similarity_by_id, first_id, second_id)
            distance = 1.0 - similarity
            distances[first][second] = distance
            distances[second][first] = distance
    return distances


def _pair_similarity(
    similarity_by_id: dict[str, dict[str, float]],
    first_id: str,
    second_id: str,
) -> float:
    scores = [
        score
        for score in (
            similarity_by_id.get(first_id, {}).get(second_id),
            similarity_by_id.get(second_id, {}).get(first_id),
        )
        if score is not None
    ]
    if not scores:
        return 0.0
    return max(0.0, min(1.0, float(np.mean(scores))))


def _classical_mds(distances: np.ndarray) -> np.ndarray | None:
    count = len(distances)
    if count == 0:
        return None

    squared = distances.astype(np.float64) ** 2
    centering = np.eye(count) - np.full((count, count), 1.0 / count)
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]

    coords = np.zeros((count, 3), dtype=np.float32)
    used = 0
    for eigen_index in order:
        value = float(values[eigen_index])
        if value <= 1e-9:
            continue
        coords[:, used] = vectors[:, eigen_index] * math.sqrt(value)
        used += 1
        if used == 3:
            break

    if used == 0:
        return None
    return coords


def _spread_apart(
    coords: np.ndarray,
    *,
    target_distance: float,
    iterations: int = 80,
) -> np.ndarray:
    if len(coords) < 2:
        return coords

    spread = coords.astype(np.float32, copy=True)
    for _ in range(iterations):
        deltas = np.zeros_like(spread)
        for first in range(len(spread)):
            for second in range(first + 1, len(spread)):
                diff = spread[first] - spread[second]
                distance = float(np.linalg.norm(diff))
                if distance == 0.0:
                    diff = _deterministic_direction(first, second)
                    distance = float(np.linalg.norm(diff))
                if distance >= target_distance:
                    continue

                push = (target_distance - distance) / target_distance * 0.035
                direction = diff / distance
                deltas[first] += direction * push
                deltas[second] -= direction * push
        spread += deltas
        spread -= spread.mean(axis=0)
    return spread


def _deterministic_direction(first: int, second: int) -> np.ndarray:
    angle = (first * 97 + second * 57) % 360
    radians = math.radians(angle)
    return np.asarray(
        [
            math.cos(radians),
            math.sin(radians),
            0.5 if (first + second) % 2 == 0 else -0.5,
        ],
        dtype=np.float32,
    )
