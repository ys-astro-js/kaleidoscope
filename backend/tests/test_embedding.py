import numpy as np

from app.embedding import normalize_vector


def test_normalize_vector() -> None:
    vector = normalize_vector(np.asarray([3.0, 4.0], dtype=np.float32))
    assert np.allclose(vector, np.asarray([0.6, 0.8], dtype=np.float32))


def test_normalize_zero_vector() -> None:
    vector = normalize_vector(np.asarray([0.0, 0.0], dtype=np.float32))
    assert np.allclose(vector, np.asarray([0.0, 0.0], dtype=np.float32))

