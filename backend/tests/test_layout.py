from app.layout import compute_layout


def test_empty_layout() -> None:
    assert compute_layout({}) == {}


def test_small_layouts() -> None:
    one = compute_layout({"a": [1.0, 0.0]})
    two = compute_layout({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    three = compute_layout({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})

    assert one["a"] == (0.0, 0.0, 0.0)
    assert len(two) == 2
    assert len(three) == 3
    assert all(len(coords) == 3 for coords in three.values())


def test_four_track_layout_does_not_require_umap() -> None:
    layout = compute_layout(
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [1.0, 1.0],
            "d": [0.5, 0.5],
        }
    )

    assert set(layout) == {"a", "b", "c", "d"}
    assert all(len(coords) == 3 for coords in layout.values())
