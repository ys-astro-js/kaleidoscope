import logging
from pathlib import Path

import pytest

from app import cover_alignment
from app.cover_alignment import (
    CoverAlignmentFeature,
    cover_alignment_scores,
    extract_cover_alignment_feature,
)


def test_cover_alignment_missing_assets_returns_none_and_warns_once(
    caplog,
    tmp_path: Path,
) -> None:
    cover_alignment._MISSING_ASSETS_WARNING_EMITTED = False
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"audio")

    with caplog.at_level(logging.WARNING):
        assert extract_cover_alignment_feature(audio_path, tmp_path / "clews") is None
        assert extract_cover_alignment_feature(audio_path, tmp_path / "clews") is None

    messages = [
        record.message
        for record in caplog.records
        if "CLEWS assets are not ready" in record.message
    ]
    assert len(messages) == 1


def test_cover_alignment_embedder_failure_returns_none_and_warns_once(
    caplog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cover_alignment._LOAD_WARNING_EMITTED = False
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"audio")
    model_dir = tmp_path / "clews"
    source_dir = model_dir / "source" / "clews-main"
    source_dir.mkdir(parents=True)
    (source_dir / "models").mkdir()
    (source_dir / "inference.py").write_text("# official CLEWS source marker\n", encoding="utf-8")
    (model_dir / "checkpoint_best.ckpt").write_bytes(b"checkpoint")
    (model_dir / "configuration.yaml").write_text("model:\n  name: fake\n", encoding="utf-8")

    class BrokenEmbedder:
        def embed_file(self, audio_path: Path):
            raise RuntimeError("missing dependency")

    cover_alignment._optional_clews_embedder.cache_clear()
    monkeypatch.setattr(
        cover_alignment,
        "_optional_clews_embedder",
        lambda checkpoint_path: BrokenEmbedder(),
    )

    with caplog.at_level(logging.WARNING):
        assert extract_cover_alignment_feature(audio_path, model_dir) is None
        assert extract_cover_alignment_feature(audio_path, model_dir) is None

    messages = [
        record.message
        for record in caplog.records
        if "CLEWS inference is unavailable" in record.message
    ]
    assert len(messages) == 1


def test_cover_alignment_scores_match_loop_semantics() -> None:
    query = CoverAlignmentFeature(
        model_key="clews",
        global_embedding=[1.0, 0.0, 0.0],
        segment_embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        segment_start_seconds=[0.0, 5.0, 10.0],
    )
    candidate = CoverAlignmentFeature(
        model_key="clews",
        global_embedding=[0.0, 1.0, 0.0],
        segment_embeddings=[
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        segment_start_seconds=[0.0, 5.0, 10.0],
    )

    scores = cover_alignment_scores(query, candidate)

    assert scores.global_score == 0.0
    assert scores.best_segment_score == pytest.approx(1.0)
    assert scores.alignment_consistency == pytest.approx((1.0 + 1.0) / 3.0)
