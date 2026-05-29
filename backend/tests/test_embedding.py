import numpy as np

from app.embedding import (
    MIN_SEGMENT_ERROR,
    MuQEmbedder,
    aggregate_segment_vectors,
    normalize_vector,
    select_audio_segments,
)


def test_normalize_vector() -> None:
    vector = normalize_vector(np.asarray([3.0, 4.0], dtype=np.float32))
    assert np.allclose(vector, np.asarray([0.6, 0.8], dtype=np.float32))


def test_normalize_zero_vector() -> None:
    vector = normalize_vector(np.asarray([0.0, 0.0], dtype=np.float32))
    assert np.allclose(vector, np.asarray([0.0, 0.0], dtype=np.float32))


def test_select_audio_segments_uses_overlap_and_discards_short_tail() -> None:
    wav = np.arange(95, dtype=np.float32)

    segments = select_audio_segments(
        wav,
        sample_rate=1,
        segment_seconds=30,
        hop_seconds=15,
    )

    assert [segment[0] for segment in segments] == [0, 15, 30, 45, 60]
    assert [len(segment) for segment in segments] == [30, 30, 30, 30, 30]


def test_select_audio_segments_keeps_only_full_segments() -> None:
    wav = np.arange(60, dtype=np.float32)

    segments = select_audio_segments(wav, sample_rate=1, segment_seconds=30, hop_seconds=15)

    assert [segment[0] for segment in segments] == [0, 15, 30]
    assert [len(segment) for segment in segments] == [30, 30, 30]


def test_select_audio_segments_filters_low_energy_windows() -> None:
    wav = np.concatenate(
        [
            np.zeros(30, dtype=np.float32),
            np.full(30, 0.01, dtype=np.float32),
            np.ones(30, dtype=np.float32),
        ]
    )

    segments = select_audio_segments(
        wav,
        sample_rate=1,
        segment_seconds=30,
        hop_seconds=30,
        min_rms_ratio=0.5,
    )

    assert len(segments) == 1
    assert np.allclose(segments[0], np.ones(30, dtype=np.float32))


def test_select_audio_segments_returns_no_segments_for_short_audio() -> None:
    wav = np.arange(29, dtype=np.float32)

    segments = select_audio_segments(wav, sample_rate=1, segment_seconds=30)

    assert segments == []


def test_aggregate_segment_vectors_normalizes_each_segment_before_averaging() -> None:
    vectors = [
        np.asarray([10.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ]

    aggregate = aggregate_segment_vectors(vectors)

    expected = normalize_vector(np.asarray([0.5, 0.5], dtype=np.float32))
    assert np.allclose(aggregate, expected)


def test_embed_file_rejects_audio_without_full_segment(monkeypatch) -> None:
    import librosa

    embedder = MuQEmbedder("unused", sample_rate=1)
    monkeypatch.setattr(librosa, "load", lambda path, sr, mono: (np.arange(29, dtype=np.float32), sr))

    def fail_load():
        raise AssertionError("model should not load for short audio")

    monkeypatch.setattr(embedder, "_load", fail_load)

    try:
        embedder.embed_file("short.wav")
    except ValueError as exc:
        assert str(exc) == MIN_SEGMENT_ERROR
    else:
        raise AssertionError("short audio should fail")


def test_embed_file_returns_segment_and_average_embeddings(monkeypatch) -> None:
    import librosa
    import torch

    calls = []

    class FakeModel:
        def __call__(self, wavs, output_hidden_states):
            calls.append(wavs.shape)
            means = wavs.mean(dim=1)
            hidden_state = torch.zeros((wavs.shape[0], 2, 2), dtype=torch.float32)
            hidden_state[:, 0, 0] = means
            hidden_state[:, 1, 1] = means
            return type(
                "Output",
                (),
                {"last_hidden_state": hidden_state},
            )()

    embedder = MuQEmbedder("unused", sample_rate=1)
    monkeypatch.setattr(librosa, "load", lambda path, sr, mono: (np.arange(60, dtype=np.float32), sr))
    monkeypatch.setattr(embedder, "_load", lambda: (FakeModel(), "cpu"))

    embeddings = embedder.embed_file("long.wav")

    assert calls == [torch.Size([3, 30])]
    assert len(embeddings.segment_semantic) == 3
    assert len(embeddings.global_semantic) == 2
    assert embeddings.segments == embeddings.segment_semantic
    assert embeddings.average == embeddings.global_semantic
    assert np.allclose(embeddings.global_semantic, normalize_vector(np.asarray([0.5, 0.5])))


def test_embed_file_uses_mulan_audio_path_for_mulan_model(monkeypatch) -> None:
    import librosa
    import torch

    calls = []

    class FakeMulan:
        def __call__(self, *, wavs):
            calls.append(wavs.shape)
            return torch.tensor([[2.0, 0.0]])

    embedder = MuQEmbedder("OpenMuQ/MuQ-MuLan-large", sample_rate=1)
    monkeypatch.setattr(librosa, "load", lambda path, sr, mono: (np.arange(30, dtype=np.float32), sr))
    monkeypatch.setattr(embedder, "_load", lambda: (FakeMulan(), "cpu"))

    embeddings = embedder.embed_file("long.wav")

    assert calls == [torch.Size([1, 30])]
    assert embeddings.segment_semantic == [[1.0, 0.0]]
    assert embeddings.global_semantic == [1.0, 0.0]
