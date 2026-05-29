from pathlib import Path
import importlib
import sys
import types

import numpy as np
import torch

from app.clews_inference import (
    CLEWS_CHECKPOINT_NAME,
    CLEWS_CONFIG_NAME,
    ClewsEmbedder,
    _clear_shadowed_clews_modules,
    _prepend_sys_path,
    find_clews_assets,
)
from app import clews_inference


def test_find_clews_assets_requires_checkpoint_config_and_source(tmp_path: Path) -> None:
    assert find_clews_assets(tmp_path) is None

    (tmp_path / CLEWS_CHECKPOINT_NAME).write_bytes(b"checkpoint")
    (tmp_path / CLEWS_CONFIG_NAME).write_text("model:\n  name: fake\n", encoding="utf-8")
    source_dir = tmp_path / "source" / "clews-main"
    (source_dir / "models").mkdir(parents=True)
    (source_dir / "inference.py").write_text("# official CLEWS source marker\n", encoding="utf-8")

    assets = find_clews_assets(tmp_path)

    assert assets is not None
    assert assets.checkpoint_path == tmp_path / CLEWS_CHECKPOINT_NAME
    assert assets.config_path == tmp_path / CLEWS_CONFIG_NAME
    assert assets.source_dir == source_dir


def test_clews_embedder_returns_normalized_features_with_five_second_starts(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"audio")
    embedder = ClewsEmbedder(tmp_path / CLEWS_CHECKPOINT_NAME)

    class FakeAudioUtils:
        def load_audio(self, path: str, *, sample_rate: int, n_channels: int):
            assert path == str(audio_path)
            assert sample_rate == 24_000
            assert n_channels == 1
            return torch.zeros(1, 24_000)

    class FakeModel:
        def __call__(self, audio, *, shingle_hop: float, shingle_len):
            assert shingle_hop == 5.0
            assert shingle_len is None
            return torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])

    embedder._model = FakeModel()
    embedder._audio_utils = FakeAudioUtils()
    embedder._sample_rate = 24_000
    embedder._device = "cpu"

    feature = embedder.embed_file(audio_path)

    assert feature is not None
    assert feature.model_key == "clews"
    assert feature.segment_start_seconds == [0.0, 5.0]
    assert np.allclose(feature.segment_embeddings[0], [0.6, 0.8])
    assert np.allclose(feature.segment_embeddings[1], [0.0, 1.0])
    assert np.isclose(np.linalg.norm(feature.global_embedding), 1.0)


def test_clews_embedder_falls_back_to_librosa_audio_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "track.m4a"
    audio_path.write_bytes(b"audio")
    embedder = ClewsEmbedder(tmp_path / CLEWS_CHECKPOINT_NAME)

    class BrokenAudioUtils:
        def load_audio(self, path: str, *, sample_rate: int, n_channels: int):
            return None

    class FakeModel:
        def __call__(self, audio, *, shingle_hop: float, shingle_len):
            assert audio.shape == (1, 24_000)
            return torch.tensor([[[1.0, 0.0]]])

    embedder._model = FakeModel()
    embedder._audio_utils = BrokenAudioUtils()
    embedder._sample_rate = 24_000
    embedder._device = "cpu"
    monkeypatch.setattr(
        clews_inference,
        "_load_audio_with_librosa",
        lambda path, sample_rate: torch.zeros(1, sample_rate),
    )

    feature = embedder.embed_file(audio_path)

    assert feature is not None
    assert feature.segment_embeddings == [[1.0, 0.0]]


def test_clews_source_import_clears_shadowed_top_level_models_module(tmp_path: Path) -> None:
    source_dir = tmp_path / "clews-source"
    models_dir = source_dir / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "__init__.py").write_text("", encoding="utf-8")
    (models_dir / "clews.py").write_text("loaded_from = 'clews-source'\n", encoding="utf-8")
    shadow_module = types.ModuleType("models")
    sys.modules["models"] = shadow_module

    try:
        _prepend_sys_path(source_dir)
        _clear_shadowed_clews_modules(source_dir)
        imported = importlib.import_module("models.clews")

        assert imported.loaded_from == "clews-source"
    finally:
        for module_name in ("models", "models.clews"):
            sys.modules.pop(module_name, None)
        source_dir_text = str(source_dir.resolve())
        if source_dir_text in sys.path:
            sys.path.remove(source_dir_text)
