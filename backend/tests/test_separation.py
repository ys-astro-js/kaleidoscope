from pathlib import Path
import logging
from types import SimpleNamespace

import numpy as np
import torch

from app.separation import (
    INSTRUMENTAL_STEM,
    MODEL_SPECS,
    ResolvedModel,
    VOCALS_STEM,
    create_hf_separator_class,
    _coerce_roformer_tuple_fields,
    _ensure_cuda_is_preferred,
    _enable_roformer_gpu_overlap_add,
    _is_mel_band_roformer_config,
    resolve_model,
    separate_instrumental,
)


def test_resolve_model_downloads_checkpoint_and_writes_alias_config(monkeypatch, tmp_path: Path) -> None:
    def fake_list_hf_files(repo_id: str) -> list[str]:
        assert repo_id == MODEL_SPECS["hyperace_v2_inst"].repo_id
        return [
            "bs_roformer.py",
            "v2_inst/config.yaml",
            "v2_inst/model.ckpt",
            "v2_inst/bs_roformer.py",
        ]

    def fake_download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".yaml"):
            path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  instruments:",
                        "    - vocals",
                        "    - instrument",
                        "  target_instrument: instrument",
                    ]
                ),
                encoding="utf-8",
            )
        else:
            path.write_bytes(b"checkpoint")
        return path

    monkeypatch.setattr("app.separation.list_hf_files", fake_list_hf_files)
    monkeypatch.setattr("app.separation.download_hf_file", fake_download_hf_file)

    resolved = resolve_model(MODEL_SPECS["hyperace_v2_inst"], tmp_path)

    assert resolved.model_filename == "hyperace_v2_inst.ckpt"
    assert resolved.checkpoint_path.name == "model.ckpt"
    assert resolved.roformer_source_path is not None
    assert resolved.roformer_source_path.name == "bs_roformer.py"
    assert resolved.roformer_source_path.parent.name == "v2_inst"
    alias_config = resolved.config_path.read_text(encoding="utf-8")
    assert "- Vocals" in alias_config
    assert "- Instrumental" in alias_config
    assert "target_instrument: Instrumental" in alias_config
    assert "hyperace: true" in alias_config
    assert "hyperace_source_path:" in alias_config


def test_custom_separator_resolves_hf_model_paths(tmp_path: Path) -> None:
    class FakeSeparator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def download_model_files(self, model_filename):
            return ("fallback", "MDX", "Fallback", "/tmp/fallback.onnx", None)

    checkpoint_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    resolved = ResolvedModel(
        model_filename="hyperace_v2_inst.ckpt",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        friendly_name="HyperACE v2 instrumental",
    )

    separator_class = create_hf_separator_class(FakeSeparator)
    separator = separator_class(resolved_models=[resolved], output_dir=str(tmp_path))

    assert separator.download_model_files("hyperace_v2_inst.ckpt") == (
        "hyperace_v2_inst.ckpt",
        "MDXC",
        "HyperACE v2 instrumental",
        str(checkpoint_path),
        str(config_path),
    )
    assert separator.download_model_files("other.onnx")[0] == "fallback"


def test_detects_nested_mel_band_roformer_config() -> None:
    assert _is_mel_band_roformer_config({"model": {"num_bands": 60}})
    assert not _is_mel_band_roformer_config({"model": {"freqs_per_bands": [1, 2, 3]}})


def test_coerces_roformer_tuple_fields() -> None:
    config = {"multi_stft_resolutions_window_sizes": [4096, 2048], "freqs_per_bands": [2, 2]}

    _coerce_roformer_tuple_fields(config)

    assert config["multi_stft_resolutions_window_sizes"] == (4096, 2048)
    assert config["freqs_per_bands"] == (2, 2)


def test_ensure_cuda_is_preferred_sets_torch_and_onnx_provider(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(name: str) -> str:
            return name

    class FakeSeparator:
        torch_device = None
        onnx_execution_provider = ["CPUExecutionProvider"]

    separator = FakeSeparator()
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    monkeypatch.setattr(
        "app.separation._onnx_cuda_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    _ensure_cuda_is_preferred(separator)

    assert separator.torch_device == "cuda"
    assert separator.onnx_execution_provider == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_roformer_gpu_overlap_add_preserves_single_stem_output_shape() -> None:
    from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

    class EchoModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return batch * 0.5

    _enable_roformer_gpu_overlap_add()
    fake_separator = SimpleNamespace(
        is_roformer=True,
        pitch_shift=0,
        sample_rate=2,
        override_model_segment_size=False,
        segment_size=0,
        model_data_cfgdict=SimpleNamespace(
            inference=SimpleNamespace(dim_t=3),
            model=SimpleNamespace(stft_hop_length=2),
            audio=SimpleNamespace(sample_rate=2, hop_length=2),
            training=SimpleNamespace(
                target_instrument=INSTRUMENTAL_STEM,
                instruments=[INSTRUMENTAL_STEM],
            ),
        ),
        logger=logging.getLogger(__name__),
        model_run=EchoModel(),
        overlap=1,
        is_primary_stem_main_target=False,
        primary_stem_name=INSTRUMENTAL_STEM,
        secondary_stem_name=VOCALS_STEM,
    )
    fake_separator.overlap_add = MDXCSeparator.overlap_add.__get__(
        fake_separator,
        type(fake_separator),
    )
    mix = np.arange(16, dtype=np.float32).reshape(2, 8)

    output = MDXCSeparator.demix(fake_separator, mix)

    assert output.shape == mix.shape
    np.testing.assert_allclose(output, mix * 0.5, rtol=1e-5, atol=1e-5)


def test_separate_runs_instrumental_models_only(monkeypatch, tmp_path: Path) -> None:
    resolved = {
        key: ResolvedModel(
            model_filename=f"{key}.ckpt",
            checkpoint_path=tmp_path / f"{key}.ckpt",
            config_path=tmp_path / f"{key}.yaml",
            friendly_name=key,
        )
        for key in ("hyperace_v2_inst", "deux")
    }
    separate_calls = []
    ensemble_calls = []

    def fake_resolve_separator_models(model_dir: str):
        return resolved

    def fake_separate_model_stems(
        input_path: Path,
        *,
        output_dir: Path,
        model_dir: Path,
        model: ResolvedModel,
        output_single_stem: str | None,
        output_names: dict[str, str],
    ) -> dict[str, Path]:
        separate_calls.append((model.model_filename, output_single_stem, output_names))
        outputs = {}
        for stem, name in output_names.items():
            path = output_dir / f"{name}.wav"
            path.write_bytes(b"stem")
            outputs[stem] = path
        return outputs

    def fake_ensemble_stem_pair(source_paths: list[Path], target_path: Path) -> None:
        ensemble_calls.append(([path.name for path in source_paths], target_path.name))
        target_path.write_bytes(b"ensemble")

    monkeypatch.setattr("app.separation.resolve_separator_models", fake_resolve_separator_models)
    monkeypatch.setattr("app.separation._separate_model_stems", fake_separate_model_stems)
    monkeypatch.setattr("app.separation._ensemble_stem_pair", fake_ensemble_stem_pair)

    result = separate_instrumental(
        tmp_path / "source.wav",
        track_id="track-1",
        output_dir=tmp_path / "stems",
        model_dir=tmp_path / "models",
    )

    assert [call[0] for call in separate_calls] == [
        "hyperace_v2_inst.ckpt",
        "deux.ckpt",
    ]
    assert separate_calls[1][1] is None
    assert separate_calls[1][2] == {
        "Vocals": "deux_vocals",
        "Instrumental": "deux_instrumental",
    }
    assert ensemble_calls == [
        (
            ["hyperace_instrumental.wav", "deux_instrumental.wav"],
            "track-1.instrumental.wav",
        ),
    ]
    assert result.instrumental_path.exists()
