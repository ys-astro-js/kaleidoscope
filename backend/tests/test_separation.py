from pathlib import Path

from app.separation import (
    MODEL_SPECS,
    ResolvedModel,
    create_hf_separator_class,
    _coerce_roformer_tuple_fields,
    _is_mel_band_roformer_config,
    resolve_model,
)


def test_resolve_model_downloads_checkpoint_and_writes_alias_config(monkeypatch, tmp_path: Path) -> None:
    def fake_list_hf_files(repo_id: str) -> list[str]:
        assert repo_id == MODEL_SPECS["hyperace_v2_voc"].repo_id
        return [
            "bs_roformer.py",
            "v2_voc/bs_roformer.py",
            "v2_voc/config.yaml",
            "v2_voc/model.ckpt",
            "v2_inst/config.yaml",
            "v2_inst/model.ckpt",
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
                        "  target_instrument: vocals",
                    ]
                ),
                encoding="utf-8",
            )
        else:
            path.write_bytes(b"checkpoint")
        return path

    monkeypatch.setattr("app.separation.list_hf_files", fake_list_hf_files)
    monkeypatch.setattr("app.separation.download_hf_file", fake_download_hf_file)

    resolved = resolve_model(MODEL_SPECS["hyperace_v2_voc"], tmp_path)

    assert resolved.model_filename == "hyperace_v2_voc.ckpt"
    assert resolved.checkpoint_path.name == "model.ckpt"
    assert resolved.roformer_source_path is not None
    assert resolved.roformer_source_path.name == "bs_roformer.py"
    assert resolved.roformer_source_path.parent.name == "v2_voc"
    alias_config = resolved.config_path.read_text(encoding="utf-8")
    assert "- Vocals" in alias_config
    assert "- Instrumental" in alias_config
    assert "target_instrument: Vocals" in alias_config
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
        model_filename="hyperace_v2_voc.ckpt",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        friendly_name="HyperACE v2 vocals",
    )

    separator_class = create_hf_separator_class(FakeSeparator)
    separator = separator_class(resolved_models=[resolved], output_dir=str(tmp_path))

    assert separator.download_model_files("hyperace_v2_voc.ckpt") == (
        "hyperace_v2_voc.ckpt",
        "MDXC",
        "HyperACE v2 vocals",
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
