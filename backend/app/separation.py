from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import logging
from pathlib import Path
import sys
import types
from typing import Any


HYPERACE_REPO_ID = "pcunwa/BS-Roformer-HyperACE"
DEUX_REPO_ID = "becruily/mel-band-roformer-deux"
HYPERACE_SOURCE_FILE = "bs_roformer.py"
VOCALS_STEM = "Vocals"
INSTRUMENTAL_STEM = "Instrumental"
ENSEMBLE_ALGORITHM = "avg_fft"
OUTPUT_FORMAT = "WAV"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    path_prefix: str
    target_stem: str
    friendly_name: str


@dataclass(frozen=True)
class ResolvedModel:
    model_filename: str
    checkpoint_path: Path
    config_path: Path
    friendly_name: str
    roformer_source_path: Path | None = None


@dataclass(frozen=True)
class StemSeparationResult:
    vocals_path: Path
    instrumental_path: Path


MODEL_SPECS = {
    "hyperace_v2_voc": ModelSpec(
        key="hyperace_v2_voc",
        repo_id=HYPERACE_REPO_ID,
        path_prefix="v2_voc/",
        target_stem=VOCALS_STEM,
        friendly_name="HyperACE v2 vocals",
    ),
    "hyperace_v2_inst": ModelSpec(
        key="hyperace_v2_inst",
        repo_id=HYPERACE_REPO_ID,
        path_prefix="v2_inst/",
        target_stem=INSTRUMENTAL_STEM,
        friendly_name="HyperACE v2 instrumental",
    ),
    "deux": ModelSpec(
        key="deux",
        repo_id=DEUX_REPO_ID,
        path_prefix="",
        target_stem=VOCALS_STEM,
        friendly_name="Mel-band RoFormer Deux",
    ),
}


def separate_vocals_and_instrumental(
    input_path: Path,
    *,
    track_id: str,
    output_dir: Path,
    model_dir: Path,
) -> StemSeparationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_models = resolve_separator_models(str(model_dir))

    vocals_path = _separate_single_stem(
        input_path,
        track_id=track_id,
        output_dir=output_dir,
        model_dir=model_dir,
        models=[resolved_models["hyperace_v2_voc"], resolved_models["deux"]],
        stem_name=VOCALS_STEM,
        output_name=f"{track_id}.vocals",
    )
    instrumental_path = _separate_single_stem(
        input_path,
        track_id=track_id,
        output_dir=output_dir,
        model_dir=model_dir,
        models=[resolved_models["hyperace_v2_inst"], resolved_models["deux"]],
        stem_name=INSTRUMENTAL_STEM,
        output_name=f"{track_id}.instrumental",
    )
    return StemSeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)


@lru_cache(maxsize=4)
def resolve_separator_models(model_dir: str) -> dict[str, ResolvedModel]:
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return {
        key: resolve_model(spec, target_dir / key)
        for key, spec in MODEL_SPECS.items()
    }


def resolve_model(spec: ModelSpec, local_dir: Path) -> ResolvedModel:
    files = list_hf_files(spec.repo_id)
    checkpoint_file = _select_repo_file(files, spec.path_prefix, (".ckpt", ".pth"))
    config_file = _select_repo_file(files, spec.path_prefix, (".yaml", ".yml"))

    checkpoint_path = download_hf_file(spec.repo_id, checkpoint_file, local_dir)
    config_path = download_hf_file(spec.repo_id, config_file, local_dir)
    roformer_source_path = None
    if spec.repo_id == HYPERACE_REPO_ID:
        source_file = _select_hyperace_source_file(files, spec.path_prefix)
        roformer_source_path = download_hf_file(spec.repo_id, source_file, local_dir)

    alias_config_path = local_dir / f"{spec.key}.roformer.yaml"
    write_alias_config(
        config_path,
        alias_config_path,
        hyperace_source_path=roformer_source_path,
    )

    return ResolvedModel(
        model_filename=f"{spec.key}.ckpt",
        checkpoint_path=checkpoint_path,
        config_path=alias_config_path,
        friendly_name=spec.friendly_name,
        roformer_source_path=roformer_source_path,
    )


def list_hf_files(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(repo_id, repo_type="model")


def download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=str(local_dir),
    )
    return Path(path)


def write_alias_config(
    source_path: Path,
    target_path: Path,
    *,
    hyperace_source_path: Path | None = None,
) -> Path:
    import yaml

    with source_path.open("r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid separator config: {source_path}")

    training = data.get("training")
    if isinstance(training, dict):
        instruments = training.get("instruments")
        if isinstance(instruments, list):
            training["instruments"] = [_canonical_stem_name(item) for item in instruments]
        target = training.get("target_instrument")
        if target is not None:
            training["target_instrument"] = _canonical_stem_name(target)

    if hyperace_source_path is not None:
        model = data.setdefault("model", {})
        if not isinstance(model, dict):
            raise ValueError(f"Invalid model config section: {source_path}")
        model["hyperace"] = True
        model["hyperace_source_path"] = str(hyperace_source_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_plain_yaml(data), file, sort_keys=False)
    return target_path


def create_hf_separator_class(separator_base=None):
    if separator_base is None:
        install_hyperace_roformer_loader()
        from audio_separator.separator import Separator as separator_base

    class HuggingFaceSeparator(separator_base):
        def __init__(self, *, resolved_models: list[ResolvedModel], **kwargs):
            super().__init__(**kwargs)
            self._resolved_models = {
                model.model_filename: model
                for model in resolved_models
            }

        def download_model_files(self, model_filename):
            model = self._resolved_models.get(model_filename)
            if model is None:
                return super().download_model_files(model_filename)
            return (
                model.model_filename,
                "MDXC",
                model.friendly_name,
                str(model.checkpoint_path),
                str(model.config_path),
            )

    return HuggingFaceSeparator


def install_hyperace_roformer_loader() -> None:
    from audio_separator.separator.roformer import roformer_loader
    from audio_separator.separator.roformer.model_loading_result import (
        ImplementationVersion,
        ModelLoadingResult,
    )

    loader_class = roformer_loader.RoformerLoader
    if getattr(loader_class, "_kaleidoscope_hyperace_loader", False):
        return

    original_load_model = loader_class.load_model

    def load_model(self, model_path: str, config: dict[str, Any], device: str = "cpu"):
        if not _is_hyperace_config(config):
            if _is_mel_band_roformer_config(config):
                return _load_mel_band_roformer_model(
                    self,
                    model_path,
                    config,
                    device,
                    ModelLoadingResult,
                    ImplementationVersion,
                )
            return original_load_model(self, model_path, config, device)

        try:
            normalized_config = self.config_normalizer.normalize_config(
                config,
                "bs_roformer",
                apply_defaults=True,
                validate=True,
            )
            model = _create_hyperace_model(normalized_config)
            state_dict = _load_torch_state_dict(model_path)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            result = ModelLoadingResult.success_result(
                model=model,
                implementation=ImplementationVersion.NEW,
                config=normalized_config,
            )
            result.add_model_info("model_type", "hyperace_bs_roformer")
            result.add_model_info("loading_method", "hf_custom_source")
            result.add_model_info("device", device)
            return result
        except Exception as exc:
            return ModelLoadingResult.failure_result(
                error_message=f"HyperACE RoFormer load failed: {exc}",
                implementation=ImplementationVersion.NEW,
            )

    loader_class.load_model = load_model
    loader_class._kaleidoscope_hyperace_loader = True


def _is_hyperace_config(config: dict[str, Any]) -> bool:
    return bool(_model_config(config).get("hyperace"))


def _is_mel_band_roformer_config(config: dict[str, Any]) -> bool:
    model_config = _model_config(config)
    return "num_bands" in model_config or "n_mels" in model_config or "mel_bands" in model_config


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model", config)
    return model_config if isinstance(model_config, dict) else {}


def _load_mel_band_roformer_model(
    loader,
    model_path: str,
    config: dict[str, Any],
    device: str,
    model_loading_result,
    implementation_version,
):
    try:
        normalized_config = loader.config_normalizer.normalize_config(
            config,
            "mel_band_roformer",
            apply_defaults=True,
            validate=True,
        )
        _coerce_roformer_tuple_fields(normalized_config)
        result = loader._load_with_new_implementation(
            model_path,
            normalized_config,
            "mel_band_roformer",
            device,
        )
        result.add_model_info("model_type_detection", "nested_config")
        return result
    except Exception as exc:
        return model_loading_result.failure_result(
            error_message=f"MelBand RoFormer load failed: {exc}",
            implementation=implementation_version.NEW,
        )


def _coerce_roformer_tuple_fields(config: dict[str, Any]) -> None:
    for key in ("freqs_per_bands", "multi_stft_resolutions_window_sizes"):
        if isinstance(config.get(key), list):
            config[key] = tuple(config[key])


def _create_hyperace_model(config: dict[str, Any]):
    source_path = config.get("hyperace_source_path")
    if not source_path:
        raise ValueError("HyperACE config is missing hyperace_source_path")

    BSRoformer = _load_hyperace_bs_roformer_class(str(source_path))
    return BSRoformer(**_hyperace_model_args(config))


def _load_torch_state_dict(model_path: str) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported HyperACE checkpoint format: {type(checkpoint).__name__}")
    return checkpoint


@lru_cache(maxsize=4)
def _load_hyperace_bs_roformer_class(source_path: str):
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"HyperACE source file not found: {source}")

    from audio_separator.separator.uvr_lib_v5.roformer import attend

    try:
        import torchaudio  # noqa: F401
    except Exception:
        sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    sys.modules.setdefault("models", types.ModuleType("models"))
    sys.modules.setdefault("models.bs_roformer", types.ModuleType("models.bs_roformer"))
    sys.modules["models.bs_roformer.attend"] = attend

    module_name = f"kaleidoscope_hyperace_bs_roformer_{abs(hash(source.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import HyperACE source from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BSRoformer


def _hyperace_model_args(config: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "dim": config["dim"],
        "depth": config["depth"],
        "stereo": config.get("stereo", False),
        "num_stems": config.get("num_stems", 1),
        "time_transformer_depth": config.get("time_transformer_depth", 2),
        "freq_transformer_depth": config.get("freq_transformer_depth", 2),
        "linear_transformer_depth": config.get("linear_transformer_depth", 0),
        "freqs_per_bands": tuple(config["freqs_per_bands"]),
        "dim_head": config.get("dim_head", 64),
        "heads": config.get("heads", 8),
        "attn_dropout": config.get("attn_dropout", 0.0),
        "ff_dropout": config.get("ff_dropout", 0.0),
        "flash_attn": config.get("flash_attn", True),
        "dim_freqs_in": config.get("dim_freqs_in", 1025),
        "stft_n_fft": config.get("stft_n_fft", 2048),
        "stft_hop_length": config.get("stft_hop_length", 512),
        "stft_win_length": config.get("stft_win_length", 2048),
        "stft_normalized": config.get("stft_normalized", False),
        "mask_estimator_depth": config.get("mask_estimator_depth", 2),
        "multi_stft_resolution_loss_weight": config.get("multi_stft_resolution_loss_weight", 1.0),
        "multi_stft_resolutions_window_sizes": tuple(
            config.get("multi_stft_resolutions_window_sizes", (4096, 2048, 1024, 512, 256))
        ),
        "multi_stft_hop_size": config.get("multi_stft_hop_size", 147),
        "multi_stft_normalized": config.get("multi_stft_normalized", False),
        "mlp_expansion_factor": config.get("mlp_expansion_factor", 4),
        "use_torch_checkpoint": config.get("use_torch_checkpoint", False),
        "skip_connection": config.get("skip_connection", False),
        "sage_attention": config.get("sage_attention", False),
    }
    return args


def _separate_single_stem(
    input_path: Path,
    *,
    track_id: str,
    output_dir: Path,
    model_dir: Path,
    models: list[ResolvedModel],
    stem_name: str,
    output_name: str,
) -> Path:
    target_path = output_dir / f"{output_name}.{OUTPUT_FORMAT.lower()}"
    if target_path.exists():
        target_path.unlink()

    separator_class = create_hf_separator_class()
    separator = separator_class(
        resolved_models=models,
        log_level=logging.WARNING,
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format=OUTPUT_FORMAT,
        output_single_stem=stem_name,
        ensemble_algorithm=ENSEMBLE_ALGORITHM,
        ensemble_weights=[1.0] * len(models),
    )
    separator.load_model([model.model_filename for model in models])
    output_files = separator.separate(str(input_path), {stem_name: output_name})

    if target_path.exists():
        return target_path

    for output_file in output_files:
        path = Path(output_file)
        if path.exists() and path.stem == output_name:
            return path

    raise RuntimeError(f"{stem_name} separation did not create an output file for {track_id}")


def _select_repo_file(files: list[str], path_prefix: str, suffixes: tuple[str, ...]) -> str:
    candidates = [
        file
        for file in files
        if file.startswith(path_prefix)
        and file.lower().endswith(suffixes)
        and "/" not in file[len(path_prefix):]
    ]
    if not candidates:
        raise ValueError(f"No {suffixes} file found under {path_prefix or 'repo root'}")
    return sorted(candidates, key=_repo_file_sort_key)[0]


def _select_hyperace_source_file(files: list[str], path_prefix: str) -> str:
    preferred = f"{path_prefix}{HYPERACE_SOURCE_FILE}"
    if preferred in files:
        return preferred
    if HYPERACE_SOURCE_FILE in files:
        return HYPERACE_SOURCE_FILE
    raise ValueError(f"No {HYPERACE_SOURCE_FILE} found for HyperACE model")


def _repo_file_sort_key(filename: str) -> tuple[int, str]:
    name = Path(filename).name.lower()
    return (0 if "config" in name else 1, filename)


def _canonical_stem_name(value: object) -> str:
    aliases = {
        "voc": VOCALS_STEM,
        "vocal": VOCALS_STEM,
        "vocals": VOCALS_STEM,
        "inst": INSTRUMENTAL_STEM,
        "instrum": INSTRUMENTAL_STEM,
        "instrument": INSTRUMENTAL_STEM,
        "instrumental": INSTRUMENTAL_STEM,
        "no_vocals": INSTRUMENTAL_STEM,
        "other": INSTRUMENTAL_STEM,
    }
    text = str(value)
    return aliases.get(text.strip().lower(), text)


def _plain_yaml(value):
    if isinstance(value, dict):
        return {key: _plain_yaml(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_yaml(item) for item in value]
    return value
