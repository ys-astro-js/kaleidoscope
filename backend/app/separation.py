from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types
from typing import Any


HYPERACE_REPO_ID = "pcunwa/BS-Roformer-HyperACE"
DEUX_REPO_ID = "becruily/mel-band-roformer-deux"
HYPERACE_SOURCE_FILE = "bs_roformer.py"
VOCALS_STEM = "Vocals"
INSTRUMENTAL_STEM = "Instrumental"
ENSEMBLE_ALGORITHM = "avg_fft"
OUTPUT_FORMAT = "WAV"
SEPARATOR_SAMPLE_RATE = 44100


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
    instrumental_path: Path


MODEL_SPECS = {
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


def separate_instrumental(
    input_path: Path,
    *,
    track_id: str,
    output_dir: Path,
    model_dir: Path,
) -> StemSeparationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_models = resolve_separator_models(str(model_dir))
    instrumental_path = output_dir / f"{track_id}.instrumental.{OUTPUT_FORMAT.lower()}"

    if instrumental_path.exists():
        instrumental_path.unlink()

    with tempfile.TemporaryDirectory(prefix=f"{track_id}.stems.", dir=output_dir) as temp_dir:
        temp_path = Path(temp_dir)
        hyperace_instrumental = _separate_model_stems(
            input_path,
            output_dir=temp_path,
            model_dir=model_dir,
            model=resolved_models["hyperace_v2_inst"],
            output_single_stem=INSTRUMENTAL_STEM,
            output_names={INSTRUMENTAL_STEM: "hyperace_instrumental"},
        )[INSTRUMENTAL_STEM]
        deux_stems = _separate_model_stems(
            input_path,
            output_dir=temp_path,
            model_dir=model_dir,
            model=resolved_models["deux"],
            output_single_stem=None,
            output_names={
                VOCALS_STEM: "deux_vocals",
                INSTRUMENTAL_STEM: "deux_instrumental",
            },
        )

        _ensemble_stem_pair(
            [hyperace_instrumental, deux_stems[INSTRUMENTAL_STEM]],
            instrumental_path,
        )

    return StemSeparationResult(instrumental_path=instrumental_path)


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
    from audio_separator.separator.uvr_lib_v5.roformer import attend
    from audio_separator.separator.roformer import roformer_loader
    from audio_separator.separator.roformer.model_loading_result import (
        ImplementationVersion,
        ModelLoadingResult,
    )

    _enable_modern_cuda_flash_attention(attend)
    _enable_roformer_gpu_overlap_add()

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


def _enable_modern_cuda_flash_attention(attend_module) -> None:
    attend_class = getattr(attend_module, "Attend", None)
    flash_config = getattr(attend_module, "FlashAttentionConfig", None)
    if attend_class is None or flash_config is None:
        return
    if getattr(attend_class, "_kaleidoscope_modern_cuda_flash", False):
        return

    original_init = attend_class.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        flash_enabled = _attend_flash_arg(args, kwargs)
        if not flash_enabled:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            major, _ = torch.cuda.get_device_capability(torch.device("cuda"))
        except Exception:
            return
        if major >= 8:
            # audio-separator only forced flash SDPA for A100. Ampere/Ada GPUs also
            # support PyTorch flash SDPA for the fp16 RoFormer attention shapes.
            self.cuda_config = flash_config(True, False, False)

    attend_class.__init__ = patched_init
    attend_class._kaleidoscope_modern_cuda_flash = True


def _enable_roformer_gpu_overlap_add() -> None:
    from audio_separator.separator.architectures import mdxc_separator

    separator_class = mdxc_separator.MDXCSeparator
    if getattr(separator_class, "_kaleidoscope_gpu_overlap_add", False):
        return

    original_demix = separator_class.demix

    def patched_demix(self, mix):
        if not getattr(self, "is_roformer", False):
            return original_demix(self, mix)

        try:
            return _demix_roformer_with_gpu_overlap_add(self, mix, mdxc_separator)
        except RuntimeError as exc:
            if not _is_cuda_out_of_memory(exc):
                raise
            self.logger.warning(
                "RoFormer GPU overlap-add ran out of CUDA memory; retrying with the audio-separator CPU path."
            )
            mdxc_separator.torch.cuda.empty_cache()
            return original_demix(self, mix)

    separator_class.demix = patched_demix
    separator_class._kaleidoscope_gpu_overlap_add = True


def _demix_roformer_with_gpu_overlap_add(self, mix, mdxc_module):
    orig_mix = mix
    sample_rate = self.sample_rate

    if self.pitch_shift != 0:
        self.logger.debug(f"Shifting pitch by -{self.pitch_shift} semitones...")
        mix, sample_rate = mdxc_module.spec_utils.change_pitch_semitones(
            mix,
            self.sample_rate,
            semitone_shift=-self.pitch_shift,
        )

    device = next(self.model_run.parameters()).device
    mix = mdxc_module.torch.as_tensor(mix, dtype=mdxc_module.torch.float32, device=device)

    if self.override_model_segment_size:
        mdx_segment_size = self.segment_size
        self.logger.debug(f"Using configured segment size: {mdx_segment_size}")
    else:
        mdx_segment_size = self.model_data_cfgdict.inference.dim_t
        self.logger.debug(f"Using model default segment size: {mdx_segment_size}")

    num_stems = (
        1
        if self.model_data_cfgdict.training.target_instrument
        else len(self.model_data_cfgdict.training.instruments)
    )
    self.logger.debug(f"Number of stems: {num_stems}")

    stft_hop_len = getattr(self.model_data_cfgdict.model, "stft_hop_length", None)
    if stft_hop_len is None:
        stft_hop_len = self.model_data_cfgdict.audio.hop_length
        self.logger.debug(
            f"Model.stft_hop_length missing; falling back to audio.hop_length={stft_hop_len}"
        )

    chunk_size = int(stft_hop_len) * (int(mdx_segment_size) - 1)
    self.logger.debug(
        f"Chunk size: {chunk_size} (using stft_hop_length={stft_hop_len} and dim_t={mdx_segment_size})"
    )

    desired_step = int(self.overlap * self.model_data_cfgdict.audio.sample_rate)
    step = chunk_size if desired_step <= 0 else min(desired_step, chunk_size)
    self.logger.debug(f"Step: {step} (desired={desired_step})")

    window = mdxc_module.torch.as_tensor(
        mdxc_module.signal.windows.hamming(chunk_size),
        dtype=mdxc_module.torch.float32,
        device=device,
    )

    with mdxc_module.torch.no_grad():
        req_shape = (len(self.model_data_cfgdict.training.instruments),) + tuple(mix.shape)
        # Keep RoFormer overlap-add on the model device to avoid per-chunk CUDA/CPU synchronization.
        result = mdxc_module.torch.zeros(req_shape, dtype=mdxc_module.torch.float32, device=device)
        counter = mdxc_module.torch.zeros(req_shape, dtype=mdxc_module.torch.float32, device=device)

        for i in mdxc_module.tqdm(range(0, mix.shape[1], step)):
            part = mix[:, i : i + chunk_size]
            length = part.shape[-1]
            if i + chunk_size > mix.shape[1]:
                part = mix[:, -chunk_size:]
                length = chunk_size
            x = self.model_run(part.unsqueeze(0))[0]
            if i + chunk_size > mix.shape[1]:
                start_idx = result.shape[-1] - chunk_size
                result = self.overlap_add(result, x, window, start_idx, length)
                safe_len = min(length, x.shape[-1], window.shape[0])
                if safe_len > 0:
                    counter[..., start_idx : start_idx + safe_len] += window[:safe_len]
            else:
                result = self.overlap_add(result, x, window, i, length)
                safe_len = min(length, x.shape[-1], window.shape[0])
                if safe_len > 0:
                    counter[..., i : i + safe_len] += window[:safe_len]

    inferenced_outputs = result / counter.clamp(min=1e-10)
    return _finalize_demixed_outputs(self, inferenced_outputs, num_stems, sample_rate, orig_mix, mdxc_module)


def _finalize_demixed_outputs(
    self,
    inferenced_outputs,
    num_stems: int,
    sample_rate: int,
    orig_mix,
    mdxc_module,
):
    if num_stems > 1:
        self.logger.debug("Number of stems is greater than 1, detaching individual sources and correcting pitch if necessary...")

        sources = {}
        for key, value in zip(
            self.model_data_cfgdict.training.instruments,
            inferenced_outputs.cpu().detach().numpy(),
        ):
            self.logger.debug(f"Processing instrument: {key}")
            if self.pitch_shift != 0:
                self.logger.debug(f"Applying pitch correction for {key}")
                sources[key] = self.pitch_fix(value, sample_rate, orig_mix)
            else:
                sources[key] = value

        if self.is_primary_stem_main_target and num_stems == 1:
            self.logger.debug(f"Primary stem: {self.primary_stem_name} is main target, detaching and matching array shapes if necessary...")
            if sources[self.primary_stem_name].shape[1] != orig_mix.shape[1]:
                sources[self.primary_stem_name] = mdxc_module.spec_utils.match_array_shapes(
                    sources[self.primary_stem_name],
                    orig_mix,
                )
            sources[self.secondary_stem_name] = orig_mix - sources[self.primary_stem_name]

        self.logger.debug("Deleting inferenced outputs to free up memory")
        del inferenced_outputs

        self.logger.debug("Returning separated sources")
        return sources

    self.logger.debug("Processing single source...")

    sources = {
        key: value.cpu().detach().numpy()
        for key, value in zip(
            [self.model_data_cfgdict.training.target_instrument],
            inferenced_outputs,
        )
    }
    inferenced_output = sources[self.model_data_cfgdict.training.target_instrument]

    self.logger.debug("Demix process completed for single source.")

    self.logger.debug("Deleting inferenced outputs to free up memory")
    del inferenced_outputs

    if self.pitch_shift != 0:
        self.logger.debug("Applying pitch correction for single instrument")
        primary = self.pitch_fix(inferenced_output, sample_rate, orig_mix)
    else:
        primary = inferenced_output

    if self.is_primary_stem_main_target:
        self.logger.debug("Single-target model detected; computing residual secondary stem from original mix")
        if primary.shape[1] != orig_mix.shape[1]:
            primary = mdxc_module.spec_utils.match_array_shapes(primary, orig_mix)
        secondary = orig_mix - primary
        return {
            self.primary_stem_name: primary,
            self.secondary_stem_name: secondary,
        }

    self.logger.debug("Returning inferenced output for single instrument")
    return primary


def _is_cuda_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _attend_flash_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if "flash" in kwargs:
        return bool(kwargs["flash"])
    if len(args) >= 2:
        return bool(args[1])
    return False


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


def _separate_model_stems(
    input_path: Path,
    *,
    output_dir: Path,
    model_dir: Path,
    model: ResolvedModel,
    output_single_stem: str | None,
    output_names: dict[str, str],
) -> dict[str, Path]:
    configure_torch_cuda_for_separator()
    separator_class = create_hf_separator_class()
    separator = separator_class(
        resolved_models=[model],
        log_level=logging.WARNING,
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format=OUTPUT_FORMAT,
        output_single_stem=output_single_stem,
        use_autocast=True,
    )
    _ensure_cuda_is_preferred(separator)
    separator.load_model([model.model_filename])
    output_files = separator.separate(str(input_path), output_names)
    return _collect_stem_outputs(output_dir, output_names, output_files)


def _collect_stem_outputs(
    output_dir: Path,
    output_names: dict[str, str],
    output_files: list[str],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    output_paths = [Path(output_file) for output_file in output_files]
    for stem_name, output_name in output_names.items():
        expected_path = output_dir / f"{output_name}.{OUTPUT_FORMAT.lower()}"
        if expected_path.exists():
            outputs[stem_name] = expected_path
            continue

        for path in output_paths:
            if path.exists() and path.stem == output_name:
                outputs[stem_name] = path
                break
        else:
            raise RuntimeError(f"{stem_name} separation did not create an output file")
    return outputs


def _ensemble_stem_pair(source_paths: list[Path], target_path: Path) -> None:
    import librosa
    import soundfile as sf
    from audio_separator.separator.ensembler import Ensembler

    waveforms = []
    for path in source_paths:
        waveform, _ = librosa.load(path, mono=False, sr=SEPARATOR_SAMPLE_RATE)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        waveforms.append(waveform)

    ensembler = Ensembler(
        logging.getLogger(__name__),
        ENSEMBLE_ALGORITHM,
        weights=[1.0] * len(waveforms),
    )
    ensembled = ensembler.ensemble(waveforms)
    if ensembled is None:
        raise RuntimeError(f"No waveforms were available for {target_path.name}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target_path, ensembled.T, SEPARATOR_SAMPLE_RATE)


def configure_torch_cuda_for_separator() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _ensure_cuda_is_preferred(separator) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    separator.torch_device = torch.device("cuda")
    providers = _onnx_cuda_providers()
    if providers:
        separator.onnx_execution_provider = providers


def _onnx_cuda_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []

    available = ort.get_available_providers()
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    return providers


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
