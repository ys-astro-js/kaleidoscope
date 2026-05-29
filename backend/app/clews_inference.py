import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.cover_alignment import CoverAlignmentFeature

CLEWS_CHECKPOINT_NAME = "checkpoint_best.ckpt"
CLEWS_CONFIG_NAME = "configuration.yaml"
CLEWS_SOURCE_MANIFEST_NAME = "source_manifest.json"
CLEWS_SOURCE_DIR_NAME = "source"
CLEWS_SEGMENT_HOP_SECONDS = 5.0


@dataclass(frozen=True)
class ClewsAssetPaths:
    checkpoint_path: Path
    config_path: Path
    source_dir: Path


def find_clews_assets(model_dir: Path) -> ClewsAssetPaths | None:
    checkpoint_path = _find_checkpoint(model_dir)
    if checkpoint_path is None:
        return None

    config_path = _find_config(model_dir, checkpoint_path)
    if config_path is None:
        return None

    source_dir = _find_source_dir(model_dir)
    if source_dir is None:
        return None

    return ClewsAssetPaths(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        source_dir=source_dir,
    )


def clews_asset_manifest(model_dir: Path) -> dict[str, str] | None:
    manifest_path = model_dir / CLEWS_SOURCE_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): str(value) for key, value in payload.items()}


class ClewsEmbedder:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        segment_hop_seconds: float = CLEWS_SEGMENT_HOP_SECONDS,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.segment_hop_seconds = segment_hop_seconds
        self._model = None
        self._audio_utils = None
        self._sample_rate = None
        self._device = None

    def embed_file(self, audio_path: Path) -> CoverAlignmentFeature | None:
        model, audio_utils, sample_rate, device = self._load()
        audio = audio_utils.load_audio(
            str(audio_path),
            sample_rate=sample_rate,
            n_channels=1,
        )
        if audio is None:
            audio = _load_audio_with_librosa(audio_path, sample_rate)
        if audio is None:
            return None

        torch = importlib.import_module("torch")
        with torch.inference_mode():
            if hasattr(audio, "to"):
                audio = audio.to(device)
            output = model(
                audio,
                shingle_hop=self.segment_hop_seconds,
                shingle_len=None,
            )
        segments = _segments_from_output(output)
        embeddings = [_normalize_vector(segment) for segment in segments]
        embeddings = [embedding for embedding in embeddings if embedding is not None]
        if not embeddings:
            return None

        global_embedding = _normalize_vector(np.asarray(embeddings, dtype=np.float32).mean(axis=0))
        if global_embedding is None:
            return None

        return CoverAlignmentFeature(
            model_key="clews",
            global_embedding=global_embedding.tolist(),
            segment_embeddings=[embedding.tolist() for embedding in embeddings],
            segment_start_seconds=[
                index * self.segment_hop_seconds
                for index in range(len(embeddings))
            ],
        )

    def _load(self):
        if (
            self._model is not None
            and self._audio_utils is not None
            and self._sample_rate is not None
            and self._device is not None
        ):
            return self._model, self._audio_utils, self._sample_rate, self._device

        assets = _assets_from_checkpoint(self.checkpoint_path)
        self._model, self._audio_utils, self._sample_rate, self._device = _load_official_clews(
            assets,
        )
        return self._model, self._audio_utils, self._sample_rate, self._device


def _assets_from_checkpoint(checkpoint_path: Path) -> ClewsAssetPaths:
    model_dir = checkpoint_path.parent
    assets = find_clews_assets(model_dir)
    if assets is None:
        raise FileNotFoundError(
            f"CLEWS assets are incomplete under {model_dir}. "
            f"Expected {CLEWS_CHECKPOINT_NAME}, {CLEWS_CONFIG_NAME}, and official source files."
        )
    return assets


def _load_official_clews(assets: ClewsAssetPaths):
    _prepend_sys_path(assets.source_dir)
    _clear_shadowed_clews_modules(assets.source_dir)

    torch = importlib.import_module("torch")
    lightning = importlib.import_module("lightning")
    omegaconf = importlib.import_module("omegaconf")
    audio_utils = importlib.import_module("utils.audio_utils")
    pytorch_utils = importlib.import_module("utils.pytorch_utils")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    conf = omegaconf.OmegaConf.load(str(assets.config_path))
    model_module = importlib.import_module(f"models.{conf.model.name}")
    sample_rate = int(conf.data.samplerate)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fabric = lightning.Fabric(accelerator=device, devices=1)
    fabric.launch()
    with fabric.init_module():
        model = model_module.Model(conf.model, sr=sample_rate)
    model = fabric.setup(model)
    state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
    fabric.load(str(assets.checkpoint_path), state, weights_only=False)
    model, _, _, _, _, _, _ = pytorch_utils.set_state(state)
    model.eval()
    return model, audio_utils, sample_rate, device


def _find_checkpoint(model_dir: Path) -> Path | None:
    for name in (CLEWS_CHECKPOINT_NAME, "model.ckpt"):
        path = model_dir / name
        if path.exists():
            return path
    matches = sorted(model_dir.glob("*.ckpt"))
    return matches[0] if matches else None


def _find_config(model_dir: Path, checkpoint_path: Path) -> Path | None:
    for path in (
        checkpoint_path.with_name(CLEWS_CONFIG_NAME),
        checkpoint_path.parent / CLEWS_CONFIG_NAME,
        model_dir / CLEWS_CONFIG_NAME,
    ):
        if path.exists():
            return path
    matches = sorted(model_dir.glob("*.yaml"))
    return matches[0] if matches else None


def _find_source_dir(model_dir: Path) -> Path | None:
    root = model_dir / CLEWS_SOURCE_DIR_NAME
    candidates = [root, *sorted(path for path in root.glob("*") if path.is_dir())] if root.exists() else []
    for candidate in candidates:
        if (candidate / "inference.py").exists() and (candidate / "models").exists():
            return candidate
    return None


def _prepend_sys_path(path: Path) -> None:
    path_text = str(path.resolve())
    if sys.path and sys.path[0] == path_text:
        return
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


def _clear_shadowed_clews_modules(source_dir: Path) -> None:
    for module_name in ("models", "utils", "lib"):
        module = sys.modules.get(module_name)
        if module is None or _module_is_under_path(module, source_dir):
            continue
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name, None)


def _module_is_under_path(module, root: Path) -> bool:
    root = root.resolve()
    module_paths = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        module_paths.append(Path(module_file))
    module_search_paths = getattr(module, "__path__", None)
    if module_search_paths:
        module_paths.extend(Path(path) for path in module_search_paths)

    for module_path in module_paths:
        try:
            resolved = module_path.resolve()
        except OSError:
            continue
        if resolved == root or resolved.is_relative_to(root):
            return True
    return False


def _segments_from_output(output) -> np.ndarray:
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    return array


def _load_audio_with_librosa(audio_path: Path, sample_rate: int):
    librosa = importlib.import_module("librosa")
    torch = importlib.import_module("torch")
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return None
    return torch.from_numpy(array).unsqueeze(0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return (array / norm).astype(np.float32)
