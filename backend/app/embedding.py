from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_SEGMENT_SECONDS = 30.0
DEFAULT_HOP_SECONDS = 15.0
MIN_ACTIVE_RMS_RATIO = 0.05
MIN_SEGMENT_ERROR = "Audio does not contain a usable 30 second segment"


@dataclass(frozen=True)
class TrackEmbeddings:
    global_semantic: list[float]
    segment_semantic: list[list[float]]

    @property
    def average(self) -> list[float]:
        return self.global_semantic

    @property
    def segments(self) -> list[list[float]]:
        return self.segment_semantic


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def select_audio_segments(
    wav: np.ndarray,
    sample_rate: int,
    *,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_rms_ratio: float = MIN_ACTIVE_RMS_RATIO,
) -> list[np.ndarray]:
    segment_samples = max(1, int(sample_rate * segment_seconds))
    hop_samples = max(1, int(sample_rate * hop_seconds))
    total_samples = len(wav)
    if total_samples < segment_samples:
        return []

    segments = [
        wav[start : start + segment_samples].astype(np.float32, copy=False)
        for start in range(0, total_samples - segment_samples + 1, hop_samples)
    ]
    rms_values = np.asarray([_rms(segment) for segment in segments], dtype=np.float32)
    peak_rms = float(rms_values.max()) if len(rms_values) else 0.0
    if peak_rms <= 0.0:
        return []

    threshold = peak_rms * max(0.0, min_rms_ratio)
    return [
        segment
        for segment, rms in zip(segments, rms_values, strict=True)
        if rms > 0.0 and rms >= threshold
    ]


def aggregate_segment_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.asarray([], dtype=np.float32)

    normalized = np.asarray([normalize_vector(vector) for vector in vectors], dtype=np.float32)
    return normalize_vector(normalized.mean(axis=0))


def _rms(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float32))))


class MuQEmbedder:
    def __init__(self, model_id: str, sample_rate: int) -> None:
        self.model_id = model_id
        self.sample_rate = sample_rate
        self._model = None
        self._device = None

    def embed_file(self, path: str) -> TrackEmbeddings:
        import librosa
        import torch

        wav = None
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)

        try:
            segments = select_audio_segments(wav, self.sample_rate)
            if not segments:
                raise ValueError(MIN_SEGMENT_ERROR)

            model, device = self._load()
            segment_vectors = []
            with torch.inference_mode():
                for segment in segments:
                    wavs = torch.from_numpy(segment).to(dtype=torch.float32, device=device).unsqueeze(0)
                    output = model(wavs, output_hidden_states=False)
                    pooled = output.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
                    segment_vectors.append(pooled)
                    del output, pooled, wavs
            segment_embeddings = [normalize_vector(vector).tolist() for vector in segment_vectors]
            global_embedding = aggregate_segment_vectors(segment_vectors).tolist()
            return TrackEmbeddings(
                global_semantic=global_embedding,
                segment_semantic=segment_embeddings,
            )
        finally:
            del wav
            if self._device == "cuda":
                torch.cuda.empty_cache()

    def _load(self):
        if self._model is not None and self._device is not None:
            return self._model, self._device

        import torch
        from muq import MuQ

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = MuQ.from_pretrained(self.model_id).to(self._device).eval()
        return self._model, self._device


@lru_cache
def get_embedder(model_id: str, sample_rate: int) -> MuQEmbedder:
    return MuQEmbedder(model_id, sample_rate)
