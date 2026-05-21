from functools import lru_cache

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class MuQEmbedder:
    def __init__(self, model_id: str, sample_rate: int) -> None:
        self.model_id = model_id
        self.sample_rate = sample_rate
        self._model = None
        self._device = None

    def embed_file(self, path: str) -> list[float]:
        import librosa
        import torch

        model, device = self._load()
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        wavs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(wavs, output_hidden_states=True)
            pooled = output.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()

        return normalize_vector(pooled).tolist()

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

