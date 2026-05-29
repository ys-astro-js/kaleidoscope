from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DISCOGS_VINET_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/raraz15/Discogs-VINet/main/"
    "logs/checkpoints/Discogs-VINet-MIREX-full_set/model_checkpoint.pth"
)
DISCOGS_VINET_CHECKPOINT_NAME = "discogs-vinet-mirex-full-set.pth"
DISCOGS_VINET_SAMPLE_RATE = 22_050
DISCOGS_VINET_HOP_LENGTH = 512
DISCOGS_VINET_CQT_BINS = 84
DISCOGS_VINET_BINS_PER_OCTAVE = 12
DISCOGS_VINET_CONTEXT_LENGTH = 7_600
DISCOGS_VINET_DOWNSAMPLE_FACTOR = 20
DISCOGS_VINET_CHUNK_OVERLAP = 0.5


@dataclass(frozen=True)
class CoverIdentityFeature:
    global_embedding: list[float]
    chunk_embeddings: list[list[float]]
    chunk_start_seconds: list[float]


@dataclass(frozen=True)
class CoverIdentityScores:
    global_score: float | None = None
    best_segment_score: float | None = None
    alignment_consistency: float | None = None


def extract_cover_identity_feature(
    audio_path: Path,
    model_dir: Path,
) -> CoverIdentityFeature | None:
    return get_discogs_vinet_embedder(model_dir).embed_file(audio_path)


def get_discogs_vinet_embedder(model_dir: Path) -> "DiscogsVINetEmbedder":
    return _cached_discogs_vinet_embedder(str(model_dir.resolve()))


@lru_cache(maxsize=2)
def _cached_discogs_vinet_embedder(model_dir: str) -> "DiscogsVINetEmbedder":
    return DiscogsVINetEmbedder(Path(model_dir))


def cover_identity_scores(
    query: CoverIdentityFeature,
    candidate: CoverIdentityFeature,
) -> CoverIdentityScores:
    return CoverIdentityScores(
        global_score=_cosine_score(query.global_embedding, candidate.global_embedding),
        best_segment_score=_top_chunk_score(query.chunk_embeddings, candidate.chunk_embeddings),
        alignment_consistency=_chunk_consistency_score(
            query.chunk_embeddings,
            candidate.chunk_embeddings,
        ),
    )


class DiscogsVINetEmbedder:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.checkpoint_path = self.model_dir / DISCOGS_VINET_CHECKPOINT_NAME
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.amp = self.device.type == "cuda"
        self.model = self._load_model()

    def embed_file(self, audio_path: Path) -> CoverIdentityFeature | None:
        cqt = _extract_magnitude_cqt(audio_path)
        if cqt is None:
            return None

        global_embedding = self._infer(cqt[None, None, :, :])[0]
        chunks, chunk_start_seconds = _chunk_cqt(cqt)
        chunk_embeddings = self._infer(chunks[:, None, :, :])
        return CoverIdentityFeature(
            global_embedding=global_embedding.astype(float).tolist(),
            chunk_embeddings=[embedding.astype(float).tolist() for embedding in chunk_embeddings],
            chunk_start_seconds=[float(seconds) for seconds in chunk_start_seconds],
        )

    def _load_model(self) -> nn.Module:
        checkpoint_path = _ensure_checkpoint(self.checkpoint_path)
        model = CQTNet(
            ch_in=40,
            ch_out=512,
            norm="bn",
            pool="adaptive_max",
            l2_normalize=True,
            projection="linear",
        ).to(self.device)
        checkpoint = _torch_load(checkpoint_path, self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    @torch.inference_mode()
    def _infer(self, cqt_batch: np.ndarray) -> np.ndarray:
        cqt_tensor = torch.as_tensor(cqt_batch, dtype=torch.float32, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            embeddings = self.model(cqt_tensor)
        return embeddings.detach().cpu().numpy()


class CQTNet(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        norm: str,
        pool: str,
        l2_normalize: bool,
        projection: str,
    ) -> None:
        super().__init__()
        self.l2_normalize = l2_normalize

        if norm.lower() == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm.lower() == "ibn":
            norm_layer = IBN
        elif norm.lower() == "in":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported Discogs-VINet norm: {norm}")

        self.front_end = nn.Sequential(
            nn.Conv2d(1, ch_in, kernel_size=(12, 3), dilation=(1, 1), padding=(6, 0), bias=False),
            norm_layer(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, 2 * ch_in, kernel_size=(13, 3), dilation=(1, 2), bias=False),
            norm_layer(2 * ch_in),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(2 * ch_in, 2 * ch_in, kernel_size=(13, 3), dilation=(1, 1), bias=False),
            norm_layer(2 * ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * ch_in, 2 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            norm_layer(2 * ch_in),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(2 * ch_in, 4 * ch_in, kernel_size=(3, 3), dilation=(1, 1), bias=False),
            norm_layer(4 * ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * ch_in, 4 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            norm_layer(4 * ch_in),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(4 * ch_in, 8 * ch_in, kernel_size=(3, 3), dilation=(1, 1), bias=False),
            norm_layer(8 * ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * ch_in, 8 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            norm_layer(8 * ch_in),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(8 * ch_in, 16 * ch_in, kernel_size=(3, 3), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(16 * ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 * ch_in, 16 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(16 * ch_in),
            nn.ReLU(inplace=True),
        )
        if pool.lower() == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool.lower() == "gem":
            self.pool = GeM()
        elif pool.lower() == "softpool":
            self.pool = SoftPool(16 * ch_in)
        else:
            raise ValueError(f"Unsupported Discogs-VINet pool: {pool}")

        if projection.lower() == "linear":
            self.proj = Linear(16 * ch_in, ch_out, bias=False)
        elif projection.lower() == "affine":
            self.proj = Linear(16 * ch_in, ch_out)
        elif projection.lower() == "mlp":
            self.proj = nn.Sequential(
                Linear(16 * ch_in, 32 * ch_in),
                nn.ReLU(inplace=True),
                Linear(32 * ch_in, ch_out, bias=False),
            )
        elif projection.lower() == "none":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported Discogs-VINet projection: {projection}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_end(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        if self.l2_normalize:
            x = _l2_normalize(x)
        return x


class IBN(nn.Module):
    def __init__(self, planes: int) -> None:
        super().__init__()
        self.ndim = planes // 2
        self.instance_norm = nn.InstanceNorm2d(self.ndim, affine=True)
        self.batch_norm = nn.BatchNorm2d(self.ndim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.split(x, self.ndim, 1)
        return torch.cat(
            (self.instance_norm(first.contiguous()), self.batch_norm(second.contiguous())),
            1,
        )


class GeM(nn.Module):
    def __init__(self, p: torch.Tensor = torch.log(torch.tensor(3.0)), eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.exp(self.p)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class Linear(nn.Module):
    def __init__(self, nin: int, nout: int, dim: int = -1, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(nin, nout, bias=bias)
        self.dim = dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.dim != -1:
            h = h.transpose(self.dim, -1)
        h = self.lin(h)
        if self.dim != -1:
            h = h.transpose(self.dim, -1)
        return h


class SoftPool(nn.Module):
    def __init__(self, ncha: int) -> None:
        super().__init__()
        self.lin = Linear(ncha, 2 * ncha, dim=1, bias=False)
        self.norm = nn.InstanceNorm1d(ncha, affine=True)
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.flatten(h)
        h = self.lin(h)
        h, a = torch.chunk(h, 2, dim=1)
        a = torch.softmax(self.norm(a), dim=-1)
        return (h * a).sum(dim=-1)


def _extract_magnitude_cqt(audio_path: Path) -> np.ndarray | None:
    import librosa

    audio, _ = librosa.load(audio_path, sr=DISCOGS_VINET_SAMPLE_RATE, mono=True)
    if len(audio) < DISCOGS_VINET_SAMPLE_RATE:
        return None

    cqt = librosa.core.cqt(
        y=audio,
        sr=DISCOGS_VINET_SAMPLE_RATE,
        hop_length=DISCOGS_VINET_HOP_LENGTH,
        n_bins=DISCOGS_VINET_CQT_BINS,
        bins_per_octave=DISCOGS_VINET_BINS_PER_OCTAVE,
    )
    cqt = np.abs(cqt.T).astype(np.float16).astype(np.float32)
    if cqt.size == 0 or np.isnan(cqt).any() or np.isinf(cqt).any():
        return None
    if cqt.shape[0] < DISCOGS_VINET_CONTEXT_LENGTH:
        cqt = np.pad(
            cqt,
            ((0, DISCOGS_VINET_CONTEXT_LENGTH - cqt.shape[0]), (0, 0)),
            "constant",
            constant_values=0,
        )
    cqt = _mean_downsample_cqt(cqt, DISCOGS_VINET_DOWNSAMPLE_FACTOR)
    cqt = np.where(cqt < 0, 0, cqt)
    cqt /= float(np.max(cqt)) + 1e-6
    return cqt.T


def _chunk_cqt(cqt: np.ndarray) -> tuple[np.ndarray, list[float]]:
    window = DISCOGS_VINET_CONTEXT_LENGTH // DISCOGS_VINET_DOWNSAMPLE_FACTOR
    if cqt.shape[1] <= window:
        return cqt[None, :, :], [0.0]

    step = max(1, int(window * (1.0 - DISCOGS_VINET_CHUNK_OVERLAP)))
    starts = list(range(0, cqt.shape[1] - window + 1, step))
    final_start = cqt.shape[1] - window
    if starts[-1] != final_start:
        starts.append(final_start)

    chunks = np.stack([cqt[:, start : start + window] for start in starts])
    seconds_per_downsampled_frame = (
        DISCOGS_VINET_HOP_LENGTH
        * DISCOGS_VINET_DOWNSAMPLE_FACTOR
        / DISCOGS_VINET_SAMPLE_RATE
    )
    return chunks, [start * seconds_per_downsampled_frame for start in starts]


def _mean_downsample_cqt(cqt: np.ndarray, mean_window_length: int) -> np.ndarray:
    frame_count, bin_count = cqt.shape
    new_frame_count = int(frame_count // mean_window_length)
    downsampled = np.zeros((new_frame_count, bin_count), dtype=cqt.dtype)
    for index in range(new_frame_count):
        downsampled[index, :] = cqt[
            index * mean_window_length : (index + 1) * mean_window_length,
            :,
        ].mean(axis=0)
    return downsampled


def _ensure_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        urlretrieve(DISCOGS_VINET_CHECKPOINT_URL, tmp_path)
        tmp_path.replace(checkpoint_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return checkpoint_path


def _torch_load(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = torch.norm(x, p=2, dim=1, keepdim=True)
    norms = norms + ((norms == 0).type_as(norms) * eps)
    return x / norms


def _top_chunk_score(
    query_embeddings: list[list[float]],
    candidate_embeddings: list[list[float]],
) -> float | None:
    if not query_embeddings or not candidate_embeddings:
        return None
    scores = [
        score
        for query in query_embeddings
        for candidate in candidate_embeddings
        if (score := _cosine_score(query, candidate)) is not None
    ]
    if not scores:
        return None
    return max(scores)


def _chunk_consistency_score(
    query_embeddings: list[list[float]],
    candidate_embeddings: list[list[float]],
) -> float | None:
    if not query_embeddings or not candidate_embeddings:
        return None
    target_matches = min(len(query_embeddings), len(candidate_embeddings), 6)
    coverage_target = min(max(len(query_embeddings), len(candidate_embeddings)), 6)
    scored_pairs = sorted(
        (
            (score, query_index, candidate_index)
            for query_index, query in enumerate(query_embeddings)
            for candidate_index, candidate in enumerate(candidate_embeddings)
            if (score := _cosine_score(query, candidate)) is not None
        ),
        reverse=True,
    )
    used_queries = set()
    used_candidates = set()
    selected_scores: list[float] = []
    for score, query_index, candidate_index in scored_pairs:
        if query_index in used_queries or candidate_index in used_candidates:
            continue
        selected_scores.append(score)
        used_queries.add(query_index)
        used_candidates.add(candidate_index)
        if len(selected_scores) >= target_matches:
            break
    if not selected_scores:
        return None
    coverage = len(selected_scores) / coverage_target
    return _clamp_score(float(np.mean(selected_scores)) * coverage)


def _cosine_score(query: list[float], candidate: list[float]) -> float | None:
    query_array = _as_normalized_array(query)
    candidate_array = _as_normalized_array(candidate)
    if query_array is None or candidate_array is None:
        return None
    if len(query_array) != len(candidate_array):
        return None
    return _clamp_score(float(np.dot(query_array, candidate_array)))


def _as_normalized_array(vector: list[float]) -> np.ndarray | None:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return array / norm


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))
