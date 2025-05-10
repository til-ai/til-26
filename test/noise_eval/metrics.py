"""Distortion / similarity metrics for comparing original and noised images.

Every metric is a callable dataclass-style object that takes two numpy HWC
uint8 images and returns a **scalar float** (lower = less distortion for
distance metrics, higher = less distortion for similarity metrics).

Performance note
----------------
``DistortionContext`` pre-computes the float32 difference array once per image
pair.  Metrics that override ``compute_from_context`` use this shared array
instead of re-casting both images independently.

For batched evaluation, ``BatchDistortionContext`` pre-computes the same arrays
over a stacked (B, H, W, C) batch.  Override ``compute_batch_from_context`` in
each metric for vectorized computation; the default falls back to a per-image
loop so any metric works in both modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity

# ====================================================================== #
#  Batched SSIM (GPU via torch)                                          #
# ====================================================================== #


def _batched_ssim_torch(
    orig_f: np.ndarray,  # (B, H, W, C) float32
    noised_f: np.ndarray,  # (B, H, W, C) float32
    win_size: int = 7,
    data_range: float = 255.0,
) -> np.ndarray:  # (B,)
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_t = torch.from_numpy(orig_f).permute(0, 3, 1, 2).contiguous().to(device)
    noised_t = torch.from_numpy(noised_f).permute(0, 3, 1, 2).contiguous().to(device)

    pad = win_size // 2
    n_channels = orig_t.size(1)
    NP = win_size * win_size
    cov_norm = NP / (NP - 1)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = (
        torch.ones(n_channels, 1, win_size, win_size, device=device, dtype=orig_t.dtype)
        / NP
    )

    mu1 = F.conv2d(orig_t, kernel, padding=pad, groups=n_channels)
    mu2 = F.conv2d(noised_t, kernel, padding=pad, groups=n_channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = cov_norm * (
        F.conv2d(orig_t * orig_t, kernel, padding=pad, groups=n_channels) - mu1_sq
    )
    sigma2_sq = cov_norm * (
        F.conv2d(noised_t * noised_t, kernel, padding=pad, groups=n_channels) - mu2_sq
    )
    sigma12 = cov_norm * (
        F.conv2d(orig_t * noised_t, kernel, padding=pad, groups=n_channels) - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if pad > 0:
        ssim_map = ssim_map[..., pad:-pad, pad:-pad]

    return ssim_map.mean(dim=1).detach().cpu().numpy().mean(axis=(1, 2))


# ====================================================================== #
#  Single-image context                                                  #
# ====================================================================== #


@dataclass(slots=True)
class DistortionContext:
    """Pre-computed artefacts shared across all metrics for one image pair."""

    original: np.ndarray
    noised: np.ndarray
    diff_f32: np.ndarray
    abs_diff: np.ndarray

    @classmethod
    def from_pair(cls, original: np.ndarray, noised: np.ndarray) -> "DistortionContext":
        diff = original.astype(np.float32) - noised.astype(np.float32)
        return cls(
            original=original, noised=noised, diff_f32=diff, abs_diff=np.abs(diff)
        )


# ====================================================================== #
#  Batch context                                                         #
# ====================================================================== #


@dataclass(slots=True)
class BatchDistortionContext:
    """Pre-computed artefacts shared across all metrics for a (B, H, W, C) batch."""

    originals: np.ndarray  # (B, H, W, C) uint8
    noised: np.ndarray  # (B, H, W, C) uint8
    diff_f32: np.ndarray  # (B, H, W, C) float32
    abs_diff: np.ndarray  # (B, H, W, C) float32

    @classmethod
    def from_batch(
        cls, originals: np.ndarray, noised: np.ndarray
    ) -> "BatchDistortionContext":
        diff = originals.astype(np.float32) - noised.astype(np.float32)
        return cls(
            originals=originals, noised=noised, diff_f32=diff, abs_diff=np.abs(diff)
        )


# ====================================================================== #
#  Base class                                                            #
# ====================================================================== #


class DistortionMetric(ABC):
    """Base class for all distortion / similarity metrics."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def higher_is_better(self) -> bool:
        return False

    @abstractmethod
    def compute(self, original: np.ndarray, noised: np.ndarray) -> float:
        """Return the metric value for a single HWC uint8 image pair."""
        ...

    def compute_from_context(self, ctx: DistortionContext) -> float:
        return self.compute(ctx.original, ctx.noised)

    def compute_batch_from_context(self, ctx: BatchDistortionContext) -> np.ndarray:
        """Return a (B,) array of per-image metric values.

        Override in subclasses for vectorized computation.  The default falls
        back to a per-image loop so any metric works in both modes.
        """
        return np.array(
            [
                self.compute(ctx.originals[i], ctx.noised[i])
                for i in range(len(ctx.originals))
            ]
        )

    def __call__(self, original: np.ndarray, noised: np.ndarray) -> float:
        return self.compute(original, noised)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ====================================================================== #
#  L2 (Euclidean) Distance                                               #
# ====================================================================== #


class L2Distance(DistortionMetric):
    """Root-mean-square L2 distance: ``sqrt( mean( (orig - noised)^2 ) )``"""

    @property
    def name(self) -> str:
        return "L2 (RMSE)"

    def compute(self, original: np.ndarray, noised: np.ndarray) -> float:
        diff = original.astype(np.float32) - noised.astype(np.float32)
        return float(np.sqrt(np.mean(diff**2)))

    def compute_from_context(self, ctx: DistortionContext) -> float:
        return float(np.sqrt(np.mean(ctx.diff_f32**2)))

    def compute_batch_from_context(self, ctx: BatchDistortionContext) -> np.ndarray:
        return np.sqrt(np.mean(ctx.diff_f32**2, axis=(1, 2, 3)))


# ====================================================================== #
#  SSIM — Structural Similarity Index                                    #
# ====================================================================== #


class SSIM(DistortionMetric):
    """Mean SSIM computed per channel and averaged.

    Single-image path uses scikit-image; batched path uses a GPU conv pass
    via ``_batched_ssim_torch``, matching skimage's defaults.
    """

    def __init__(self, win_size: int = 7, use_cuda: bool = False) -> None:
        self.win_size = win_size
        self.use_cuda = use_cuda

    @property
    def name(self) -> str:
        return "SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, original: np.ndarray, noised: np.ndarray) -> float:
        return float(
            structural_similarity(
                original.astype(np.float32),
                noised.astype(np.float32),
                channel_axis=2,
                data_range=255.0,
                win_size=self.win_size,
            )
        )

    def compute_from_context(self, ctx: DistortionContext) -> float:
        return self.compute(ctx.original, ctx.noised)

    def compute_batch_from_context(self, ctx: BatchDistortionContext) -> np.ndarray:
        if self.use_cuda:
            return _batched_ssim_torch(
                ctx.originals.astype(np.float32),
                ctx.noised.astype(np.float32),
                win_size=self.win_size,
            )
        return np.array(
            [
                self.compute(ctx.originals[i], ctx.noised[i])
                for i in range(len(ctx.originals))
            ]
        )
