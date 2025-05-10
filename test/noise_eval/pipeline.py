"""Evaluation pipeline that runs multiple metrics over image pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

from .metrics import SSIM, DistortionMetric, L2Distance


def _make_box_mask(H: int, W: int, boxes: np.ndarray) -> np.ndarray:
    """Boolean mask (H, W) from COCO-format boxes [x, y, w, h]."""
    mask = np.zeros((H, W), dtype=bool)
    if boxes.size == 0:
        return mask
    for box in boxes:
        x, y, w, h = box
        x1, y1 = int(x), int(y)
        x2, y2 = min(int(x + w), W), min(int(y + h), H)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


@dataclass
class MetricResult:
    name: str
    value: float
    higher_is_better: bool

    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.value:.4f} ({'↑' if self.higher_is_better else '↓'})"
        )


@dataclass
class ImageReport:
    results: list[MetricResult] = field(default_factory=list)

    def __repr__(self) -> str:
        return "\n".join(str(r) for r in self.results)

    def to_dict(self) -> dict[str, float]:
        return {r.name: r.value for r in self.results}


@dataclass
class BatchSummary:
    per_image: list[ImageReport]
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    min: dict[str, float] = field(default_factory=dict)
    max: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        header = f"BatchSummary over {len(self.per_image)} images\n{'=' * 50}"
        rows = [
            f"  {name:30s}  mean={self.mean[name]:8.4f}  std={self.std[name]:8.4f}  "
            f"min={self.min[name]:8.4f}  max={self.max[name]:8.4f}"
            for name in self.mean
        ]
        return header + "\n" + "\n".join(rows)


class EvalPipeline:
    def __init__(
        self,
        metrics: Optional[Sequence[DistortionMetric]] = None,
        use_cuda: bool = False,
    ) -> None:
        if metrics is not None:
            self.metrics: list[DistortionMetric] = list(metrics)
        else:
            self.metrics = [
                L2Distance(),
                SSIM(use_cuda=use_cuda),
            ]

    def evaluate_batched_with_boxes(
        self,
        originals: Sequence[np.ndarray],
        noised_images: Sequence[Optional[np.ndarray]],
        boxes_list: Sequence[np.ndarray],
    ) -> BatchSummary:
        """Evaluate global L2/SSIM plus inside-bbox L2 and SSIM per image.

        boxes_list must be COCO-format [x, y, w, h] arrays, one (M_i, 4)
        array per image.  Images with no boxes fall back to global metrics
        for the inside values.  None entries in noised_images are treated
        as failed predictions and scored at worst-case values.
        """
        assert (
            len(originals) == len(noised_images) == len(boxes_list)
        ), f"Length mismatch: {len(originals)}, {len(noised_images)}, {len(boxes_list)}"

        ssim_metric = next((m for m in self.metrics if isinstance(m, SSIM)), SSIM())
        win_size = ssim_metric.win_size

        per_image: list[ImageReport] = []

        for orig, noised, boxes in tqdm(
            zip(originals, noised_images, boxes_list), total=len(originals)
        ):
            if noised is None:
                per_image.append(
                    ImageReport(
                        results=[
                            MetricResult("L2 (RMSE)", 255.0, False),
                            MetricResult("L2 inside", 255.0, False),
                            MetricResult("SSIM inside", -1.0, True),
                        ]
                    )
                )
                continue

            orig_f = orig.astype(np.float32)
            noised_f = noised.astype(np.float32)
            H, W = orig.shape[:2]

            diff = orig_f - noised_f
            l2_global = float(np.sqrt(np.mean(diff**2)))

            # Single structural_similarity call gives both the scalar mean
            # and the per-pixel map needed for inside-bbox SSIM.
            ssim_global, ssim_map = structural_similarity(
                orig_f,
                noised_f,
                channel_axis=2,
                data_range=255.0,
                win_size=win_size,
                full=True,
            )
            ssim_global = float(ssim_global)
            if ssim_map.ndim == 3:
                ssim_map = ssim_map.mean(axis=-1)  # (H, W, C) → (H, W)

            boxes_arr = np.asarray(boxes)
            mask = _make_box_mask(H, W, boxes_arr)

            if mask.any():
                sq_diff_px = (diff**2).mean(axis=-1)  # (H, W)
                l2_inside = float(np.sqrt(sq_diff_px[mask].mean()))
                ssim_inside = float(ssim_map[mask].mean())
            else:
                l2_inside = l2_global
                ssim_inside = ssim_global

            per_image.append(
                ImageReport(
                    results=[
                        MetricResult("L2 (RMSE)", l2_global, False),
                        MetricResult("L2 inside", l2_inside, False),
                        MetricResult("SSIM inside", ssim_inside, True),
                    ]
                )
            )

        summary = BatchSummary(per_image=per_image)
        if per_image:
            metric_names = ["L2 (RMSE)", "L2 inside", "SSIM inside"]
            for name in metric_names:
                try:
                    vals = np.array(
                        [
                            next(r.value for r in report.results if r.name == name)
                            for report in per_image
                        ]
                    )
                    summary.mean[name] = float(vals.mean())
                    summary.std[name] = float(vals.std())
                    summary.min[name] = float(vals.min())
                    summary.max[name] = float(vals.max())
                except StopIteration:
                    print(
                        f"Warning: metric '{name}' not found in per-image reports, skipping summary stats."
                    )

        return summary

    def __repr__(self) -> str:
        return f"EvalPipeline(metrics={[m.name for m in self.metrics]})"
