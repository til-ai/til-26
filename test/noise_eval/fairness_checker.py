from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_OP_LABELS = {"le": "≤", "ge": "≥", "lt": "<", "gt": ">", "eq": "=="}

_OPS = {
    "le": lambda v, t: v <= t,
    "ge": lambda v, t: v >= t,
    "lt": lambda v, t: v < t,
    "gt": lambda v, t: v > t,
    "eq": lambda v, t: v == t,
}


@dataclass
class CheckResult:
    metric: str
    value: float
    threshold: float
    op: str
    passed: bool
    missing: bool = False  # True when the metric was not present in the input

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("MISSING" if self.missing else "FAIL")
        op_sym = _OP_LABELS.get(self.op, self.op)
        return (
            f"  [{status:7s}]  {self.metric:<26s}  "
            f"value={self.value:>9.4f}  {op_sym}  threshold={self.threshold:.4f}"
        )


@dataclass
class EvaluationResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self, width: int = 72) -> str:
        sep = "─" * width
        lines = [
            "═" * width,
            f"  Fairness Evaluation  —  {'PASS ✓' if self.passed else 'FAIL ✗'}",
            sep,
            f"  {'Check':<8s}  {'Metric':<26s}  {'Value':>9s}     {'Threshold'}",
            sep,
        ]
        for c in self.checks:
            lines.append(str(c))
        lines += [
            sep,
            f"  Result: {'ALL CHECKS PASSED' if self.passed else 'ONE OR MORE CHECKS FAILED'}",
            "═" * width,
        ]
        return "\n".join(lines)


class FairnessChecker:
    """Loads threshold config and evaluates a metrics dict against it.

    Parameters
    ----------
    config_path : str | Path
        Path to a YAML file in the format produced by eval_thresholds.yaml.
    missing_metric_policy : str
        ``"fail"`` (default) — treat a missing metric as a failed check.
        ``"warn"`` — log a warning but do not fail on missing metrics.
        ``"skip"`` — silently skip checks for missing metrics.
    """

    def __init__(
        self,
        config_path: str | Path,
        missing_metric_policy: str = "fail",
    ) -> None:
        if missing_metric_policy not in ("fail", "warn", "skip"):
            raise ValueError(
                f"missing_metric_policy must be fail/warn/skip, got {missing_metric_policy!r}"
            )
        self.missing_metric_policy = missing_metric_policy

        config_path = Path(config_path)
        with open(config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

        self._checks: list[dict[str, Any]] = self._config.get("checks", [])
        if not self._checks:
            raise ValueError(f"No 'checks' entries found in {config_path}")

    # ------------------------------------------------------------------ #

    def evaluate(self, metrics: dict[str, float]) -> EvaluationResult:
        """Evaluate a dict of computed metric values against all checks.

        Parameters
        ----------
        metrics : dict[str, float]
            Mapping of metric name → mean value.  Must contain the metric
            names defined in the threshold config.

        Returns
        -------
        EvaluationResult
            ``.passed`` is True only if every check passes.
        """
        check_results: list[CheckResult] = []
        all_passed = True

        for spec in self._checks:
            metric = spec["metric"]
            op = spec["op"]
            thr = float(spec["threshold"])

            if metric not in metrics:
                missing = True
                passed = self.missing_metric_policy != "fail"
                value = float("nan")
                if self.missing_metric_policy == "warn":
                    import warnings

                    warnings.warn(
                        f"FairnessChecker: metric '{metric}' not found in input"
                    )
            else:
                missing = False
                value = float(metrics[metric])
                fn = _OPS.get(op)
                if fn is None:
                    raise ValueError(f"Unknown op {op!r} for metric {metric!r}")
                passed = fn(value, thr)

            if not passed:
                all_passed = False

            check_results.append(
                CheckResult(
                    metric=metric,
                    value=value,
                    threshold=thr,
                    op=op,
                    passed=passed,
                    missing=missing,
                )
            )

        return EvaluationResult(
            passed=all_passed,
            checks=check_results,
            metrics=dict(metrics),
        )

    def evaluate_from_pipeline(
        self,
        eval_pipeline,
        cleans: list,
        noised: list,
        gt_boxes: list | None = None,
    ) -> EvaluationResult:
        """Run EvalPipeline then immediately evaluate.

        Parameters
        ----------
        eval_pipeline : EvalPipeline
            An initialised pipeline instance.
        cleans, noised : list[np.ndarray]
            Batches of clean and noised HWC uint8 images.
        gt_boxes : list[np.ndarray] | None
            Per-image GT bounding boxes (xyxy).  Required for inside-box
            metrics (``L2_inside``, ``SSIM_inside``).
        """
        if gt_boxes is not None:
            raw = eval_pipeline.evaluate_batch_with_boxes(cleans, noised, gt_boxes)
        else:
            raw = eval_pipeline.evaluate_batch_arr(cleans, noised)

        import numpy as np

        metrics = {k: float(np.mean(v)) for k, v in raw.items()}
        return self.evaluate(metrics)

    # ------------------------------------------------------------------ #

    @property
    def version(self) -> str:
        return self._config.get("version", "unknown")

    @property
    def n_fair(self) -> int:
        return int(self._config.get("n_fair", -1))

    @property
    def n_unfair(self) -> int:
        return int(self._config.get("n_unfair", -1))

    def __repr__(self) -> str:
        return (
            f"FairnessChecker(v={self.version}, "
            f"{len(self._checks)} checks, "
            f"derived from {self.n_fair} fair / {self.n_unfair} unfair configs)"
        )
