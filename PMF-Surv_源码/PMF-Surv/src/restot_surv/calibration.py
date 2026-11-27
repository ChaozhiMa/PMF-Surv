from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


EPSILON = 1.0e-12


def _as_float_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_survival_data(
    times: Sequence[float],
    events: Sequence[float],
    risks: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = _as_float_vector(times, "times")
    event_values = _as_float_vector(events, "events")
    risk = _as_float_vector(risks, "risks")
    if not (time.shape == event_values.shape == risk.shape):
        raise ValueError("times, events, and risks must have equal lengths.")
    if (time < 0.0).any():
        raise ValueError("times must be non-negative.")
    event = event_values > 0.5
    if not event.any():
        raise ValueError("At least one observed event is required.")
    if float(np.std(risk)) <= EPSILON:
        raise ValueError("risks must not be constant.")
    return time, event, risk


def _efron_negative_log_likelihood(
    beta: float,
    times: np.ndarray,
    events: np.ndarray,
    standardized_risks: np.ndarray,
) -> float:
    linear_predictor = np.clip(
        float(beta) * standardized_risks,
        -50.0,
        50.0,
    )
    exp_predictor = np.exp(linear_predictor)
    log_likelihood = 0.0
    for event_time in np.unique(times[events]):
        deaths = events & (times == event_time)
        death_count = int(deaths.sum())
        risk_set = times >= event_time
        risk_total = float(exp_predictor[risk_set].sum())
        death_total = float(exp_predictor[deaths].sum())
        log_likelihood += float(linear_predictor[deaths].sum())
        for tied_index in range(death_count):
            denominator = risk_total - tied_index * death_total / death_count
            if denominator <= 0.0 or not np.isfinite(denominator):
                return float("inf")
            log_likelihood -= float(np.log(denominator))
    return -log_likelihood


def _breslow_baseline(
    times: np.ndarray,
    events: np.ndarray,
    standardized_risks: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    event_times = np.unique(times[events])
    exp_predictor = np.exp(
        np.clip(float(beta) * standardized_risks, -50.0, 50.0)
    )
    increments: List[float] = []
    for event_time in event_times:
        death_count = int(np.sum(events & (times == event_time)))
        denominator = float(exp_predictor[times >= event_time].sum())
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError("Invalid Breslow risk-set denominator.")
        increments.append(death_count / denominator)
    cumulative_hazard = np.cumsum(np.asarray(increments, dtype=np.float64))
    return event_times.astype(np.float64), cumulative_hazard


@dataclass(frozen=True)
class RankPreservingCoxCalibrator:
    risk_mean: float
    risk_scale: float
    risk_coefficient: float
    baseline_times: List[float]
    baseline_cumulative_hazard: List[float]
    fit_patients: int
    fit_events: int
    optimizer_at_lower_bound: bool
    optimizer_at_upper_bound: bool

    @classmethod
    def fit(
        cls,
        times: Sequence[float],
        events: Sequence[float],
        risks: Sequence[float],
        coefficient_bounds: Tuple[float, float] = (1.0e-4, 20.0),
    ) -> "RankPreservingCoxCalibrator":
        time, event, risk = _validate_survival_data(times, events, risks)
        risk_mean = float(np.mean(risk))
        risk_scale = float(np.std(risk, ddof=0))
        standardized = (risk - risk_mean) / risk_scale
        lower, upper = map(float, coefficient_bounds)
        if not 0.0 < lower < upper:
            raise ValueError("coefficient_bounds must be positive and increasing.")

        result = minimize_scalar(
            _efron_negative_log_likelihood,
            bounds=(lower, upper),
            method="bounded",
            args=(time, event, standardized),
            options={"xatol": 1.0e-10, "maxiter": 500},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Positive Cox optimization failed: {result.message}")
        coefficient = float(result.x)
        baseline_times, baseline_hazard = _breslow_baseline(
            time,
            event,
            standardized,
            coefficient,
        )
        bound_tolerance = max(1.0e-6, 1.0e-4 * (upper - lower))
        return cls(
            risk_mean=risk_mean,
            risk_scale=risk_scale,
            risk_coefficient=coefficient,
            baseline_times=baseline_times.tolist(),
            baseline_cumulative_hazard=baseline_hazard.tolist(),
            fit_patients=int(time.size),
            fit_events=int(event.sum()),
            optimizer_at_lower_bound=bool(coefficient - lower <= bound_tolerance),
            optimizer_at_upper_bound=bool(upper - coefficient <= bound_tolerance),
        )

    def standardized_risk(self, risks: Sequence[float]) -> np.ndarray:
        risk = _as_float_vector(risks, "risks")
        return (risk - float(self.risk_mean)) / float(self.risk_scale)

    def cumulative_baseline_hazard(
        self,
        evaluation_times: Sequence[float],
    ) -> np.ndarray:
        query = _as_float_vector(evaluation_times, "evaluation_times")
        baseline_times = np.asarray(self.baseline_times, dtype=np.float64)
        baseline_hazard = np.asarray(
            self.baseline_cumulative_hazard,
            dtype=np.float64,
        )
        indices = np.searchsorted(baseline_times, query, side="right") - 1
        result = np.zeros_like(query)
        valid = indices >= 0
        result[valid] = baseline_hazard[indices[valid]]
        return result

    def survival(
        self,
        risks: Sequence[float],
        evaluation_times: Sequence[float],
    ) -> np.ndarray:
        standardized = self.standardized_risk(risks)
        baseline_hazard = self.cumulative_baseline_hazard(evaluation_times)
        relative_risk = np.exp(
            np.clip(
                float(self.risk_coefficient) * standardized,
                -50.0,
                50.0,
            )
        )
        return np.exp(-relative_risk[:, None] * baseline_hazard[None, :])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RankPreservingCoxCalibrator":
        return cls(**dict(values))


def _censoring_survival_at(
    query: np.ndarray,
    censoring_times: np.ndarray,
    censoring_survival: np.ndarray,
    left_limit: bool = False,
) -> np.ndarray:
    if censoring_times.size == 0:
        return np.ones_like(query, dtype=np.float64)
    side = "left" if left_limit else "right"
    indices = np.searchsorted(censoring_times, query, side=side) - 1
    result = np.ones_like(query, dtype=np.float64)
    valid = indices >= 0
    result[valid] = censoring_survival[indices[valid]]
    return np.clip(result, 1.0e-6, None)


def brier_scores(
    times: Sequence[float],
    events: Sequence[float],
    survival_at_boundaries: np.ndarray,
    evaluation_times: Sequence[float],
    censoring_times: Sequence[float],
    censoring_survival: Sequence[float],
) -> np.ndarray:
    time = _as_float_vector(times, "times")
    event = _as_float_vector(events, "events") > 0.5
    predictions = np.asarray(survival_at_boundaries, dtype=np.float64)
    query_times = _as_float_vector(evaluation_times, "evaluation_times")
    censor_time = np.asarray(censoring_times, dtype=np.float64)
    censor_surv = np.asarray(censoring_survival, dtype=np.float64)
    if predictions.shape != (time.size, query_times.size):
        raise ValueError("survival_at_boundaries has an invalid shape.")

    scores = []
    for column, boundary in enumerate(query_times):
        target = (time > boundary).astype(np.float64)
        before_or_at = (time <= boundary) & event
        after = time > boundary
        weights = np.zeros_like(time)
        if before_or_at.any():
            weights[before_or_at] = 1.0 / _censoring_survival_at(
                time[before_or_at],
                censor_time,
                censor_surv,
                left_limit=True,
            )
        if after.any():
            weights[after] = 1.0 / _censoring_survival_at(
                np.full(int(after.sum()), boundary, dtype=np.float64),
                censor_time,
                censor_surv,
            )
        scores.append(
            float(
                np.sum(weights * (predictions[:, column] - target) ** 2)
                / max(time.size, 1)
            )
        )
    return np.asarray(scores, dtype=np.float64)


def integrated_brier_score(
    scores: Sequence[float],
    evaluation_times: Sequence[float],
) -> float:
    values = _as_float_vector(scores, "scores")
    times = _as_float_vector(evaluation_times, "evaluation_times")
    if values.shape != times.shape:
        raise ValueError("scores and evaluation_times must have equal lengths.")
    if values.size == 1:
        return float(values[0])
    time_span = float(times[-1] - times[0])
    if time_span <= 0.0:
        return float(np.mean(values))
    return float(np.trapz(values, times) / time_span)


def fit_censoring_distribution(
    times: Sequence[float],
    events: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the reverse Kaplan-Meier distribution used for IPCW metrics."""

    time = _as_float_vector(times, "times")
    event_values = _as_float_vector(events, "events")
    if time.shape != event_values.shape:
        raise ValueError("times and events must have equal lengths.")
    event = event_values > 0.5
    unique_times = np.unique(time)
    survival = 1.0
    values: List[float] = []
    for current_time in unique_times:
        at_risk = int(np.sum(time >= current_time))
        censored = int(np.sum((time == current_time) & ~event))
        if at_risk > 0:
            survival *= 1.0 - censored / at_risk
        values.append(survival)
    return unique_times.astype(np.float64), np.asarray(values, dtype=np.float64)


IBS_GRID = {"os": np.linspace(6.0, 24.0, 37), "pfs": np.linspace(3.0, 12.0, 37)}


def load_center_patient_ids(
    center_specs: Mapping[str, Any], id_col: str, project_root: Path,
) -> Dict[str, set[str]]:
    center_ids = {}
    for center, spec in center_specs.items():
        path = Path(spec["csv"] if isinstance(spec, Mapping) else spec).expanduser()
        if not path.is_absolute():
            path = project_root / path
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or id_col not in rows.fieldnames:
                raise ValueError(f"Missing {id_col!r} in the {center} center list.")
            center_ids[str(center)] = {str(row[id_col]).strip() for row in rows}
    return center_ids


def _ibs(
    times: np.ndarray,
    events: np.ndarray,
    survival: np.ndarray,
    grid: np.ndarray,
    reference: Tuple[np.ndarray, np.ndarray],
) -> float:
    scores = brier_scores(times, events, survival, grid, *reference)
    return integrated_brier_score(scores, grid)


def cox_probability_metrics(
    validation: Mapping[str, np.ndarray],
    external: Mapping[str, np.ndarray],
    patient_ids: Sequence[str] = (),
    center_ids: Mapping[str, Sequence[str]] | None = None,
    *,
    bootstrap: int = 0,
    seed: int = 2026,
) -> Dict[str, float]:
    """Fit the probability mapping on validation once and assess frozen external predictions."""
    if int(bootstrap) < 0:
        raise ValueError("bootstrap must be non-negative.")
    n = len(external["os_risk"])
    masks = {"": np.ones(n, dtype=bool)}
    if center_ids:
        labels = np.asarray([str(value).strip() for value in patient_ids])
        if labels.shape != (n,):
            raise ValueError("External identifiers and predictions must align.")
        for center, ids in center_ids.items():
            mask = np.isin(labels, list(ids))
            if not mask.any():
                raise ValueError(f"No external predictions match center {center!r}.")
            masks[str(center)] = mask
    result = {}
    for endpoint, grid in IBS_GRID.items():
        val_time = validation[f"{endpoint}_time"]
        val_event = validation[f"{endpoint}_event"]
        val_risk = validation[f"{endpoint}_risk"]
        calibrator = RankPreservingCoxCalibrator.fit(val_time, val_event, val_risk)
        primary = fit_censoring_distribution(val_time, val_event)
        external_time = np.asarray(external[f"{endpoint}_time"])
        external_event = np.asarray(external[f"{endpoint}_event"])
        survival = calibrator.survival(external[f"{endpoint}_risk"], grid)
        result[f"{endpoint}_cox_beta"] = calibrator.risk_coefficient
        for center, mask in masks.items():
            prefix = f"{center}_{endpoint}" if center else endpoint
            times, events, curves = external_time[mask], external_event[mask], survival[mask]
            center_reference = fit_censoring_distribution(times, events)
            result[f"{prefix}_ibs_primary"] = _ibs(times, events, curves, grid, primary)
            result[f"{prefix}_ibs_center"] = _ibs(times, events, curves, grid, center_reference)
            if bootstrap and (center or not center_ids):
                # A stable order and the same seed reuse patient draws across endpoints/models.
                rng = np.random.default_rng(seed)
                samples = {"primary": [], "center": []}
                for _ in range(int(bootstrap)):
                    selected = rng.integers(len(times), size=len(times))
                    t, e, s = times[selected], events[selected], curves[selected]
                    samples["primary"].append(_ibs(t, e, s, grid, primary))
                    samples["center"].append(
                        _ibs(t, e, s, grid, fit_censoring_distribution(t, e))
                    )
                for key, values in samples.items():
                    lo, hi = np.quantile(values, [0.025, 0.975])
                    result[f"{prefix}_ibs_{key}_ci_low"] = float(lo)
                    result[f"{prefix}_ibs_{key}_ci_high"] = float(hi)
        result[f"{endpoint}_ibs"] = result[f"{endpoint}_ibs_primary"]
    return result
