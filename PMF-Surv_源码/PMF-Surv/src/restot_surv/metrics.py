from __future__ import annotations

from typing import Sequence

import numpy as np


def concordance_index(times: Sequence[float], events: Sequence[float], risks: Sequence[float]) -> float:
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64) > 0.5
    risk = np.asarray(risks, dtype=np.float64)
    observed_indices = np.flatnonzero(event)
    comparable = 0
    concordant = 0.0
    for index in observed_indices:
        mask = time > time[index]
        comparable += int(mask.sum())
        differences = risk[index] - risk[mask]
        concordant += float((differences > 0).sum())
        concordant += 0.5 * float((differences == 0).sum())
    return float(concordant / comparable) if comparable else float("nan")


def integrated_brier_score(
    times: Sequence[float],
    events: Sequence[float],
    survival: np.ndarray,
    boundaries: Sequence[float],
    censoring_times: Sequence[float],
    censoring_survival: Sequence[float],
) -> float:
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64) > 0.5
    curves = np.asarray(survival, dtype=np.float64)
    censor_time = np.asarray(censoring_times, dtype=np.float32)
    censor_survival = np.asarray(censoring_survival, dtype=np.float32)
    scores = []
    for index, boundary in enumerate(np.asarray(boundaries, dtype=np.float64)):
        target = (time > boundary).astype(np.float64)
        before_or_at = (time <= boundary) & event
        after = time > boundary
        weights = np.zeros_like(time)
        if before_or_at.any():
            query = np.asarray(time[before_or_at], dtype=np.float32)
            weights[before_or_at] = 1.0 / _numpy_censoring_survival_at(
                query,
                censor_time,
                censor_survival,
                left_limit=True,
            )
        if after.any():
            query = np.full(int(after.sum()), boundary, dtype=np.float32)
            weights[after] = 1.0 / _numpy_censoring_survival_at(
                query,
                censor_time,
                censor_survival,
            )
        scores.append(
            float(
                np.sum(weights * (curves[:, index + 1] - target) ** 2)
                / max(time.size, 1)
            )
        )
    if not scores:
        return float("nan")
    evaluation_times = np.asarray(boundaries, dtype=np.float64)
    if len(scores) == 1:
        return float(scores[0])
    time_span = float(evaluation_times[-1] - evaluation_times[0])
    if time_span <= 0.0:
        return float(np.mean(scores))
    return float(np.trapz(scores, evaluation_times) / time_span)


def _numpy_censoring_survival_at(
    query: np.ndarray,
    times: np.ndarray,
    survival: np.ndarray,
    left_limit: bool = False,
) -> np.ndarray:
    if times.size == 0:
        return np.ones_like(query, dtype=np.float64)
    side = "left" if left_limit else "right"
    indices = np.searchsorted(times, query, side=side) - 1
    result = np.ones_like(query, dtype=np.float64)
    valid = indices >= 0
    result[valid] = survival[indices[valid]]
    return np.clip(result, 1e-6, None)
