from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather


def fit_time_boundaries(times: Sequence[float], events: Sequence[float], num_bins: int) -> np.ndarray:
    times_array = np.asarray(times, dtype=np.float64)
    events_array = np.asarray(events, dtype=np.float64) > 0.5
    source = times_array[events_array]
    if source.size < num_bins - 1:
        source = times_array[np.isfinite(times_array)]
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    boundaries = np.quantile(source, quantiles).astype(np.float32)
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1e-4)
    return boundaries


def fit_censoring_distribution(
    times: Sequence[float],
    events: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64) > 0.5
    finite = np.isfinite(time)
    time = time[finite]
    event = event[finite]
    if time.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    unique_times = np.unique(time)
    survival = 1.0
    values = []
    for current in unique_times:
        at_risk = int(np.sum(time >= current))
        censored = int(np.sum((time == current) & ~event))
        if at_risk > 0:
            survival *= 1.0 - censored / at_risk
        values.append(survival)
    return unique_times.astype(np.float32), np.asarray(values, dtype=np.float32)


def censoring_survival_at(
    query: torch.Tensor,
    censoring_times: torch.Tensor,
    censoring_survival: torch.Tensor,
    left_limit: bool = False,
) -> torch.Tensor:
    if censoring_times.numel() == 0:
        return torch.ones_like(query, dtype=torch.float32)
    indices = torch.searchsorted(
        censoring_times.float(),
        query.float().contiguous(),
        right=not left_limit,
    ) - 1
    result = torch.ones_like(query, dtype=torch.float32)
    valid = indices >= 0
    if valid.any():
        result[valid] = censoring_survival.float()[indices[valid]]
    return result.clamp_min(1e-6)


def ipcw_event_weights(
    times: torch.Tensor,
    events: torch.Tensor,
    censoring_times: torch.Tensor,
    censoring_survival: torch.Tensor,
    survival_floor: float,
    weight_cap: float | torch.Tensor = float("inf"),
) -> torch.Tensor:
    """Compute stabilized Uno-style event weights at exact event months."""

    event = events.float().reshape(-1) > 0.5
    survival = censoring_survival_at(
        times.float().reshape(-1),
        censoring_times,
        censoring_survival,
        left_limit=True,
    ).clamp_min(float(survival_floor))
    weights = event.float() / survival.square()
    cap = torch.as_tensor(weight_cap, dtype=weights.dtype, device=weights.device)
    if torch.isfinite(cap):
        weights = weights.clamp_max(cap)
    return weights


def fit_ipcw_weight_cap(
    times: Sequence[float],
    events: Sequence[float],
    censoring_times: torch.Tensor,
    censoring_survival: torch.Tensor,
    survival_floor: float,
    quantile: float,
) -> float:
    """Fit one deterministic IPCW cap from the training event distribution."""

    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("IPCW weight quantile must be in (0, 1].")
    time = torch.as_tensor(times, dtype=torch.float32)
    event = torch.as_tensor(events, dtype=torch.float32)
    weights = ipcw_event_weights(
        time,
        event,
        censoring_times.cpu(),
        censoring_survival.cpu(),
        survival_floor=survival_floor,
    )
    observed_weights = weights[event > 0.5]
    if observed_weights.numel() == 0:
        return 1.0
    return float(torch.quantile(observed_weights, float(quantile)).item())


def discretize_times(times: torch.Tensor, boundaries: torch.Tensor, num_bins: int) -> torch.Tensor:
    return torch.bucketize(times.float(), boundaries.float()).clamp(0, num_bins - 1)


def interval_geometry(
    boundaries: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real-month interval centers, widths, and the modeled horizon."""

    if int(num_bins) <= 0:
        raise ValueError("num_bins must be positive.")
    if boundaries.numel() != int(num_bins) - 1:
        raise ValueError(
            f"Expected {int(num_bins) - 1} boundaries, got {boundaries.numel()}."
        )
    starts = torch.cat([boundaries.new_zeros(1), boundaries.float()])
    if starts.numel() == 1:
        last_width = starts.new_ones(1)
    else:
        last_width = (starts[-1:] - starts[-2:-1]).clamp_min(1.0)
    ends = torch.cat([starts[1:], starts[-1:] + last_width])
    widths = (ends - starts).clamp_min(1.0e-4)
    centers = starts + 0.5 * widths
    return centers, widths, ends[-1]


class MTLRHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_bins: int, dropout: float) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins),
        )
        self.register_buffer("boundaries", torch.zeros(self.num_bins - 1))
        self.register_buffer("boundaries_set", torch.tensor(False))
        self.register_buffer("censoring_times", torch.empty(0))
        self.register_buffer("censoring_survival", torch.empty(0))
        self.register_buffer("rank_ipcw_cap", torch.tensor(float("inf")))

    def set_boundaries(self, boundaries: torch.Tensor) -> None:
        values = torch.as_tensor(boundaries, dtype=torch.float32, device=self.boundaries.device)
        if values.numel() != self.num_bins - 1:
            raise ValueError(f"Expected {self.num_bins - 1} boundaries, got {values.numel()}.")
        self.boundaries.copy_(values)
        self.boundaries_set.fill_(True)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)

    def set_censoring_distribution(
        self,
        times: torch.Tensor,
        survival: torch.Tensor,
    ) -> None:
        self.censoring_times = torch.as_tensor(
            times,
            dtype=torch.float32,
            device=self.boundaries.device,
        )
        self.censoring_survival = torch.as_tensor(
            survival,
            dtype=torch.float32,
            device=self.boundaries.device,
        )

    def set_rank_ipcw_cap(self, cap: float | torch.Tensor) -> None:
        value = torch.as_tensor(
            cap,
            dtype=torch.float32,
            device=self.boundaries.device,
        )
        if value.numel() != 1 or float(value.item()) <= 0.0:
            raise ValueError("The IPCW rank cap must be one positive value.")
        self.rank_ipcw_cap.copy_(value.reshape(()))

    @staticmethod
    def _scores(logits: torch.Tensor) -> torch.Tensor:
        return torch.flip(torch.cumsum(torch.flip(logits.float(), dims=(1,)), dim=1), dims=(1,))

    def loss(self, logits: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        if not bool(self.boundaries_set.item()):
            raise RuntimeError("Time boundaries must be set before training.")
        bins = discretize_times(times, self.boundaries, self.num_bins)
        log_probability = self._scores(logits) - torch.logsumexp(self._scores(logits), dim=1, keepdim=True)
        observed = events > 0.5
        losses = logits.new_zeros(logits.shape[0], dtype=torch.float32)
        if observed.any():
            losses[observed] = -log_probability[observed].gather(1, bins[observed, None]).squeeze(1)
        if (~observed).any():
            censor_bins = torch.bucketize(
                times[~observed].float(),
                self.boundaries.float(),
                right=True,
            ).clamp(0, self.num_bins - 1)
            columns = torch.arange(self.num_bins, device=logits.device)[None, :]
            valid = columns >= censor_bins[:, None]
            censor_log_probability = log_probability[~observed].masked_fill(~valid, float("-inf"))
            losses[~observed] = -torch.logsumexp(censor_log_probability, dim=1)
        return losses.mean()

    def survival(self, logits: torch.Tensor) -> torch.Tensor:
        interval_probability = F.softmax(self._scores(logits), dim=1)
        return torch.flip(
            torch.cumsum(torch.flip(interval_probability, dims=(1,)), dim=1),
            dims=(1,),
        ).clamp(0.0, 1.0)

    def risk(self, logits: torch.Tensor) -> torch.Tensor:
        survival = self.survival(logits)
        _, widths, _ = interval_geometry(self.boundaries, self.num_bins)
        return -(survival * widths[None, :]).sum(dim=1)

    def brier_loss(self, logits: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        survival = self.survival(logits)
        if self.boundaries.numel() == 0:
            return logits.new_zeros(())
        scores = []
        observed = events > 0.5
        for index, boundary in enumerate(self.boundaries):
            target = (times > boundary).float()
            before_or_at = (times <= boundary) & observed
            after = times > boundary
            weights = torch.zeros_like(times, dtype=torch.float32)
            if before_or_at.any():
                weights[before_or_at] = 1.0 / censoring_survival_at(
                    times[before_or_at],
                    self.censoring_times,
                    self.censoring_survival,
                    left_limit=True,
                )
            if after.any():
                weights[after] = 1.0 / censoring_survival_at(
                    boundary.expand_as(times[after]),
                    self.censoring_times,
                    self.censoring_survival,
                )
            error = (survival[:, index + 1] - target).square()
            scores.append((weights * error).sum() / max(times.numel(), 1))
        return torch.stack(scores).mean()


class PiecewiseExponentialHead(MTLRHead):
    """Utilities for non-negative piecewise-constant interval hazard rates."""

    def __init__(self, num_bins: int) -> None:
        # Reuse the boundary/censoring buffers and Brier implementation while
        # replacing all MTLR probability semantics below.
        super().__init__(1, 1, num_bins, 0.0)
        self.net = nn.Identity()

    def loss(
        self,
        rates: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """Exact event/right-censoring NLL using per-interval exposure time."""

        if not bool(self.boundaries_set.item()):
            raise RuntimeError("Time boundaries must be set before training.")
        if rates.ndim != 2 or rates.shape[1] != self.num_bins:
            raise ValueError(f"Expected hazard rates with shape [B,{self.num_bins}].")
        rates_work = rates.float().clamp_min(1.0e-8)
        starts = torch.cat([self.boundaries.new_zeros(1), self.boundaries.float()])
        _, widths, _ = interval_geometry(self.boundaries, self.num_bins)
        elapsed = (times.float().reshape(-1, 1) - starts[None]).clamp_min(0.0)
        if self.num_bins == 1:
            exposure = elapsed
        else:
            closed_exposure = torch.minimum(
                elapsed[:, :-1],
                widths[None, :-1],
            )
            # The final rate continues beyond the nominal plotting horizon so
            # late events/censoring retain their exact observed exposure.
            exposure = torch.cat([closed_exposure, elapsed[:, -1:]], dim=1)
        cumulative_hazard = (rates_work * exposure).sum(dim=1)
        event_bins = discretize_times(times, self.boundaries, self.num_bins)
        event_rate = rates_work.gather(1, event_bins[:, None]).squeeze(1)
        negative_log_likelihood = cumulative_hazard - events.float() * torch.log(
            event_rate
        )
        return negative_log_likelihood.mean()

    def survival(self, rates: torch.Tensor) -> torch.Tensor:
        """Return S(0) followed by survival at all 12 interval endpoints."""

        _, widths, _ = interval_geometry(self.boundaries, self.num_bins)
        interval_hazard = rates.float().clamp_min(0.0) * widths[None]
        endpoint_survival = torch.exp(-torch.cumsum(interval_hazard, dim=1))
        return torch.cat(
            [torch.ones_like(endpoint_survival[:, :1]), endpoint_survival],
            dim=1,
        )

    def risk(self, rates: torch.Tensor) -> torch.Tensor:
        """Use negative restricted mean survival; larger values mean higher risk."""

        rates_work = rates.float().clamp_min(1.0e-8)
        _, widths, _ = interval_geometry(self.boundaries, self.num_bins)
        start_survival = self.survival(rates_work)[:, :-1]
        interval_area = start_survival * (
            -torch.expm1(-rates_work * widths[None]) / rates_work
        )
        return -interval_area.sum(dim=1)


def cox_partial_likelihood(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    risk = risk.float()
    times = times.float()
    observed = events > 0.5
    event_times = torch.unique(times[observed])
    if event_times.numel() == 0:
        return risk.sum() * 0.0
    log_likelihood = risk.new_zeros(())
    event_count = observed.sum().float()
    for event_time in event_times:
        tied_events = observed & (times == event_time)
        risk_set = times >= event_time
        tied_count = tied_events.sum().float()
        log_likelihood = log_likelihood + risk[tied_events].sum()
        log_likelihood = log_likelihood - tied_count * torch.logsumexp(
            risk[risk_set],
            dim=0,
        )
    return -log_likelihood / event_count.clamp_min(1.0)


def distributed_cox_partial_likelihood(
    risk: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return cox_partial_likelihood(risk, times, events)
    world_size = dist.get_world_size()
    if world_size <= 1:
        return cox_partial_likelihood(risk, times, events)
    batch_sizes = [
        torch.zeros((), dtype=torch.long, device=risk.device)
        for _ in range(world_size)
    ]
    local_size = torch.tensor(
        risk.shape[0],
        dtype=torch.long,
        device=risk.device,
    )
    dist.all_gather(batch_sizes, local_size)
    if len({int(value.item()) for value in batch_sizes}) != 1:
        raise RuntimeError(
            "Distributed Cox requires the same local batch size on every rank."
        )
    global_risk = torch.cat(all_gather(risk.float()), dim=0)
    gathered_times = [torch.empty_like(times.float()) for _ in range(world_size)]
    gathered_events = [torch.empty_like(events.float()) for _ in range(world_size)]
    dist.all_gather(gathered_times, times.float().contiguous())
    dist.all_gather(gathered_events, events.float().contiguous())
    return cox_partial_likelihood(
        global_risk,
        torch.cat(gathered_times, dim=0),
        torch.cat(gathered_events, dim=0),
    )


def endpoint_consistency(
    os_survival: torch.Tensor,
    pfs_survival: torch.Tensor,
    os_boundaries: torch.Tensor,
    pfs_boundaries: torch.Tensor,
) -> torch.Tensor:
    common = torch.unique(torch.cat([os_boundaries, pfs_boundaries])).sort().values

    def sample(curve: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(
            common,
            boundaries,
            right=True,
        ).clamp(0, curve.shape[1] - 1)
        return curve[:, indices]

    return (sample(pfs_survival, pfs_boundaries) - sample(os_survival, os_boundaries)).clamp_min(0).mean()
