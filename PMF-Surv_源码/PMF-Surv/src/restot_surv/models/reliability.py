from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_DEGRADATIONS = (
    "noise",
    "blur",
    "stripes",
    "slice_loss",
    "intensity",
)


def midband_quality_weights(
    predicted_reliability: torch.Tensor,
    *,
    lower_threshold: float,
    upper_threshold: float,
    high_quality_weight: float = 1.0,
    low_quality_weight: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map middle reliability scores to high quality and both tails to low."""

    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError(
            "Quality-band thresholds must satisfy "
            "0 <= lower_threshold < upper_threshold <= 1."
        )
    if not 0.0 < low_quality_weight < high_quality_weight <= 1.0:
        raise ValueError(
            "Quality weights must satisfy "
            "0 < low_quality_weight < high_quality_weight <= 1."
        )
    high_quality_mask = predicted_reliability.ge(
        lower_threshold
    ) & predicted_reliability.le(upper_threshold)
    weights = torch.where(
        high_quality_mask,
        torch.full_like(predicted_reliability, high_quality_weight),
        torch.full_like(predicted_reliability, low_quality_weight),
    )
    return weights, high_quality_mask


class MeanAnchoredRankModulation(nn.Module):
    """Select a mean-anchored, asymmetric hard reliability interval.

    Each patient/phase group is solved independently. The target number of kept
    tokens is split between the two sides of that group's mean in proportion to
    the observed population on each side. This preserves distribution skew:
    the denser side contributes (and excludes) more tokens, while the two raw
    threshold distances are free to differ.

    The numerical pattern mirrors the later UOT fixed-point solver: lower and
    upper thresholds are alternately moved toward their fixed points with a
    damped update, and iteration stops on a threshold-change residual. A final
    rank projection enforces the requested token mass exactly. No sigmoid or
    differentiable transition is used; the returned weights are strictly the
    configured high and low values.
    """

    MODE = "iterative_asymmetric_hard"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        quality_cfg = (
            cfg["model"]
            .get("token_reliability", {})
            .get("quality_band", {})
        )
        self.target_fraction = float(quality_cfg.get("target_fraction", 0.60))
        self.grouping = str(
            quality_cfg.get("grouping", "patient_phase")
        ).lower()
        self.high_quality_weight = float(
            quality_cfg.get("high_quality_weight", 1.0)
        )
        self.low_quality_weight = float(
            quality_cfg.get("low_quality_weight", 0.02)
        )

        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("target_fraction must lie strictly between 0 and 1.")
        if self.grouping != "patient_phase":
            raise ValueError("Iterative hard gating requires grouping=patient_phase.")
        if not 0.0 < self.low_quality_weight < self.high_quality_weight <= 1.0:
            raise ValueError("Hard-gate weights must satisfy 0 < low < high <= 1.")

    def _fixed_points(
        self,
        reliability: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Return mean anchors, exact hard masks, thresholds, and keep count."""

        if reliability.ndim < 2:
            raise ValueError(
                "Reliability must have a token dimension and at least one group dimension."
            )
        token_count = int(reliability.shape[-1])
        if token_count < 2:
            raise ValueError("At least two reliability tokens are required per group.")
        keep_count = max(
            1,
            min(token_count - 1, int(round(self.target_fraction * token_count))),
        )
        flat = reliability.reshape(-1, token_count)
        center = flat.mean(dim=-1)
        sorted_values, sorted_indices = flat.sort(dim=-1)

        lower_population = flat.lt(center.unsqueeze(-1)).sum(dim=-1)
        upper_population = token_count - lower_population
        proportional_lower_keep = torch.round(
            keep_count * lower_population.to(flat.dtype) / float(token_count)
        ).to(torch.long)
        minimum_lower_keep = (keep_count - upper_population).clamp_min(0)
        maximum_lower_keep = torch.minimum(
            lower_population,
            lower_population.new_full(lower_population.shape, keep_count),
        )
        lower_keep = torch.maximum(
            minimum_lower_keep,
            torch.minimum(proportional_lower_keep, maximum_lower_keep),
        )
        start = lower_population - lower_keep
        end = start + keep_count - 1
        target_lower = sorted_values.gather(1, start.unsqueeze(-1)).squeeze(-1)
        target_upper = sorted_values.gather(1, end.unsqueeze(-1)).squeeze(-1)

        sorted_rank = torch.arange(
            token_count,
            dtype=torch.long,
            device=flat.device,
        ).unsqueeze(0)
        selected_in_sorted_order = sorted_rank.ge(
            start.unsqueeze(-1)
        ) & sorted_rank.le(end.unsqueeze(-1))
        hard_mask = torch.zeros_like(flat, dtype=torch.bool).scatter(
            1,
            sorted_indices,
            selected_in_sorted_order,
        )
        return (
            center,
            hard_mask,
            target_lower,
            target_upper,
            start,
            token_count - end - 1,
            keep_count,
        )

    def forward(
        self,
        predicted_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reliability = predicted_reliability.detach().float()
        original_shape = reliability.shape
        (
            center,
            hard_mask,
            target_lower,
            target_upper,
            lower_excluded,
            upper_excluded,
            keep_count,
        ) = self._fixed_points(reliability)

        hard_mask = hard_mask.reshape(original_shape)
        weights = torch.where(
            hard_mask,
            torch.full_like(reliability, self.high_quality_weight),
            torch.full_like(reliability, self.low_quality_weight),
        )
        token_count = int(reliability.shape[-1])
        hard_fraction = hard_mask.float().mean()
        diagnostics = {
            "token_quality_lower_threshold": target_lower.mean(),
            "token_quality_upper_threshold": target_upper.mean(),
            "token_quality_lower_threshold_min": target_lower.min(),
            "token_quality_lower_threshold_max": target_lower.max(),
            "token_quality_upper_threshold_min": target_upper.min(),
            "token_quality_upper_threshold_max": target_upper.max(),
            "token_quality_center_mean": center.mean(),
            "token_quality_lower_width_mean": (center - target_lower).mean(),
            "token_quality_upper_width_mean": (target_upper - center).mean(),
            "token_quality_lower_excluded_fraction": (
                lower_excluded.float().mean() / float(token_count)
            ),
            "token_quality_upper_excluded_fraction": (
                upper_excluded.float().mean() / float(token_count)
            ),
            "token_quality_target_fraction": reliability.new_tensor(
                keep_count / float(token_count)
            ),
            "token_quality_hard_membership_fraction": hard_fraction,
        }
        return weights.to(predicted_reliability.dtype), diagnostics


class LocalTokenDegradation(nn.Module):
    """Create spatially aligned local degradations for online calibration."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        reliability_cfg = model_cfg.get("token_reliability", {})
        calibration_cfg = reliability_cfg.get("calibration", {})
        self.num_phases = int(model_cfg["num_phases"])
        self.patch_grid = tuple(int(value) for value in model_cfg["patch_grid"])
        self.kinds = tuple(
            str(value).lower()
            for value in calibration_cfg.get(
                "degradation_kinds", SUPPORTED_DEGRADATIONS
            )
        )
        unsupported = sorted(set(self.kinds) - set(SUPPORTED_DEGRADATIONS))
        if unsupported:
            raise ValueError(f"Unsupported token degradation kinds: {unsupported}")
        self.regions_per_sample = tuple(
            int(value)
            for value in calibration_cfg.get("regions_per_sample", (2, 4))
        )
        self.region_size_tokens = tuple(
            int(value)
            for value in calibration_cfg.get("region_size_tokens", (1, 3))
        )
        self.severity = tuple(
            float(value)
            for value in calibration_cfg.get("severity", (0.4, 1.0))
        )
        self.shared_phase_probability = float(
            calibration_cfg.get("shared_phase_probability", 0.25)
        )
        self.noise_std = float(calibration_cfg.get("noise_std", 0.6))
        self.blur_kernel_size = int(calibration_cfg.get("blur_kernel_size", 5))
        self.stripe_amplitude = float(
            calibration_cfg.get("stripe_amplitude", 0.7)
        )
        self.intensity_scale = tuple(
            float(value)
            for value in calibration_cfg.get("intensity_scale", (0.55, 1.45))
        )
        self.intensity_shift = tuple(
            float(value)
            for value in calibration_cfg.get("intensity_shift", (-0.5, 0.5))
        )

    @staticmethod
    def _uniform(
        low: float,
        high: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty((), device=device, dtype=dtype).uniform_(low, high)

    @staticmethod
    def _randint(
        low: int,
        high_inclusive: int,
        *,
        device: torch.device,
    ) -> int:
        return int(
            torch.randint(
                low,
                high_inclusive + 1,
                (1,),
                device=device,
            ).item()
        )

    def _kind_indices(
        self,
        batch_size: int,
        device: torch.device,
        forced_kind: str | int | None,
    ) -> torch.Tensor:
        if forced_kind is None:
            return torch.randint(
                0,
                len(self.kinds),
                (batch_size,),
                device=device,
            )
        if isinstance(forced_kind, str):
            normalized = forced_kind.lower()
            if normalized not in self.kinds:
                raise ValueError(
                    f"Degradation kind {forced_kind!r} is not configured."
                )
            index = self.kinds.index(normalized)
        else:
            index = int(forced_kind)
            if index < 0 or index >= len(self.kinds):
                raise ValueError("forced_kind index is out of range.")
        return torch.full((batch_size,), index, device=device, dtype=torch.long)

    def _strength_grid(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = int(image.shape[0])
        device = image.device
        strength = torch.zeros(
            batch_size,
            self.num_phases,
            *self.patch_grid,
            device=device,
            dtype=torch.float32,
        )
        region_min, region_max = self.regions_per_sample
        size_min, size_max = self.region_size_tokens
        severity_min, severity_max = self.severity
        for batch_index in range(batch_size):
            region_count = self._randint(
                region_min,
                region_max,
                device=device,
            )
            for _ in range(region_count):
                sizes = [
                    min(
                        self._randint(size_min, size_max, device=device),
                        grid_size,
                    )
                    for grid_size in self.patch_grid
                ]
                starts = [
                    self._randint(
                        0,
                        grid_size - region_size,
                        device=device,
                    )
                    for grid_size, region_size in zip(self.patch_grid, sizes)
                ]
                region_severity = self._uniform(
                    severity_min,
                    severity_max,
                    device=device,
                    dtype=torch.float32,
                )
                if (
                    torch.rand((), device=device).item()
                    < self.shared_phase_probability
                ):
                    phase_indices = range(self.num_phases)
                else:
                    phase_indices = (
                        self._randint(0, self.num_phases - 1, device=device),
                    )
                region = tuple(
                    slice(start, start + size)
                    for start, size in zip(starts, sizes)
                )
                for phase_index in phase_indices:
                    current = strength[(batch_index, phase_index, *region)]
                    strength[(batch_index, phase_index, *region)] = torch.maximum(
                        current,
                        region_severity,
                    )
        return strength

    def _apply_noise(
        self,
        sample: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        return sample + torch.randn_like(sample) * self.noise_std * strength

    def _apply_blur(
        self,
        sample: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        kernel = self.blur_kernel_size
        blurred = F.avg_pool3d(
            sample[:, None].float(),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )[:, 0].to(sample.dtype)
        return sample * (1.0 - strength) + blurred * strength

    def _apply_stripes(
        self,
        sample: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        spatial_shape = sample.shape[-3:]
        axis = self._randint(0, 2, device=sample.device)
        stripe_width = self._randint(1, 3, device=sample.device)
        coordinate_shape = [1, 1, 1, 1]
        coordinate_shape[axis + 1] = spatial_shape[axis]
        coordinate = torch.arange(
            spatial_shape[axis],
            device=sample.device,
            dtype=sample.dtype,
        ).reshape(coordinate_shape)
        pattern = ((coordinate // stripe_width).remainder(2.0) * 2.0) - 1.0
        return sample + pattern * self.stripe_amplitude * strength

    def _apply_slice_loss(
        self,
        sample: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        stride = self._randint(2, 5, device=sample.device)
        offset = self._randint(0, stride - 1, device=sample.device)
        coordinate = torch.arange(
            sample.shape[-3],
            device=sample.device,
        ).reshape(1, -1, 1, 1)
        lost_slice = ((coordinate + offset).remainder(stride) == 0).to(
            sample.dtype
        )
        lost_strength = strength * lost_slice
        return sample * (1.0 - lost_strength)

    def _apply_intensity(
        self,
        sample: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        scale = self._uniform(
            *self.intensity_scale,
            device=sample.device,
            dtype=sample.dtype,
        )
        shift = self._uniform(
            *self.intensity_shift,
            device=sample.device,
            dtype=sample.dtype,
        )
        locally_changed = sample * scale + shift
        return sample * (1.0 - strength) + locally_changed * strength

    def forward(
        self,
        image: torch.Tensor,
        forced_kind: str | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 5 or int(image.shape[1]) != self.num_phases:
            raise ValueError(
                f"Expected [B,{self.num_phases},D,H,W], got {tuple(image.shape)}."
            )
        batch_size = int(image.shape[0])
        strength_grid = self._strength_grid(image)
        strength_volume = F.interpolate(
            strength_grid.reshape(
                batch_size * self.num_phases,
                1,
                *self.patch_grid,
            ),
            size=image.shape[-3:],
            mode="nearest",
        ).reshape_as(image).to(image.dtype)
        kind_indices = self._kind_indices(
            batch_size,
            image.device,
            forced_kind,
        )
        degraded = []
        operations = {
            "noise": self._apply_noise,
            "blur": self._apply_blur,
            "stripes": self._apply_stripes,
            "slice_loss": self._apply_slice_loss,
            "intensity": self._apply_intensity,
        }
        for batch_index in range(batch_size):
            kind = self.kinds[int(kind_indices[batch_index].item())]
            degraded.append(
                operations[kind](
                    image[batch_index],
                    strength_volume[batch_index],
                )
            )
        token_strength = strength_grid.flatten(2)
        return torch.stack(degraded, dim=0), token_strength, kind_indices


class RegionalFeatureStabilityScorer(nn.Module):
    """Estimate the stability score of each regional imaging feature."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        reliability_cfg = model_cfg.get("token_reliability", {})
        self.num_phases = int(model_cfg["num_phases"])
        self.token_dim = int(model_cfg["token_dim"])
        hidden_dim = int(reliability_cfg.get("hidden_dim", self.token_dim))
        self.minimum_probability = float(
            reliability_cfg.get("minimum_probability", 0.02)
        )
        feature_dim = self.token_dim + 1
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.phase_embedding = nn.Parameter(
            torch.zeros(1, self.num_phases, 1, hidden_dim)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        initial_probability = float(
            reliability_cfg.get("initial_probability", 0.90)
        )
        initial_weight_std = float(
            reliability_cfg.get("initial_weight_std", 1.0e-3)
        )
        initial_probability = min(max(initial_probability, 1e-4), 1.0 - 1e-4)
        nn.init.normal_(self.phase_embedding, std=0.02)
        nn.init.normal_(
            self.output.weight,
            mean=0.0,
            std=initial_weight_std,
        )
        nn.init.constant_(
            self.output.bias,
            math.log(initial_probability / (1.0 - initial_probability)),
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if fused_tokens.ndim != 4:
            raise ValueError("fused_tokens must have shape [B,P,N,C].")
        if (
            int(fused_tokens.shape[1]) != self.num_phases
            or int(fused_tokens.shape[-1]) != self.token_dim
        ):
            raise ValueError("Token reliability input does not match the model.")
        normalized = F.layer_norm(fused_tokens.float(), (self.token_dim,))
        token_scale = (
            fused_tokens.float().square().mean(dim=-1, keepdim=True).sqrt()
        )
        features = torch.cat([normalized, token_scale], dim=-1)
        hidden = self.input_projection(self.feature_norm(features))
        hidden = hidden + self.phase_embedding
        hidden = hidden + self.residual(hidden)
        reliability = torch.sigmoid(self.output(self.output_norm(hidden))).squeeze(-1)
        return reliability.clamp(self.minimum_probability, 1.0 - 1e-5)


@torch.no_grad()
def token_stability_target(
    clean_fused_tokens: torch.Tensor,
    degraded_fused_tokens: torch.Tensor,
    corruption_strength: torch.Tensor,
    *,
    temperature: float,
    corruption_penalty: float,
    minimum_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_shape = clean_fused_tokens.shape[:-1]
    if degraded_fused_tokens.shape != clean_fused_tokens.shape:
        raise ValueError("Clean and degraded token tensors must align.")
    if corruption_strength.shape != expected_shape:
        raise ValueError("corruption_strength must align with token positions.")
    representation_drift = 1.0 - F.cosine_similarity(
        clean_fused_tokens.float(),
        degraded_fused_tokens.float(),
        dim=-1,
    )
    representation_drift = representation_drift.clamp_min(0.0)
    stability = torch.exp(
        -representation_drift / max(float(temperature), 1e-6)
    )
    known_damage = 1.0 - float(corruption_penalty) * corruption_strength.float()
    target = (stability * known_damage).clamp(
        float(minimum_probability),
        1.0 - 1e-5,
    )
    return target, representation_drift


def token_reliability_calibration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    corruption_strength: torch.Tensor,
    *,
    ranking_margin: float,
    ranking_weight: float,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or target.shape != corruption_strength.shape:
        raise ValueError("Reliability calibration tensors must have equal shapes.")
    error = F.smooth_l1_loss(
        prediction.float(),
        target.float(),
        reduction="none",
        beta=0.05,
    )
    corrupted_weight = corruption_strength.float().clamp(0.0, 1.0)
    clean_weight = 1.0 - corrupted_weight
    corrupted_loss = (error * corrupted_weight).sum() / corrupted_weight.sum().clamp_min(
        1.0
    )
    clean_loss = (error * clean_weight).sum() / clean_weight.sum().clamp_min(1.0)
    corrupted_prediction = (
        prediction.float() * corrupted_weight
    ).sum(dim=(1, 2)) / corrupted_weight.sum(dim=(1, 2)).clamp_min(1.0)
    clean_prediction = (
        prediction.float() * clean_weight
    ).sum(dim=(1, 2)) / clean_weight.sum(dim=(1, 2)).clamp_min(1.0)
    ranking_loss = F.relu(
        float(ranking_margin) + corrupted_prediction - clean_prediction
    ).mean()
    regression_loss = 0.5 * (clean_loss + corrupted_loss)
    total = regression_loss + float(ranking_weight) * ranking_loss
    return {
        "token_reliability_calibration_loss": total,
        "token_reliability_regression_loss": regression_loss,
        "token_reliability_ranking_loss": ranking_loss,
        "token_reliability_clean_mean": clean_prediction.mean(),
        "token_reliability_corrupted_mean": corrupted_prediction.mean(),
        "token_reliability_target_mean": target.float().mean(),
        "token_reliability_prediction_mae": (
            prediction.float() - target.float()
        ).abs().mean(),
    }


IterativeAsymmetricHardGate = MeanAnchoredRankModulation
TokenReliabilityEstimator = RegionalFeatureStabilityScorer
