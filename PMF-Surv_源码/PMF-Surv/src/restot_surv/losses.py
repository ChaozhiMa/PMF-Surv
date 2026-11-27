from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

from .models.model import PMFSurv
from .models.survival import (
    cox_partial_likelihood,
    distributed_cox_partial_likelihood,
    endpoint_consistency,
    ipcw_event_weights,
)


AUXILIARY_COMPONENT_KEYS = (
    "reconstruction_intensity_loss",
    "reconstruction_edge_loss",
    "reconstruction_high_frequency_loss",
    "reconstruction_mask_fraction",
    "spatial_absolute_loss",
    "spatial_relative_loss",
    "spatial_continuity_loss",
    "spatial_triplane_loss",
    "dynamic_patient_contrastive_loss",
    "dynamic_transition_contrastive_loss",
    "dynamic_contrastive_active",
    "dynamic_contrastive_local_batch_fraction",
    "dynamic_phase_loss",
    "dynamic_replacement_loss",
)


@dataclass(frozen=True)
class LossWeights:
    survival: float
    reconstruction: float
    spatial: float
    dynamic: float
    os_rank: float
    pfs_rank: float
    post_structure_regularization: float
    ecomix_scale: float


class LossSchedule:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.root_cfg = cfg
        self.cfg = cfg["schedule"]
        self.warmup_end = int(self.cfg["structure_end_epoch"])
        self.transition_end = int(self.cfg["transition_end_epoch"])
        self.warmup_regularization_scale = float(
            self.cfg["warmup_regularization_scale"]
        )

    @staticmethod
    def _weights(
        cfg: Mapping[str, Any],
        values: Mapping[str, Any],
        rank_scale: float,
        post_structure_regularization: float,
        ecomix_scale: float,
    ) -> LossWeights:
        loss_cfg = cfg["loss"]
        return LossWeights(
            survival=float(values["survival"]),
            reconstruction=float(values["reconstruction"]),
            spatial=float(values["spatial"]),
            dynamic=float(values["dynamic"]),
            os_rank=float(loss_cfg["os_rank_weight"]) * rank_scale,
            pfs_rank=float(loss_cfg["pfs_rank_weight"]) * rank_scale,
            post_structure_regularization=float(post_structure_regularization),
            ecomix_scale=float(ecomix_scale),
        )

    def _rank_scale(self, epoch: int) -> float:
        warmup = max(int(self.root_cfg["loss"].get("rank_warmup_epochs", 1)), 1)
        return min(max(float(epoch) / float(warmup), 0.0), 1.0)

    def _transition_progress(
        self,
        epoch: int,
        step_in_epoch: int | None,
        steps_per_epoch: int | None,
    ) -> float:
        transition_epochs = max(self.transition_end - self.warmup_end, 1)
        if step_in_epoch is None or steps_per_epoch is None:
            return min(max((epoch - self.warmup_end) / transition_epochs, 0.0), 1.0)
        steps_per_epoch = max(int(steps_per_epoch), 1)
        completed_steps = (
            (epoch - self.warmup_end - 1) * steps_per_epoch
            + min(max(int(step_in_epoch), 1), steps_per_epoch)
        )
        total_steps = transition_epochs * steps_per_epoch
        return min(max(completed_steps / total_steps, 0.0), 1.0)

    def at(
        self,
        epoch: int,
        step_in_epoch: int | None = None,
        steps_per_epoch: int | None = None,
    ) -> LossWeights:
        if epoch <= self.warmup_end:
            return self._weights(
                self.root_cfg,
                self.cfg["warmup"],
                self._rank_scale(epoch),
                self.warmup_regularization_scale,
                0.0,
            )
        if epoch <= self.transition_end:
            start = self._weights(
                self.root_cfg,
                self.cfg["transition_start"],
                self._rank_scale(epoch),
                self.warmup_regularization_scale,
                0.0,
            )
            end = self._weights(
                self.root_cfg,
                self.cfg["transition_end"],
                self._rank_scale(epoch),
                1.0,
                1.0,
            )
            progress = self._transition_progress(epoch, step_in_epoch, steps_per_epoch)
            return LossWeights(
                survival=start.survival
                + progress * (end.survival - start.survival),
                reconstruction=start.reconstruction
                + progress * (end.reconstruction - start.reconstruction),
                spatial=start.spatial + progress * (end.spatial - start.spatial),
                dynamic=start.dynamic + progress * (end.dynamic - start.dynamic),
                os_rank=end.os_rank,
                pfs_rank=end.pfs_rank,
                post_structure_regularization=start.post_structure_regularization
                + progress
                * (
                    end.post_structure_regularization
                    - start.post_structure_regularization
                ),
                ecomix_scale=progress,
            )
        return self._weights(
            self.root_cfg,
            self.cfg["main"],
            self._rank_scale(epoch),
            1.0,
            1.0,
        )


def pairwise_rank_loss(
    risks: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    temperature: float,
    event_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    risk = risks.float().reshape(-1)
    time = times.float().reshape(-1)
    event = events.float().reshape(-1) > 0.5
    comparable = event[:, None] & (time[:, None] < time[None, :])
    if not comparable.any():
        return risk.sum() * 0.0
    differences = risk[:, None] - risk[None, :]
    scale = max(float(temperature), 1e-6)
    losses = F.softplus(-differences[comparable] / scale)
    if event_weights is None:
        return losses.mean()
    weights = event_weights.float().reshape(-1)
    if weights.shape != risk.shape:
        raise ValueError("event_weights must contain one value per patient.")
    pair_weights = weights[:, None].expand_as(comparable)[comparable]
    return (losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)


def distributed_pairwise_rank_loss(
    risks: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    temperature: float,
    event_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute one differentiable ranking loss over the full DDP batch."""

    if not dist.is_available() or not dist.is_initialized():
        return pairwise_rank_loss(
            risks, times, events, temperature, event_weights
        )
    world_size = dist.get_world_size()
    if world_size <= 1:
        return pairwise_rank_loss(
            risks, times, events, temperature, event_weights
        )

    batch_sizes = [
        torch.zeros((), dtype=torch.long, device=risks.device)
        for _ in range(world_size)
    ]
    local_size = torch.tensor(
        risks.shape[0],
        dtype=torch.long,
        device=risks.device,
    )
    dist.all_gather(batch_sizes, local_size)
    if len({int(value.item()) for value in batch_sizes}) != 1:
        raise RuntimeError(
            "Distributed ranking requires the same local batch size on every rank."
        )

    global_risks = torch.cat(all_gather(risks.float()), dim=0)
    gathered_times = [
        torch.empty_like(times.float()) for _ in range(world_size)
    ]
    gathered_events = [
        torch.empty_like(events.float()) for _ in range(world_size)
    ]
    dist.all_gather(gathered_times, times.float().contiguous())
    dist.all_gather(gathered_events, events.float().contiguous())
    global_event_weights = None
    if event_weights is not None:
        gathered_weights = [
            torch.empty_like(event_weights.float()) for _ in range(world_size)
        ]
        dist.all_gather(
            gathered_weights,
            event_weights.float().contiguous(),
        )
        global_event_weights = torch.cat(gathered_weights, dim=0)
    return pairwise_rank_loss(
        global_risks,
        torch.cat(gathered_times, dim=0),
        torch.cat(gathered_events, dim=0),
        temperature,
        global_event_weights,
    )


def _clinical_distance(
    continuous: torch.Tensor,
    categorical: torch.Tensor,
    composite: torch.Tensor,
) -> torch.Tensor:
    parts = []
    if continuous.numel():
        parts.append(torch.cdist(continuous.float(), continuous.float()) / max(continuous.shape[1], 1) ** 0.5)
    for values in (categorical, composite):
        if values.numel():
            parts.append((values[:, None, :] != values[None, :, :]).float().mean(dim=-1))
    return torch.stack(parts).mean(dim=0) if parts else continuous.new_zeros((len(continuous), len(continuous)))


def _endpoint_distance(time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    time = time.float()
    scale = time.std().clamp_min(1.0)
    time_distance = (time[:, None] - time[None, :]).abs() / scale
    event_distance = (event[:, None] != event[None, :]).float()
    return time_distance + event_distance


def sample_ecomix_pairs(
    batch: Mapping[str, Any],
    top_k: int,
    temperature: float,
    endpoint_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    continuous = batch["clinical_continuous"]
    distance = _clinical_distance(
        continuous,
        batch["clinical_categorical"],
        batch["clinical_composite"],
    )
    distance = distance + endpoint_weight * (
        _endpoint_distance(batch["os_time"], batch["os_event"])
        + _endpoint_distance(batch["pfs_time"], batch["pfs_event"])
    )
    batch_size = distance.shape[0]
    patient_ids = [str(value) for value in batch["patient_id"]]
    forbidden = torch.tensor(
        [[patient_ids[i] == patient_ids[j] for j in range(batch_size)] for i in range(batch_size)],
        device=distance.device,
        dtype=torch.bool,
    )
    distance = distance.masked_fill(forbidden, float("inf"))
    pairs = torch.arange(batch_size, device=distance.device)
    valid = torch.zeros(batch_size, dtype=torch.bool, device=distance.device)
    for index in range(batch_size):
        candidate_indices = torch.nonzero(
            torch.isfinite(distance[index]),
            as_tuple=False,
        ).squeeze(1)
        if candidate_indices.numel() == 0:
            continue
        candidate_distances = distance[index, candidate_indices]
        count = min(max(int(top_k), 1), int(candidate_indices.numel()))
        nearest = torch.topk(
            candidate_distances,
            k=count,
            largest=False,
        ).indices
        candidate_indices = candidate_indices[nearest]
        candidate_distances = candidate_distances[nearest]
        probabilities = torch.softmax(
            -candidate_distances / max(float(temperature), 1e-6),
            dim=0,
        )
        selected = torch.multinomial(probabilities, 1).item()
        pairs[index] = candidate_indices[selected]
        valid[index] = True
    return pairs, valid


def sample_anchor_mix_weights(
    batch_size: int,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    concentration = torch.tensor(float(alpha), device=device)
    raw = torch.distributions.Beta(
        concentration,
        concentration,
    ).sample((int(batch_size),))
    return torch.maximum(raw, 1.0 - raw).clamp(0.5, 0.95)


class Objective:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.loss_cfg = cfg["loss"]
        self.schedule = LossSchedule(cfg)
        self.ecomix_cfg = self.loss_cfg["ecomix"]
        reliability_cfg = cfg["model"].get("token_reliability", {})
        calibration_cfg = reliability_cfg.get("calibration", {})
        self.token_reliability_loss_weight = (
            float(calibration_cfg.get("loss_weight", 0.0))
            if bool(reliability_cfg.get("enabled", False))
            else 0.0
        )
    def _rank_event_weights(
        self,
        head: torch.nn.Module,
        times: torch.Tensor,
        events: torch.Tensor,
        *,
        endpoint: str,
    ) -> torch.Tensor | None:
        ipcw_cfg = self.loss_cfg.get("rank_ipcw", {})
        if not bool(ipcw_cfg.get("enabled", False)):
            return None
        if endpoint == "pfs" and not bool(
            ipcw_cfg.get("apply_to_pfs", False)
        ):
            return None
        return ipcw_event_weights(
            times,
            events,
            head.censoring_times,
            head.censoring_survival,
            survival_floor=float(ipcw_cfg.get("survival_floor", 0.1)),
            weight_cap=head.rank_ipcw_cap,
        )

    def _endpoint_consistency(
        self,
        model: PMFSurv,
        os_survival: torch.Tensor,
        pfs_survival: torch.Tensor,
    ) -> torch.Tensor:
        if model.os_prediction_mode == "time_query_independent":
            os_survival = os_survival.detach()
        return endpoint_consistency(
            os_survival,
            pfs_survival,
            model.os_head.boundaries,
            model.pfs_head.boundaries,
        )

    def _weighted_endpoint_consistency(
        self,
        model: PMFSurv,
        os_survival: torch.Tensor,
        pfs_survival: torch.Tensor,
    ) -> torch.Tensor:
        return (
            float(self.loss_cfg["endpoint_consistency_weight"])
            * self._endpoint_consistency(model, os_survival, pfs_survival)
        )

    def _survival_loss(
        self,
        model: PMFSurv,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        post_structure_regularization: float,
        os_rank_weight: float,
        pfs_rank_weight: float,
        distributed_cox: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        os_survival_nll = model.os_head.loss(
            outputs["os_head_output"], batch["os_time"], batch["os_event"]
        )
        pfs_survival_nll = model.pfs_head.loss(
            outputs["pfs_head_output"], batch["pfs_time"], batch["pfs_event"]
        )
        zero = outputs["os_risk"].sum() * 0.0
        cox_loss = (
            distributed_cox_partial_likelihood
            if distributed_cox
            else cox_partial_likelihood
        )
        os_cox = (
            cox_loss(
                outputs["os_risk"],
                batch["os_time"],
                batch["os_event"],
            )
            if float(self.loss_cfg["os_cox_weight"]) > 0.0
            else zero
        )
        pfs_cox = (
            cox_loss(
                outputs["pfs_risk"],
                batch["pfs_time"],
                batch["pfs_event"],
            )
            if float(self.loss_cfg["pfs_cox_weight"]) > 0.0
            else zero
        )
        brier_weight = float(self.loss_cfg["brier_weight"])
        if brier_weight > 0.0:
            os_brier = model.os_head.brier_loss(
                outputs["os_head_output"], batch["os_time"], batch["os_event"]
            )
            pfs_brier = model.pfs_head.brier_loss(
                outputs["pfs_head_output"], batch["pfs_time"], batch["pfs_event"]
            )
        else:
            os_brier = zero
            pfs_brier = zero
        rank_loss = (
            distributed_pairwise_rank_loss
            if distributed_cox
            else pairwise_rank_loss
        )
        os_rank_event_weights = self._rank_event_weights(
            model.os_head,
            batch["os_time"],
            batch["os_event"],
            endpoint="os",
        )
        pfs_rank_event_weights = self._rank_event_weights(
            model.pfs_head,
            batch["pfs_time"],
            batch["pfs_event"],
            endpoint="pfs",
        )
        os_rank = (
            rank_loss(
                outputs["os_risk"],
                batch["os_time"],
                batch["os_event"],
                float(self.loss_cfg["rank_temperature"]),
                os_rank_event_weights,
            )
            if float(os_rank_weight) > 0.0
            else zero
        )
        pfs_rank = (
            rank_loss(
                outputs["pfs_risk"],
                batch["pfs_time"],
                batch["pfs_event"],
                float(self.loss_cfg["rank_temperature"]),
                pfs_rank_event_weights,
            )
            if float(pfs_rank_weight) > 0.0
            else zero
        )
        consistency = self._endpoint_consistency(
            model,
            outputs["os_survival"],
            outputs["pfs_survival"],
        )
        os_loss = (
            os_survival_nll
            + float(self.loss_cfg["os_cox_weight"]) * os_cox
            + post_structure_regularization * brier_weight * os_brier
            + float(os_rank_weight) * os_rank
        )
        pfs_loss = (
            pfs_survival_nll
            + float(self.loss_cfg["pfs_cox_weight"]) * pfs_cox
            + post_structure_regularization * brier_weight * pfs_brier
            + float(pfs_rank_weight) * pfs_rank
        )
        total = (
            float(self.loss_cfg["os_weight"]) * os_loss
            + float(self.loss_cfg["pfs_weight"]) * pfs_loss
            + float(self.loss_cfg["endpoint_consistency_weight"]) * consistency
        )
        components = {
            "os_survival_nll": os_survival_nll,
            "pfs_survival_nll": pfs_survival_nll,
            "os_cox_loss": os_cox,
            "pfs_cox_loss": pfs_cox,
            "os_brier_loss": os_brier,
            "pfs_brier_loss": pfs_brier,
            "os_rank_loss": os_rank,
            "pfs_rank_loss": pfs_rank,
            "endpoint_consistency_loss": consistency,
        }
        if model.survival_head_type == "piecewise_exponential":
            components.update(
                {
                    "os_piecewise_exponential_nll": os_survival_nll,
                    "pfs_piecewise_exponential_nll": pfs_survival_nll,
                }
            )
        else:
            components.update(
                {
                    "os_mtlr_loss": os_survival_nll,
                    "pfs_mtlr_loss": pfs_survival_nll,
                }
            )
        return total, components

    def _ecomix_loss(
        self,
        model: PMFSurv,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        epoch: int,
    ) -> torch.Tensor:
        zero = outputs["image_tokens"].new_zeros(())
        if (
            not bool(self.ecomix_cfg["enabled"])
            or outputs["image_tokens"].shape[0] < 2
            or torch.rand((), device=outputs["image_tokens"].device).item()
            > float(self.ecomix_cfg["probability"])
        ):
            return zero
        permutation, valid = sample_ecomix_pairs(
            batch,
            int(self.ecomix_cfg["top_k"]),
            float(self.ecomix_cfg["temperature"]),
            float(self.ecomix_cfg["endpoint_weight"]),
        )
        if not valid.any():
            return zero
        lam = sample_anchor_mix_weights(
            outputs["image_tokens"].shape[0],
            float(self.ecomix_cfg["alpha"]),
            device=outputs["image_tokens"].device,
        )
        mixed_outputs = model.mix_clinical_evidence(
            outputs["image_tokens"],
            outputs["clinical_tokens"],
            permutation,
            lam,
        )
        prediction = model.predict_from_evidence(
            mixed_outputs["mixed_clinical_tokens"][valid],
            mixed_outputs["scaled_matched_tokens"][valid],
            mixed_outputs["image_risk_tokens"][valid],
        )
        target_weight = lam[valid, None]
        target_os = (
            target_weight * outputs["os_survival"].detach()[valid]
            + (1.0 - target_weight)
            * outputs["os_survival"].detach()[permutation[valid]]
        )
        target_pfs = (
            target_weight * outputs["pfs_survival"].detach()[valid]
            + (1.0 - target_weight)
            * outputs["pfs_survival"].detach()[permutation[valid]]
        )
        smooth = F.mse_loss(prediction["os_survival"], target_os)
        smooth = smooth + float(self.loss_cfg["pfs_weight"]) * F.mse_loss(prediction["pfs_survival"], target_pfs)
        consistent = self._weighted_endpoint_consistency(
            model,
            prediction["os_survival"],
            prediction["pfs_survival"],
        )
        return smooth + consistent

    def __call__(
        self,
        model: PMFSurv,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        epoch: int,
        training: bool,
        step_in_epoch: int | None = None,
        steps_per_epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], LossWeights]:
        weights = self.schedule.at(epoch, step_in_epoch, steps_per_epoch)
        survival, components = self._survival_loss(
            model,
            outputs,
            batch,
            weights.post_structure_regularization,
            weights.os_rank,
            weights.pfs_rank,
            distributed_cox=training,
        )
        reconstruction = outputs.get("reconstruction_loss", survival.new_zeros(()))
        spatial = outputs.get("spatial_loss", survival.new_zeros(()))
        dynamic = outputs.get("dynamic_loss", survival.new_zeros(()))
        token_reliability_calibration = outputs.get(
            "token_reliability_calibration_loss",
            survival.new_zeros(()),
        )
        ecomix = (
            self._ecomix_loss(model, outputs, batch, epoch)
            if training and weights.ecomix_scale > 0
            else survival.new_zeros(())
        )
        total = (
            weights.survival * survival
            + weights.reconstruction * reconstruction
            + weights.spatial * spatial
            + weights.dynamic * dynamic
            + self.token_reliability_loss_weight
            * token_reliability_calibration
            + weights.ecomix_scale
            * float(self.ecomix_cfg["weight"])
            * ecomix
        )
        components.update(
            {
                "survival_loss": survival,
                "reconstruction_loss": reconstruction,
                "spatial_loss": spatial,
                "dynamic_loss": dynamic,
                "token_reliability_calibration_loss": (
                    token_reliability_calibration
                ),
                "token_reliability_calibration_weight": (
                    survival.new_tensor(self.token_reliability_loss_weight)
                ),
                "ecomix_loss": ecomix,
                "total_loss": total,
            }
        )
        for key, value in outputs.items():
            if (
                key.startswith(
                    (
                        "reconstruction_",
                        "spatial_",
                        "dynamic_",
                        "token_reliability_",
                        "token_quality_",
                    )
                )
                and key not in components
                and isinstance(value, torch.Tensor)
                and value.ndim == 0
            ):
                components[key] = value
        for key in AUXILIARY_COMPONENT_KEYS:
            components.setdefault(key, survival.new_zeros(()))
        return total, components, weights
