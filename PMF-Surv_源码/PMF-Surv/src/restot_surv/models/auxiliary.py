from __future__ import annotations

from itertools import product
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

from .cross_phase_fusion import CrossPhaseFusion


def patchify_3d(image: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    pd, ph, pw = patch_size
    batch, phases, depth, height, width = image.shape
    if depth % pd or height % ph or width % pw:
        raise ValueError("Image dimensions must be divisible by patch_size.")
    patches = image.reshape(
        batch,
        phases,
        depth // pd,
        pd,
        height // ph,
        ph,
        width // pw,
        pw,
    )
    patches = patches.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
    return patches.reshape(batch, phases, -1, pd * ph * pw)


class ReconstructionAuxiliary(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        auxiliary_cfg = cfg["auxiliary"]
        self.num_phases = int(model_cfg["num_phases"])
        self.patch_size = tuple(int(v) for v in model_cfg["patch_size"])
        self.patch_grid = tuple(int(v) for v in model_cfg["patch_grid"])
        self.mask_ratio = float(auxiliary_cfg["reconstruction_mask_ratio"])
        self.shared_mask_fraction = float(
            auxiliary_cfg["reconstruction_shared_mask_fraction"]
        )
        self.block_size = tuple(
            int(v) for v in auxiliary_cfg["reconstruction_block_size"]
        )
        self.edge_weight = float(auxiliary_cfg["reconstruction_edge_weight"])
        self.high_frequency_weight = float(
            auxiliary_cfg["reconstruction_high_frequency_weight"]
        )
        self.token_dim = int(model_cfg["token_dim"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.token_dim))
        self.decoder = CrossPhaseFusion(
            cfg,
            depth_override=int(auxiliary_cfg["reconstruction_decoder_depth"]),
        )
        self.patch_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, int(torch.tensor(self.patch_size).prod().item())),
        )
        nn.init.normal_(self.mask_token, std=0.02)

    def _random_block(self, device: torch.device) -> tuple[slice, slice, slice]:
        minimum, maximum = self.block_size
        sizes = []
        starts = []
        for grid_size in self.patch_grid:
            low = min(max(minimum, 1), grid_size)
            high = min(max(maximum, low), grid_size)
            size = int(torch.randint(low, high + 1, (), device=device).item())
            start = int(torch.randint(0, grid_size - size + 1, (), device=device).item())
            sizes.append(size)
            starts.append(start)
        return tuple(
            slice(start, start + size)
            for start, size in zip(starts, sizes)
        )

    def _structured_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, phases, token_count, _ = tokens.shape
        depth, height, width = self.patch_grid
        if token_count != depth * height * width:
            raise ValueError("Token count does not match model.patch_grid.")
        target = max(1, int(round(phases * token_count * self.mask_ratio)))
        shared_target = int(round(target * self.shared_mask_fraction))
        masks = []
        for _ in range(batch):
            mask = torch.zeros(
                phases,
                depth,
                height,
                width,
                dtype=torch.bool,
                device=tokens.device,
            )
            shared = torch.zeros(
                depth,
                height,
                width,
                dtype=torch.bool,
                device=tokens.device,
            )
            attempts = 0
            while int(mask.sum().item()) < shared_target and attempts < 64:
                block = self._random_block(tokens.device)
                shared[block] = True
                mask[(slice(None),) + block] = True
                attempts += 1

            attempts = 0
            while int(mask.sum().item()) < target and attempts < 128:
                block = self._random_block(tokens.device)
                phase_count = int(
                    torch.randint(1, max(phases, 2), (), device=tokens.device).item()
                )
                selected = torch.randperm(phases, device=tokens.device)[:phase_count]
                for phase in selected.tolist():
                    mask[(int(phase),) + block] = True
                fully_masked = mask.all(dim=0) & ~shared
                if fully_masked.any():
                    locations = torch.nonzero(fully_masked, as_tuple=False)
                    visible = torch.randint(
                        0,
                        phases,
                        (locations.shape[0],),
                        device=tokens.device,
                    )
                    mask[
                        visible,
                        locations[:, 0],
                        locations[:, 1],
                        locations[:, 2],
                    ] = False
                attempts += 1

            if int(mask.sum().item()) < target:
                phase_flat = mask.flatten(1)
                shared_flat = shared.flatten()
                available = ~phase_flat
                can_hide_more = (
                    phase_flat.sum(dim=0) < phases - 1
                ) | shared_flat
                candidates = torch.nonzero(
                    available & can_hide_more.unsqueeze(0),
                    as_tuple=False,
                )
                needed = min(
                    target - int(mask.sum().item()),
                    int(candidates.shape[0]),
                )
                if needed > 0:
                    chosen = candidates[
                        torch.randperm(candidates.shape[0], device=tokens.device)[:needed]
                    ]
                    phase_flat[chosen[:, 0], chosen[:, 1]] = True
                mask = phase_flat.reshape_as(mask)
            masks.append(mask.flatten(1))
        return torch.stack(masks)

    @staticmethod
    def _edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for dimension in (2, 3, 4):
            prediction_gradient = torch.diff(prediction, dim=dimension)
            target_gradient = torch.diff(target, dim=dimension)
            losses.append(
                F.smooth_l1_loss(prediction_gradient, target_gradient)
            )
        return torch.stack(losses).mean()

    @staticmethod
    def _high_frequency_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction_high = prediction - F.avg_pool3d(
            prediction,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        target_high = target - F.avg_pool3d(
            target,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return F.smooth_l1_loss(prediction_high, target_high)

    def forward(
        self,
        tokens: torch.Tensor,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = self._structured_mask(tokens)
        masked_tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        decoded = self.decoder(masked_tokens, add_embeddings=True)
        predictions = self.patch_head(decoded[mask])
        targets = patchify_3d(image, self.patch_size)[mask]
        patch_shape = (predictions.shape[0], 1, *self.patch_size)
        prediction_volumes = predictions.float().reshape(patch_shape)
        target_volumes = targets.float().reshape(patch_shape)
        intensity = F.smooth_l1_loss(prediction_volumes, target_volumes)
        edge = self._edge_loss(prediction_volumes, target_volumes)
        high_frequency = self._high_frequency_loss(
            prediction_volumes,
            target_volumes,
        )
        total = (
            intensity
            + self.edge_weight * edge
            + self.high_frequency_weight * high_frequency
        )
        return {
            "reconstruction_loss": total,
            "reconstruction_intensity_loss": intensity,
            "reconstruction_edge_loss": edge,
            "reconstruction_high_frequency_loss": high_frequency,
            "reconstruction_mask_fraction": mask.float().mean().detach(),
        }


class SpatialAuxiliary(nn.Module):
    """Learn absolute, relative, local-continuity, and tri-plane structure."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        auxiliary_cfg = cfg["auxiliary"]
        self.patch_grid = tuple(int(v) for v in model_cfg["patch_grid"])
        token_dim = int(model_cfg["token_dim"])
        hidden = token_dim // 2
        self.relative_pairs = int(auxiliary_cfg["spatial_relative_pairs"])
        self.continuity_margin = float(
            auxiliary_cfg["spatial_continuity_margin"]
        )
        self.absolute_weight = float(auxiliary_cfg["spatial_absolute_weight"])
        self.relative_weight = float(auxiliary_cfg["spatial_relative_weight"])
        self.continuity_weight = float(
            auxiliary_cfg["spatial_continuity_weight"]
        )
        self.triplane_weight = float(auxiliary_cfg["spatial_triplane_weight"])
        self.absolute_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )
        self.relative_head = nn.Sequential(
            nn.LayerNorm(token_dim * 4),
            nn.Linear(token_dim * 4, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 3),
            nn.Tanh(),
        )
        self.plane_projector = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.triplane_predictor = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        coordinates = []
        for z, y, x in product(
            range(self.patch_grid[0]),
            range(self.patch_grid[1]),
            range(self.patch_grid[2]),
        ):
            coordinates.append(
                (
                    2.0 * z / max(self.patch_grid[0] - 1, 1) - 1.0,
                    2.0 * y / max(self.patch_grid[1] - 1, 1) - 1.0,
                    2.0 * x / max(self.patch_grid[2] - 1, 1) - 1.0,
                )
            )
        self.register_buffer("coordinates", torch.tensor(coordinates, dtype=torch.float32))

    def _relative_loss(self, features: torch.Tensor) -> torch.Tensor:
        batch, token_count, channels = features.shape
        pair_count = min(max(self.relative_pairs, 1), token_count)
        left_index = torch.randint(
            0,
            token_count,
            (batch, pair_count),
            device=features.device,
        )
        right_index = torch.randint(
            0,
            token_count,
            (batch, pair_count),
            device=features.device,
        )
        gather_index = left_index.unsqueeze(-1).expand(-1, -1, channels)
        left = torch.gather(features, 1, gather_index)
        gather_index = right_index.unsqueeze(-1).expand(-1, -1, channels)
        right = torch.gather(features, 1, gather_index)
        relation = torch.cat([left, right, right - left, right * left], dim=-1)
        prediction = self.relative_head(relation)
        coordinates = self.coordinates.to(features.device)
        left_coordinates = coordinates[left_index]
        right_coordinates = coordinates[right_index]
        target = (right_coordinates - left_coordinates) * 0.5
        return F.smooth_l1_loss(prediction.float(), target.float())

    def _continuity_loss(self, features: torch.Tensor) -> torch.Tensor:
        batch, token_count, channels = features.shape
        depth, height, width = self.patch_grid
        grid = F.normalize(features, dim=-1).reshape(
            batch,
            depth,
            height,
            width,
            channels,
        )
        adjacent_distances = []
        for dimension in (1, 2, 3):
            left = grid.narrow(dimension, 0, grid.shape[dimension] - 1)
            right = grid.narrow(dimension, 1, grid.shape[dimension] - 1)
            adjacent_distances.append(1.0 - (left * right).sum(dim=-1))
        adjacent = torch.cat(
            [value.reshape(batch, -1) for value in adjacent_distances],
            dim=1,
        ).mean(dim=1)
        shift = max(token_count // 2, 1)
        distant = 1.0 - (
            F.normalize(features, dim=-1)
            * torch.roll(F.normalize(features, dim=-1), shifts=shift, dims=1)
        ).sum(dim=-1).mean(dim=1)
        return F.relu(self.continuity_margin + adjacent - distant).mean()

    def _triplane_loss(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, channels = features.shape
        depth, height, width = self.patch_grid
        grid = features.reshape(batch, depth, height, width, channels)
        views = (
            grid.mean(dim=(2, 3)),
            grid.mean(dim=(1, 3)),
            grid.mean(dim=(1, 2)),
        )
        axial, coronal, sagittal = [
            self.plane_projector(view)
            for view in views
        ]
        axial = axial[:, :, None, None, :].expand(-1, -1, height, width, -1)
        coronal = coronal[:, None, :, None, :].expand(-1, depth, -1, width, -1)
        sagittal = sagittal[:, None, None, :, :].expand(-1, depth, height, -1, -1)
        prediction = self.triplane_predictor(
            torch.cat([axial, coronal, sagittal], dim=-1)
        ).reshape(batch, -1, channels)
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(features.detach(), dim=-1)
        return (1.0 - (prediction * target).sum(dim=-1)).mean()

    def forward(self, base_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        absolute_prediction = self.absolute_head(base_tokens)
        absolute_target = self.coordinates.view(1, 1, -1, 3).expand_as(
            absolute_prediction
        )
        absolute = F.smooth_l1_loss(
            absolute_prediction.float(),
            absolute_target.float(),
        )
        features = base_tokens.mean(dim=1)
        relative = self._relative_loss(features)
        continuity = self._continuity_loss(features)
        triplane = self._triplane_loss(features)
        total = (
            self.absolute_weight * absolute
            + self.relative_weight * relative
            + self.continuity_weight * continuity
            + self.triplane_weight * triplane
        )
        return {
            "spatial_loss": total,
            "spatial_absolute_loss": absolute,
            "spatial_relative_loss": relative,
            "spatial_continuity_loss": continuity,
            "spatial_triplane_loss": triplane,
        }


def _info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if left.shape[0] * world_size <= 1:
        return left.new_zeros(())
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    if world_size > 1:
        global_left = torch.cat(all_gather(left), dim=0)
        global_right = torch.cat(all_gather(right), dim=0)
        label_offset = dist.get_rank() * left.shape[0]
    else:
        global_left = left
        global_right = right
        label_offset = 0
    labels = (
        torch.arange(left.shape[0], device=left.device)
        + label_offset
    )
    scale = max(float(temperature), 1e-6)
    left_logits = left @ global_right.transpose(0, 1) / scale
    right_logits = right @ global_left.transpose(0, 1) / scale
    return 0.5 * (
        F.cross_entropy(left_logits, labels)
        + F.cross_entropy(right_logits, labels)
    )


class DynamicAuxiliary(nn.Module):
    """Patient and enhancement-transition consistency on the shared token space."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        auxiliary_cfg = cfg["auxiliary"]
        self.num_phases = int(model_cfg["num_phases"])
        token_dim = int(model_cfg["token_dim"])
        hidden = token_dim
        output = token_dim // 2
        self.temperature = float(auxiliary_cfg["contrastive_temperature"])
        self.patient_weight = float(auxiliary_cfg["dynamic_patient_weight"])
        self.transition_weight = float(
            auxiliary_cfg["dynamic_transition_weight"]
        )
        self.phase_weight = float(auxiliary_cfg["dynamic_phase_weight"])
        self.replacement_weight = float(
            auxiliary_cfg["dynamic_replacement_weight"]
        )
        self.patient_projector = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output),
        )
        self.transition_projector = nn.Sequential(
            nn.LayerNorm(token_dim * max(self.num_phases - 1, 1)),
            nn.Linear(token_dim * max(self.num_phases - 1, 1), hidden),
            nn.GELU(),
            nn.Linear(hidden, output),
        )
        self.phase_classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.num_phases),
        )
        self.replacement_classifier = nn.Sequential(
            nn.LayerNorm(token_dim * self.num_phases),
            nn.Linear(token_dim * self.num_phases, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, self.num_phases + 1),
        )

    def _representations(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase_features = tokens.mean(dim=2)
        patient = self.patient_projector(phase_features.mean(dim=1))
        transitions = phase_features[:, 1:] - phase_features[:, :-1]
        transition = (
            self.transition_projector(transitions.flatten(1))
            if self.num_phases > 1
            else torch.zeros_like(patient)
        )
        return patient, transition

    def _replacement_loss(self, phase_features: torch.Tensor) -> torch.Tensor:
        batch = phase_features.shape[0]
        if batch <= 1:
            return phase_features.new_zeros(())
        shift = int(torch.randint(1, batch, (), device=phase_features.device).item())
        donor = torch.roll(phase_features, shifts=shift, dims=0)
        replaced_phase = torch.randint(
            0,
            self.num_phases,
            (batch,),
            device=phase_features.device,
        )
        corrupted = phase_features.clone()
        rows = torch.arange(batch, device=phase_features.device)
        corrupted[rows, replaced_phase] = donor[rows, replaced_phase]
        clean_logits = self.replacement_classifier(phase_features.flatten(1))
        corrupted_logits = self.replacement_classifier(corrupted.flatten(1))
        clean_labels = torch.full(
            (batch,),
            self.num_phases,
            device=phase_features.device,
            dtype=torch.long,
        )
        return 0.5 * (
            F.cross_entropy(clean_logits, clean_labels)
            + F.cross_entropy(corrupted_logits, replaced_phase)
        )

    def forward(
        self,
        primary_tokens: torch.Tensor,
        contrastive_tokens: torch.Tensor | None = None,
        contrastive_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if contrastive_tokens is None:
            patient_loss = primary_tokens.new_zeros(())
            transition_loss = primary_tokens.new_zeros(())
            contrastive_active = primary_tokens.new_zeros(())
            contrastive_fraction = primary_tokens.new_zeros(())
        else:
            if contrastive_indices is None:
                raise ValueError(
                    "contrastive_indices are required with contrastive tokens."
                )
            primary_contrastive_tokens = primary_tokens[
                contrastive_indices.long()
            ]
            if (
                primary_contrastive_tokens.shape[0]
                != contrastive_tokens.shape[0]
            ):
                raise ValueError(
                    "Primary and secondary contrastive batches must match."
                )
            patient_primary, transition_primary = self._representations(
                primary_contrastive_tokens
            )
            patient_contrastive, transition_contrastive = self._representations(
                contrastive_tokens
            )
            patient_loss = _info_nce(
                patient_primary,
                patient_contrastive,
                self.temperature,
            )
            transition_loss = (
                _info_nce(
                    transition_primary,
                    transition_contrastive,
                    self.temperature,
                )
                if self.num_phases > 1
                else primary_tokens.new_zeros(())
            )
            contrastive_active = primary_tokens.new_ones(())
            contrastive_fraction = primary_tokens.new_tensor(
                contrastive_tokens.shape[0] / primary_tokens.shape[0]
            )
        phase_features = primary_tokens.mean(dim=2)
        phase_logits = self.phase_classifier(phase_features.reshape(-1, phase_features.shape[-1]))
        phase_labels = torch.arange(
            self.num_phases,
            device=primary_tokens.device,
        )
        phase_labels = (
            phase_labels.view(1, -1)
            .expand(primary_tokens.shape[0], -1)
            .reshape(-1)
        )
        phase_loss = F.cross_entropy(phase_logits, phase_labels)
        replacement_loss = self._replacement_loss(phase_features)
        total = (
            self.patient_weight * patient_loss
            + self.transition_weight * transition_loss
            + self.phase_weight * phase_loss
            + self.replacement_weight * replacement_loss
        )
        return {
            "dynamic_loss": total,
            "dynamic_patient_contrastive_loss": patient_loss,
            "dynamic_transition_contrastive_loss": transition_loss,
            "dynamic_contrastive_active": contrastive_active,
            "dynamic_contrastive_local_batch_fraction": contrastive_fraction,
            "dynamic_phase_loss": phase_loss,
            "dynamic_replacement_loss": replacement_loss,
        }


class AuxiliaryObjectives(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.reconstruction = ReconstructionAuxiliary(cfg)
        self.spatial = SpatialAuxiliary(cfg)
        self.dynamic = DynamicAuxiliary(cfg)

    def forward(
        self,
        base_tokens: torch.Tensor,
        contextual_tokens: torch.Tensor,
        image: torch.Tensor,
        dynamic_contrastive_tokens: torch.Tensor | None = None,
        dynamic_contrastive_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            **self.reconstruction(contextual_tokens, image),
            **self.spatial(base_tokens),
            **self.dynamic(
                contextual_tokens,
                dynamic_contrastive_tokens,
                dynamic_contrastive_indices,
            ),
        }
