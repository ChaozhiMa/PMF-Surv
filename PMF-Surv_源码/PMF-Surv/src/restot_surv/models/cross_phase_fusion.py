from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer

from .backbones import (
    SharedPhaseConvNeXtTinyEncoder3D,
    SharedPhaseDenseNet121Encoder3D,
    SharedPhaseEfficientNetB1Encoder3D,
    SharedPhaseResNet50Encoder3D,
    SharedPhaseViTSmallEncoder3D,
)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SharedPhaseSwinEncoder3D(nn.Module):
    """One medium-scale 3D Swin backbone shared by all registered CT phases."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        swin_cfg = model_cfg["swin"]
        self.num_phases = int(model_cfg["num_phases"])
        self.patch_grid = tuple(int(v) for v in model_cfg["patch_grid"])
        self.token_dim = int(model_cfg["token_dim"])
        feature_size = int(swin_cfg["feature_size"])
        self.backbone = SwinTransformer(
            in_chans=1,
            embed_dim=feature_size,
            window_size=tuple(int(v) for v in swin_cfg["window_size"]),
            patch_size=tuple(int(v) for v in swin_cfg["patch_size"]),
            depths=tuple(int(v) for v in swin_cfg["depths"]),
            num_heads=tuple(int(v) for v in swin_cfg["num_heads"]),
            mlp_ratio=float(swin_cfg["mlp_ratio"]),
            qkv_bias=True,
            drop_rate=float(model_cfg["dropout"]),
            attn_drop_rate=float(swin_cfg["attention_dropout"]),
            drop_path_rate=float(swin_cfg["drop_path_rate"]),
            patch_norm=True,
            use_checkpoint=bool(swin_cfg["use_checkpoint"]),
            spatial_dims=3,
            downsample="merging",
            use_v2=False,
        )
        level_channels = (feature_size * 4, feature_size * 8, feature_size * 16)
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels, self.token_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_groups(self.token_dim), self.token_dim),
                    nn.GELU(),
                )
                for channels in level_channels
            ]
        )
        self.multiscale_fusion = nn.Sequential(
            nn.Conv3d(self.token_dim * len(level_channels), self.token_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(self.token_dim), self.token_dim),
            nn.GELU(),
            nn.Conv3d(
                self.token_dim,
                self.token_dim,
                kernel_size=3,
                padding=1,
                groups=self.token_dim,
                bias=False,
            ),
            nn.GroupNorm(_groups(self.token_dim), self.token_dim),
            nn.GELU(),
        )

    @torch.no_grad()
    def forward_calibration(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        training_states = [
            (module, bool(module.training)) for module in self.modules()
        ]
        checkpoint_states = [
            (module, bool(module.use_checkpoint))
            for module in self.modules()
            if hasattr(module, "use_checkpoint")
        ]
        try:
            self.eval()
            # Checkpointing has no memory benefit under no_grad and should not
            # install extra hooks while training graphs of these blocks are live.
            for module, _ in checkpoint_states:
                module.use_checkpoint = False
            output = self.forward(image)
        finally:
            for module, use_checkpoint in checkpoint_states:
                module.use_checkpoint = use_checkpoint
            for module, training in training_states:
                module.training = training
        return output

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 5 or int(image.shape[1]) != self.num_phases:
            raise ValueError(f"Expected [B,{self.num_phases},D,H,W], got {tuple(image.shape)}.")
        batch_size = int(image.shape[0])
        x = image.reshape(batch_size * self.num_phases, 1, *image.shape[-3:])
        hierarchy = self.backbone(x, normalize=True)
        multiscale = []
        for projection, feature in zip(self.level_projections, hierarchy[2:5]):
            feature = projection(feature)
            if tuple(feature.shape[-3:]) != self.patch_grid:
                feature = F.interpolate(
                    feature,
                    size=self.patch_grid,
                    mode="trilinear",
                    align_corners=False,
                )
            multiscale.append(feature)
        fused = self.multiscale_fusion(torch.cat(multiscale, dim=1))
        fused = fused.flatten(2).transpose(1, 2)
        base_tokens = fused.reshape(
            batch_size,
            self.num_phases,
            -1,
            self.token_dim,
        )
        return base_tokens


class LocalCrossPhaseAttention(nn.Module):
    """Cross-phase attention with a small 3D tolerance neighborhood."""

    def __init__(
        self,
        token_dim: int,
        num_phases: int,
        patch_grid: Sequence[int],
        heads: int,
        dropout: float,
        tolerance_radius: int,
        reliability_log_bias_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_phases = int(num_phases)
        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.heads = int(heads)
        self.head_dim = self.token_dim // self.heads
        self.tolerance_radius = int(tolerance_radius)
        self.reliability_log_bias_strength = float(
            reliability_log_bias_strength
        )
        if self.token_dim % self.heads:
            raise ValueError("token_dim must be divisible by heads.")
        self.offsets = tuple(
            (z, y, x)
            for z in range(-self.tolerance_radius, self.tolerance_radius + 1)
            for y in range(-self.tolerance_radius, self.tolerance_radius + 1)
            for x in range(-self.tolerance_radius, self.tolerance_radius + 1)
        )
        self.center_offset = self.offsets.index((0, 0, 0))
        self.norm = nn.LayerNorm(self.token_dim)
        self.query = nn.Linear(self.token_dim, self.token_dim)
        self.key = nn.Linear(self.token_dim, self.token_dim)
        self.value = nn.Linear(self.token_dim, self.token_dim)
        self.output = nn.Linear(self.token_dim, self.token_dim)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output_dropout = nn.Dropout(float(dropout))
        self.relative_spatial_bias = nn.Parameter(
            torch.zeros(self.heads, len(self.offsets))
        )
        self.phase_pair_bias = nn.Parameter(
            torch.zeros(self.heads, self.num_phases, self.num_phases)
        )
        noncentral_candidates = max(
            (self.num_phases - 1) * (len(self.offsets) - 1),
            1,
        )
        with torch.no_grad():
            self.relative_spatial_bias[:, self.center_offset] = math.log(
                float(noncentral_candidates)
            )
        self.register_buffer(
            "candidate_mask",
            self._build_candidate_mask(),
            persistent=False,
        )

    @staticmethod
    def _axis_slices(length: int, offset: int) -> tuple[slice, slice]:
        if offset >= 0:
            return slice(0, length - offset), slice(offset, length)
        return slice(-offset, length), slice(0, length + offset)

    @classmethod
    def _shift_grid(
        cls,
        grid: torch.Tensor,
        offset: tuple[int, int, int],
    ) -> torch.Tensor:
        output = torch.zeros_like(grid)
        destination = []
        source = []
        for length, value in zip(grid.shape[2:5], offset):
            destination_slice, source_slice = cls._axis_slices(
                int(length),
                int(value),
            )
            destination.append(destination_slice)
            source.append(source_slice)
        output[
            :,
            :,
            destination[0],
            destination[1],
            destination[2],
            :,
            :,
        ] = grid[
            :,
            :,
            source[0],
            source[1],
            source[2],
            :,
            :,
        ]
        return output

    def _build_candidate_mask(self) -> torch.Tensor:
        depth, height, width = self.patch_grid
        valid_offsets = []
        base = torch.ones(1, 1, depth, height, width, 1, 1)
        for offset in self.offsets:
            valid_offsets.append(
                self._shift_grid(base, offset)
                .reshape(depth * height * width)
                .bool()
            )
        spatial_valid = torch.stack(valid_offsets, dim=-1)
        allowed = spatial_valid[
            None,
            :,
            None,
            :,
        ].expand(
            self.num_phases,
            -1,
            self.num_phases,
            -1,
        ).clone()
        same_phase = torch.eye(self.num_phases, dtype=torch.bool)[
            :,
            None,
            :,
            None,
        ]
        non_center = torch.ones(len(self.offsets), dtype=torch.bool)
        non_center[self.center_offset] = False
        allowed &= ~(
            same_phase
            & non_center[None, None, None, :]
        )
        return allowed.reshape(
            self.num_phases,
            depth * height * width,
            self.num_phases * len(self.offsets),
        )

    def _candidate_features(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        depth, height, width = self.patch_grid
        grid = features.reshape(
            batch,
            self.num_phases,
            depth,
            height,
            width,
            self.heads,
            self.head_dim,
        )
        shifted = torch.stack(
            [
                self._shift_grid(grid, offset)
                for offset in self.offsets
            ],
            dim=5,
        )
        return (
            shifted.reshape(
                batch,
                self.num_phases,
                depth * height * width,
                len(self.offsets),
                self.heads,
                self.head_dim,
            )
            .permute(0, 4, 2, 1, 3, 5)
            .reshape(
                batch,
                self.heads,
                depth * height * width,
                self.num_phases * len(self.offsets),
                self.head_dim,
            )
        )

    def _candidate_reliability(
        self,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        batch = reliability.shape[0]
        depth, height, width = self.patch_grid
        grid = reliability.reshape(
            batch,
            self.num_phases,
            depth,
            height,
            width,
            1,
            1,
        )
        shifted = torch.stack(
            [self._shift_grid(grid, offset) for offset in self.offsets],
            dim=5,
        )
        return (
            shifted.reshape(
                batch,
                self.num_phases,
                depth * height * width,
                len(self.offsets),
            )
            .permute(0, 2, 1, 3)
            .reshape(
                batch,
                depth * height * width,
                self.num_phases * len(self.offsets),
            )
        )

    def forward(
        self,
        tokens: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, phases, token_count, channels = tokens.shape
        expected_tokens = int(torch.tensor(self.patch_grid).prod().item())
        if phases != self.num_phases or token_count != expected_tokens:
            raise ValueError(
                "Cross-phase attention input does not match configured "
                "phase count or patch grid."
            )
        normalized = self.norm(tokens)
        query = (
            self.query(normalized)
            .reshape(
                batch,
                phases,
                token_count,
                self.heads,
                self.head_dim,
            )
            .permute(0, 3, 1, 2, 4)
        )
        keys = self._candidate_features(self.key(normalized))
        values = self._candidate_features(self.value(normalized))
        candidate_reliability = None
        if reliability is not None:
            if reliability.shape != (batch, phases, token_count):
                raise ValueError(
                    "Cross-phase reliability must have shape [B,P,N]."
                )
            candidate_reliability = self._candidate_reliability(
                reliability.float().clamp(1e-4, 1.0)
            )
            values = values * candidate_reliability[
                :, None, :, :, None
            ].to(values.dtype)
        scores = torch.einsum(
            "bhpnc,bhnkc->bhpnk",
            query,
            keys,
        ).float() / self.head_dim**0.5
        pair_bias = (
            self.phase_pair_bias[:, :, :, None]
            + self.relative_spatial_bias[:, None, None, :]
        ).reshape(
            self.heads,
            self.num_phases,
            self.num_phases * len(self.offsets),
        )
        scores = scores + pair_bias[None, :, :, None, :]
        if candidate_reliability is not None:
            scores = scores + self.reliability_log_bias_strength * torch.log(
                candidate_reliability[:, None, None].clamp_min(1e-4)
            )
        scores = scores.masked_fill(
            ~self.candidate_mask[None, None, :, :, :],
            torch.finfo(scores.dtype).min,
        )
        attention = self.attention_dropout(
            torch.softmax(scores, dim=-1).to(values.dtype)
        )
        context = torch.einsum(
            "bhpnk,bhnkc->bhpnc",
            attention,
            values,
        )
        context = (
            context.permute(0, 2, 3, 1, 4)
            .reshape(batch, phases, token_count, channels)
        )
        return self.output_dropout(self.output(context))


class SpatialPropagation3D(nn.Module):
    """Propagate phase-aware features through each phase's 3D token grid."""

    def __init__(
        self,
        token_dim: int,
        num_phases: int,
        patch_grid: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_phases = int(num_phases)
        self.patch_grid = tuple(int(v) for v in patch_grid)
        self.norm = nn.LayerNorm(self.token_dim)
        self.depthwise = nn.Conv3d(
            self.token_dim,
            self.token_dim,
            kernel_size=3,
            padding=1,
            groups=self.token_dim,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(self.token_dim, self.token_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(self.token_dim * 2, self.token_dim, kernel_size=1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, phases, token_count, channels = tokens.shape
        depth, height, width = self.patch_grid
        spatial = self.norm(tokens)
        spatial = spatial.reshape(batch_size * phases, depth, height, width, channels)
        spatial = spatial.permute(0, 4, 1, 2, 3).contiguous()
        spatial = self.pointwise(self.depthwise(spatial))
        return spatial.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            phases,
            token_count,
            channels,
        )


class CrossPhaseFusionBlock(nn.Module):
    """Local-tolerant cross-phase attention followed by 3D propagation."""

    def __init__(
        self,
        token_dim: int,
        num_phases: int,
        patch_grid: Sequence[int],
        heads: int,
        dropout: float,
        tolerance_radius: int,
        reliability_log_bias_strength: float,
        cross_phase_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.num_phases = int(num_phases)
        self.cross_phase_enabled = bool(cross_phase_enabled)
        self.cross_phase: LocalCrossPhaseAttention | None = (
            LocalCrossPhaseAttention(
                token_dim,
                num_phases,
                patch_grid,
                heads,
                dropout,
                tolerance_radius,
                reliability_log_bias_strength,
            )
            if self.cross_phase_enabled
            else None
        )
        self.spatial_propagation = SpatialPropagation3D(
            token_dim,
            num_phases,
            patch_grid,
            dropout,
        )
        self.output_norm = nn.LayerNorm(int(token_dim))

    def forward(
        self,
        tokens: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cross_phase is not None:
            tokens = tokens + self.cross_phase(tokens, reliability)
        elif reliability is not None:
            tokens = tokens * reliability.to(tokens.dtype).unsqueeze(-1)
        tokens = tokens + self.spatial_propagation(tokens)
        return self.output_norm(tokens)


class CrossPhaseFusion(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], depth_override: int | None = None) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.num_phases = int(model_cfg["num_phases"])
        self.patch_grid = tuple(int(v) for v in model_cfg["patch_grid"])
        self.token_dim = int(model_cfg["token_dim"])
        tolerance_radius = int(model_cfg["interaction_tolerance_radius"])
        reliability_cfg = model_cfg.get("token_reliability", {})
        reliability_log_bias_strength = float(
            reliability_cfg.get("attention_log_bias_strength", 1.0)
        )
        ablation_cfg = model_cfg.get("ablation", {})
        self.cross_phase_enabled = bool(
            ablation_cfg.get("cross_phase_interaction", True)
        )
        token_count = int(torch.tensor(self.patch_grid).prod().item())
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, token_count, self.token_dim))
        self.phase_embedding = nn.Parameter(torch.zeros(1, self.num_phases, 1, self.token_dim))
        depth = int(depth_override if depth_override is not None else model_cfg["interaction_depth"])
        self.blocks = nn.ModuleList(
            [
                CrossPhaseFusionBlock(
                    self.token_dim,
                    self.num_phases,
                    self.patch_grid,
                    int(model_cfg["interaction_heads"]),
                    float(model_cfg["dropout"]),
                    tolerance_radius,
                    reliability_log_bias_strength,
                    self.cross_phase_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.phase_embedding, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        add_embeddings: bool = True,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if add_embeddings:
            tokens = tokens + self.position_embedding + self.phase_embedding
        for block_index, block in enumerate(self.blocks):
            # Technical reliability acts exactly once: in the first local
            # cross-phase attention block. Later interaction blocks receive the
            # already contextualized representation without another gate.
            block_reliability = reliability if block_index == 0 else None
            tokens = block(tokens, block_reliability)
        return self.output_norm(tokens)



def build_image_encoder(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the configured 3D image backbone while keeping Swin as default."""

    backbone_cfg = cfg["model"].get("backbone", {})
    if not isinstance(backbone_cfg, Mapping):
        raise ValueError("model.backbone must be a mapping.")
    kind = str(backbone_cfg.get("type", "swin")).lower().replace("-", "_")
    if kind in {"swin", "swin3d", "3d_swin"}:
        return SharedPhaseSwinEncoder3D(cfg)
    if kind in {"resnet50", "resnet_50", "3d_resnet50"}:
        return SharedPhaseResNet50Encoder3D(cfg)
    if kind in {"densenet121", "densenet_121", "3d_densenet121"}:
        return SharedPhaseDenseNet121Encoder3D(cfg)
    if kind in {"convnext_tiny", "convnexttiny", "3d_convnext_tiny"}:
        return SharedPhaseConvNeXtTinyEncoder3D(cfg)
    if kind in {"vit_small", "vitsmall", "3d_vit_small"}:
        return SharedPhaseViTSmallEncoder3D(cfg)
    if kind in {"efficientnet_b1", "efficientnetb1", "3d_efficientnet_b1"}:
        return SharedPhaseEfficientNetB1Encoder3D(cfg)
    raise ValueError(
        "Unsupported model.backbone.type: "
        f"{backbone_cfg.get('type')!r}. Supported values are "
        "swin, resnet50, densenet121, convnext_tiny, vit_small, efficientnet_b1."
    )
