from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121, EfficientNetBNFeatures, ResNetFeatures, ViT


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _TokenEncoderBase(nn.Module):
    """Common input handling and calibration behavior for 3D image backbones."""

    def __init__(
        self,
        num_phases: int,
        patch_grid: Sequence[int],
        token_dim: int,
    ) -> None:
        super().__init__()
        self.num_phases = int(num_phases)
        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.token_dim = int(token_dim)
        if len(self.patch_grid) != 3 or any(value <= 0 for value in self.patch_grid):
            raise ValueError("patch_grid must contain three positive dimensions.")

    def _reshape_input(self, image: torch.Tensor) -> tuple[torch.Tensor, int]:
        if image.ndim != 5 or int(image.shape[1]) != self.num_phases:
            raise ValueError(
                f"Expected [B,{self.num_phases},D,H,W], got {tuple(image.shape)}."
            )
        batch_size = int(image.shape[0])
        return image.reshape(batch_size * self.num_phases, 1, *image.shape[-3:]), batch_size

    @torch.no_grad()
    def forward_calibration(self, image: torch.Tensor) -> torch.Tensor:
        training_states = [
            (module, bool(module.training)) for module in self.modules()
        ]
        try:
            self.eval()
            return self.forward(image)
        finally:
            for module, training in training_states:
                module.training = training


class _MultiScaleTokenEncoder(_TokenEncoderBase):
    def __init__(
        self,
        num_phases: int,
        patch_grid: Sequence[int],
        token_dim: int,
        feature_channels: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__(num_phases, patch_grid, token_dim)
        if len(feature_channels) != 3:
            raise ValueError("Multi-scale image encoders require exactly three feature levels.")
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(int(channels), self.token_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_groups(self.token_dim), self.token_dim),
                    nn.GELU(),
                )
                for channels in feature_channels
            ]
        )
        self.multiscale_fusion = nn.Sequential(
            nn.Conv3d(self.token_dim * 3, self.token_dim, kernel_size=1, bias=False),
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
            nn.Dropout3d(float(dropout)),
        )

    def _fuse_features(
        self,
        features: Sequence[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("Expected three feature levels.")
        projected: list[torch.Tensor] = []
        for projection, feature in zip(self.level_projections, features):
            if feature.ndim != 5:
                raise ValueError(f"Expected 5D feature map, got {tuple(feature.shape)}.")
            value = projection(feature)
            if tuple(value.shape[-3:]) != self.patch_grid:
                value = F.interpolate(
                    value,
                    size=self.patch_grid,
                    mode="trilinear",
                    align_corners=False,
                )
            projected.append(value)
        fused = self.multiscale_fusion(torch.cat(projected, dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        expected_tokens = self.patch_grid[0] * self.patch_grid[1] * self.patch_grid[2]
        if int(tokens.shape[1]) != expected_tokens:
            raise RuntimeError(
                f"Backbone produced {tokens.shape[1]} tokens; expected {expected_tokens}."
            )
        return tokens.reshape(
            batch_size,
            self.num_phases,
            expected_tokens,
            self.token_dim,
        )


class SharedPhaseResNet50Encoder3D(_MultiScaleTokenEncoder):
    """3D ResNet-50 applied independently to each CT phase."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_cfg = cfg["model"]
        backbone_cfg = model_cfg.get("backbone", {})
        super().__init__(
            int(model_cfg["num_phases"]),
            model_cfg["patch_grid"],
            int(model_cfg["token_dim"]),
            (512, 1024, 2048),
            float(model_cfg["dropout"]),
        )
        if bool(backbone_cfg.get("pretrained", False)):
            raise ValueError(
                "3D ResNet-50 pretrained weights are disabled for this "
                "from-scratch backbone comparison."
            )
        self.backbone = ResNetFeatures(
            model_name="resnet50",
            pretrained=False,
            spatial_dims=3,
            in_channels=1,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x, batch_size = self._reshape_input(image)
        hierarchy = self.backbone(x)
        return self._fuse_features(hierarchy[2:5], batch_size)


class SharedPhaseDenseNet121Encoder3D(_MultiScaleTokenEncoder):
    """3D DenseNet-121 applied independently to each CT phase."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_cfg = cfg["model"]
        super().__init__(
            int(model_cfg["num_phases"]),
            model_cfg["patch_grid"],
            int(model_cfg["token_dim"]),
            (512, 1024, 1024),
            float(model_cfg["dropout"]),
        )
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            pretrained=False,
        )

    def _hierarchy(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        value = x
        for name, module in self.backbone.features.named_children():
            value = module(value)
            if name in {"denseblock2", "denseblock3", "denseblock4"}:
                outputs[name] = value
        return [outputs[name] for name in ("denseblock2", "denseblock3", "denseblock4")]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x, batch_size = self._reshape_input(image)
        return self._fuse_features(self._hierarchy(x), batch_size)


class SharedPhaseEfficientNetB1Encoder3D(_MultiScaleTokenEncoder):
    """3D EfficientNet-B1 applied independently to each CT phase."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_cfg = cfg["model"]
        backbone_cfg = model_cfg.get("backbone", {})
        super().__init__(
            int(model_cfg["num_phases"]),
            model_cfg["patch_grid"],
            int(model_cfg["token_dim"]),
            (40, 112, 320),
            float(model_cfg["dropout"]),
        )
        if bool(backbone_cfg.get("pretrained", False)):
            raise ValueError(
                "MONAI does not provide 3D pretrained weights for EfficientNet-B1."
            )
        self.backbone = EfficientNetBNFeatures(
            model_name="efficientnet-b1",
            pretrained=False,
            spatial_dims=3,
            in_channels=1,
            num_classes=1,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x, batch_size = self._reshape_input(image)
        hierarchy = self.backbone(x)
        return self._fuse_features(hierarchy[2:5], batch_size)


class _DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0 or not self.training:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        random = keep + torch.rand(shape, dtype=value.dtype, device=value.device)
        return value.div(keep) * random.floor()


class _LayerNorm3D(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        channels_last = value.permute(0, 2, 3, 4, 1)
        channels_last = F.layer_norm(
            channels_last,
            (self.channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        return channels_last.permute(0, 4, 1, 2, 3).contiguous()


class _ConvNeXtBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        drop_path: float,
        layer_scale_init_value: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.pointwise1 = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(channels)
        )
        self.drop_path = _DropPath(drop_path)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = value.permute(0, 2, 3, 4, 1)
        value = self.norm(value)
        value = self.pointwise1(value)
        value = self.activation(value)
        value = self.pointwise2(value)
        value = value * self.gamma
        value = value.permute(0, 4, 1, 2, 3).contiguous()
        return residual + self.drop_path(value)


class _ConvNeXtTinyBackbone3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dims: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (3, 3, 9, 3),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("ConvNeXt-Tiny requires four stage dimensions and depths.")
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(in_channels, int(dims[0]), kernel_size=4, stride=4),
                _LayerNorm3D(int(dims[0])),
            )
        )
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    _LayerNorm3D(int(dims[index])),
                    nn.Conv3d(
                        int(dims[index]),
                        int(dims[index + 1]),
                        kernel_size=2,
                        stride=2,
                    ),
                )
            )
        total_blocks = sum(int(depth) for depth in depths)
        rates = torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for channels, depth in zip(dims, depths):
            blocks = []
            for _ in range(int(depth)):
                blocks.append(_ConvNeXtBlock3D(int(channels), rates[cursor]))
                cursor += 1
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        value = self.downsample_layers[0](value)
        value = self.stages[0](value)
        outputs.append(value)
        for index in range(1, 4):
            value = self.downsample_layers[index](value)
            value = self.stages[index](value)
            outputs.append(value)
        return outputs


class SharedPhaseConvNeXtTinyEncoder3D(_MultiScaleTokenEncoder):
    """3D ConvNeXt-Tiny applied independently to each CT phase."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_cfg = cfg["model"]
        backbone_cfg = model_cfg.get("backbone", {})
        super().__init__(
            int(model_cfg["num_phases"]),
            model_cfg["patch_grid"],
            int(model_cfg["token_dim"]),
            (192, 384, 768),
            float(model_cfg["dropout"]),
        )
        self.backbone = _ConvNeXtTinyBackbone3D(
            in_channels=1,
            dims=tuple(backbone_cfg.get("dims", (96, 192, 384, 768))),
            depths=tuple(backbone_cfg.get("depths", (3, 3, 9, 3))),
            drop_path_rate=float(backbone_cfg.get("drop_path_rate", 0.1)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x, batch_size = self._reshape_input(image)
        hierarchy = self.backbone(x)
        return self._fuse_features(hierarchy[1:4], batch_size)


class SharedPhaseViTSmallEncoder3D(_TokenEncoderBase):
    """3D ViT-Small with one token per 16x16x16 image patch."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        backbone_cfg = model_cfg.get("backbone", {})
        super().__init__(
            int(model_cfg["num_phases"]),
            model_cfg["patch_grid"],
            int(model_cfg["token_dim"]),
        )
        image_shape = tuple(int(value) for value in data_cfg["image_shape"][1:])
        patch_size = tuple(
            int(value)
            for value in backbone_cfg.get("patch_size", model_cfg.get("patch_size", (16, 16, 16)))
        )
        self.hidden_size = int(backbone_cfg.get("hidden_size", 384))
        self.backbone = ViT(
            in_channels=1,
            img_size=image_shape,
            patch_size=patch_size,
            hidden_size=self.hidden_size,
            mlp_dim=int(backbone_cfg.get("mlp_dim", self.hidden_size * 4)),
            num_layers=int(backbone_cfg.get("num_layers", 12)),
            num_heads=int(backbone_cfg.get("num_heads", 6)),
            pos_embed="conv",
            proj_type="conv",
            pos_embed_type="learnable",
            classification=False,
            dropout_rate=float(model_cfg["dropout"]),
            spatial_dims=3,
            qkv_bias=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.token_dim),
            nn.GELU(),
            nn.LayerNorm(self.token_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x, batch_size = self._reshape_input(image)
        outputs = self.backbone(x)
        tokens = outputs[0] if isinstance(outputs, tuple) else outputs
        if tokens.ndim != 3:
            raise RuntimeError(f"ViT returned an unexpected tensor shape: {tuple(tokens.shape)}")
        tokens = self.projection(tokens)
        expected_tokens = self.patch_grid[0] * self.patch_grid[1] * self.patch_grid[2]
        if int(tokens.shape[1]) != expected_tokens:
            raise RuntimeError(
                f"ViT produced {tokens.shape[1]} tokens; expected {expected_tokens}."
            )
        return tokens.reshape(
            batch_size,
            self.num_phases,
            expected_tokens,
            self.token_dim,
        )
