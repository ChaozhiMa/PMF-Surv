from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _conv3x3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class DeterministicMaxPool3d(nn.Module):
    """Deterministic kernel-3/stride-2 max pooling with first-index tie breaking."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(
            inputs,
            (1, 1, 1, 1, 1, 1),
            mode="constant",
            value=torch.finfo(inputs.dtype).min,
        )
        depth = (inputs.shape[2] + 1) // 2
        height = (inputs.shape[3] + 1) // 2
        width = (inputs.shape[4] + 1) // 2
        pooled: torch.Tensor | None = None
        for offset_depth in range(3):
            for offset_height in range(3):
                for offset_width in range(3):
                    candidate = padded[
                        :,
                        :,
                        offset_depth : offset_depth + 2 * depth : 2,
                        offset_height : offset_height + 2 * height : 2,
                        offset_width : offset_width + 2 * width : 2,
                    ]
                    pooled = (
                        candidate
                        if pooled is None
                        else torch.where(candidate > pooled, candidate, pooled)
                    )
        if pooled is None:
            raise RuntimeError("DeterministicMaxPool3d received an empty input.")
        return pooled


class ProSEBasicBlock3D(nn.Module):
    """3D-SE residual block used by the official Lite-ProSENet model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        frame_count: int | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = _conv3x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                _conv1x1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        channel_bottleneck = max(out_channels // 2, 1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(out_channels, channel_bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(channel_bottleneck, out_channels),
            nn.Sigmoid(),
        )
        self.frame_count = int(frame_count) if frame_count is not None else None
        self.temporal_excitation = (
            nn.Sequential(
                nn.Linear(self.frame_count, 2),
                nn.ReLU(inplace=True),
                nn.Linear(2, self.frame_count),
                nn.Sigmoid(),
            )
            if self.frame_count is not None
            else None
        )

    def _channel_se(self, features: torch.Tensor) -> torch.Tensor:
        global_pool = features.mean(dim=(2, 3, 4))
        global_scale = self.channel_excitation(global_pool)
        slice_pool = features.mean(dim=(3, 4)).permute(0, 2, 1)
        slice_scale = self.channel_excitation(slice_pool).permute(0, 2, 1)
        return features * global_scale[:, :, None, None, None] * slice_scale[
            :, :, :, None, None
        ]

    def _temporal_se(self, features: torch.Tensor) -> torch.Tensor:
        if self.temporal_excitation is None or self.frame_count is None:
            return features
        if features.shape[2] != self.frame_count:
            raise RuntimeError(
                "Lite-ProSENet temporal-SE frame mismatch: "
                f"expected {self.frame_count}, received {features.shape[2]}."
            )
        global_pool = features.mean(dim=(1, 3, 4))
        global_scale = self.temporal_excitation(global_pool)
        channel_pool = features.mean(dim=(3, 4))
        channel_scale = self.temporal_excitation(channel_pool)
        return features * global_scale[:, None, :, None, None] * channel_scale[
            :, :, :, None, None
        ]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(inputs)
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = self._channel_se(outputs)
        outputs = self._temporal_se(outputs)
        return self.activation(outputs + residual)


class ProSENetImageTower(nn.Module):
    """Light 3D ResNet visual tower with channel and temporal SE."""

    def __init__(
        self,
        image_shape: Sequence[int],
        base_channels: int = 64,
        image_feature_dim: int = 25,
    ) -> None:
        super().__init__()
        channels, depth, _, _ = (int(value) for value in image_shape)
        if channels != 3:
            raise ValueError("Lite-ProSENet expects three CT input channels.")
        stem_frames = (depth + 1) // 2
        stage2_frames = (stem_frames + 1) // 2
        self.stem = nn.Sequential(
            nn.Conv3d(
                channels,
                base_channels,
                kernel_size=(7, 7, 7),
                stride=(1, 2, 2),
                padding=(3, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            DeterministicMaxPool3d(),
        )
        self.stage1 = ProSEBasicBlock3D(
            base_channels,
            base_channels,
            frame_count=stem_frames,
        )
        self.stage2 = ProSEBasicBlock3D(
            base_channels,
            2 * base_channels,
            stride=2,
            frame_count=stage2_frames,
        )
        self.stage3 = ProSEBasicBlock3D(
            2 * base_channels,
            4 * base_channels,
            stride=2,
        )
        self.stage4 = ProSEBasicBlock3D(
            4 * base_channels,
            8 * base_channels,
            stride=2,
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(8 * base_channels, image_feature_dim)
        self.activation = nn.ReLU(inplace=True)
        self.output_dim = int(image_feature_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.stem(image)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        features = self.global_pool(features).flatten(1)
        return self.activation(self.projection(features))


class ClinicalSelfAttention(nn.Module):
    """The lightweight single-head residual attention used in Lite-ProSENet."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query(tokens)
        key = self.key(tokens)
        value = self.value(tokens)
        scores = query @ key.transpose(1, 2) / math.sqrt(self.hidden_dim)
        attended = torch.softmax(scores, dim=-1) @ value
        return tokens + attended


class LiteClinicalTower(nn.Module):
    """Variable-length adaptation of the original clinical Transformer tower."""

    def __init__(
        self,
        num_continuous: int,
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
        hidden_dim: int = 24,
        output_dim: int = 96,
    ) -> None:
        super().__init__()
        self.num_continuous = int(num_continuous)
        cardinalities = tuple(int(value) for value in categorical_cardinalities) + tuple(
            int(value) for value in composite_cardinalities
        )
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, hidden_dim) for cardinality in cardinalities
        )
        self.continuous_scale = nn.Parameter(
            torch.empty(self.num_continuous, hidden_dim)
        )
        self.continuous_bias = nn.Parameter(
            torch.zeros(self.num_continuous, hidden_dim)
        )
        self.variable_count = self.num_continuous + len(cardinalities)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.variable_count + 1, hidden_dim)
        )
        self.attention1 = ClinicalSelfAttention(hidden_dim)
        self.attention2 = ClinicalSelfAttention(hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear((self.variable_count + 1) * hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = int(output_dim)
        nn.init.normal_(self.continuous_scale, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.class_token, std=0.02)

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> torch.Tensor:
        if continuous.shape[1] != self.num_continuous:
            raise RuntimeError(
                f"Expected {self.num_continuous} continuous variables, "
                f"received {continuous.shape[1]}."
            )
        continuous_tokens = (
            continuous.float().unsqueeze(-1) * self.continuous_scale.unsqueeze(0)
            + self.continuous_bias.unsqueeze(0)
        )
        discrete = torch.cat([categorical.long(), composite.long()], dim=1)
        if discrete.shape[1] != len(self.categorical_embeddings):
            raise RuntimeError(
                f"Expected {len(self.categorical_embeddings)} discrete variables, "
                f"received {discrete.shape[1]}."
            )
        discrete_tokens = [
            embedding(discrete[:, index])
            for index, embedding in enumerate(self.categorical_embeddings)
        ]
        if discrete_tokens:
            variable_tokens = torch.cat(
                [continuous_tokens, torch.stack(discrete_tokens, dim=1)],
                dim=1,
            )
        else:
            variable_tokens = continuous_tokens
        class_token = self.class_token.expand(variable_tokens.shape[0], -1, -1)
        tokens = torch.cat([class_token, variable_tokens], dim=1)
        tokens = tokens + self.position_embedding
        tokens = self.attention1(tokens)
        tokens = self.attention2(tokens)
        return self.projection(tokens.flatten(1))


class NativeSurvivalTimeHead(nn.Module):
    """The two-branch multiplicative survival-time regressor in the source code."""

    def __init__(self, input_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        self.branch_one = branch()
        self.branch_two = branch()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction_one = torch.sigmoid(self.branch_one(features)).flatten()
        prediction_two = torch.sigmoid(self.branch_two(features)).flatten()
        return {
            "prediction": prediction_one * prediction_two,
            "branch_one": prediction_one,
            "branch_two": prediction_two,
        }


def _weighted_predictions(
    raw: dict[str, torch.Tensor],
    forward_difference: dict[str, torch.Tensor],
    backward_difference: dict[str, torch.Tensor],
    raw_weight: float,
) -> dict[str, torch.Tensor]:
    difference_weight = (1.0 - raw_weight) / 2.0
    return {
        name: raw_weight * raw[name]
        + difference_weight * (
            forward_difference[name] + backward_difference[name]
        )
        for name in raw
    }


class LiteProSENet(nn.Module):
    """Three-phase, 17-variable adaptation of the official Lite-ProSENet."""

    def __init__(
        self,
        image_shape: Sequence[int],
        num_continuous: int,
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
        base_channels: int = 64,
        clinical_hidden_dim: int = 24,
        clinical_output_dim: int = 96,
        image_feature_dim: int = 25,
        head_hidden_dim: int = 24,
        raw_frame_weight: float = 0.4,
    ) -> None:
        super().__init__()
        if not 0.0 <= raw_frame_weight <= 1.0:
            raise ValueError("raw_frame_weight must be in [0, 1].")
        self.raw_frame_weight = float(raw_frame_weight)
        self.image_tower = ProSENetImageTower(
            image_shape=image_shape,
            base_channels=base_channels,
            image_feature_dim=image_feature_dim,
        )
        self.clinical_tower = LiteClinicalTower(
            num_continuous=num_continuous,
            categorical_cardinalities=categorical_cardinalities,
            composite_cardinalities=composite_cardinalities,
            hidden_dim=clinical_hidden_dim,
            output_dim=clinical_output_dim,
        )
        fused_dim = image_feature_dim + clinical_output_dim
        self.os_head = NativeSurvivalTimeHead(fused_dim, head_hidden_dim)
        self.pfs_head = NativeSurvivalTimeHead(fused_dim, head_hidden_dim)

    @staticmethod
    def _forward_difference(image: torch.Tensor) -> torch.Tensor:
        difference = image[:, :, 1:] - image[:, :, :-1]
        return torch.cat([torch.zeros_like(image[:, :, :1]), difference], dim=2)

    @staticmethod
    def _backward_difference(image: torch.Tensor) -> torch.Tensor:
        difference = image[:, :, :-1] - image[:, :, 1:]
        return torch.cat([difference, torch.zeros_like(image[:, :, :1])], dim=2)

    def _fused_features(
        self,
        image: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_tower(image)
        return torch.cat([image_features, clinical_features], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        clinical_continuous: torch.Tensor,
        clinical_categorical: torch.Tensor,
        clinical_composite: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        clinical_features = self.clinical_tower(
            clinical_continuous,
            clinical_categorical,
            clinical_composite,
        )
        raw_features = self._fused_features(image, clinical_features)
        forward_features = self._fused_features(
            self._forward_difference(image),
            clinical_features,
        )
        backward_features = self._fused_features(
            self._backward_difference(image),
            clinical_features,
        )

        os_outputs = _weighted_predictions(
            self.os_head(raw_features),
            self.os_head(forward_features),
            self.os_head(backward_features),
            self.raw_frame_weight,
        )
        pfs_outputs = _weighted_predictions(
            self.pfs_head(raw_features.detach()),
            self.pfs_head(forward_features.detach()),
            self.pfs_head(backward_features.detach()),
            self.raw_frame_weight,
        )
        return {
            "os_prediction": os_outputs["prediction"],
            "os_branch_one": os_outputs["branch_one"],
            "os_branch_two": os_outputs["branch_two"],
            "os_risk": -os_outputs["prediction"],
            "pfs_prediction": pfs_outputs["prediction"],
            "pfs_branch_one": pfs_outputs["branch_one"],
            "pfs_branch_two": pfs_outputs["branch_two"],
            "pfs_risk": -pfs_outputs["prediction"],
        }
